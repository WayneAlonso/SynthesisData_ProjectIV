"""Exploratory sparse-group sensitivity analysis for Synthetic Data Fairness."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from fairness_lab.data.loader import DataLoader
from fairness_lab.data.preprocessing import DataPreprocessor
from fairness_lab.fairness.metrics import full_fairness_report
from fairness_lab.paths import ARTIFACTS_DIR
from fairness_lab.settings import Settings
from fairness_lab.utils.io import ensure_dir,write_dataframe,write_markdown


def collapse(s, rare): return s.astype(str).where(~s.astype(str).isin(set(rare)),'__OTHER__')

def fit_eval(prep,cfg,ds_name,seed,policy,min_group_size):
    s_first=prep.sensitive_train[cfg.sensitive_attributes[0]].reset_index(drop=True).astype(str)
    counts=s_first.value_counts(); rare=set(counts[counts<min_group_size].index) if policy=='merge' else set()
    idx=np.arange(len(s_first))
    if policy=='resample':
        rng=np.random.default_rng(seed); target=int(counts.max()); chunks=[]
        for group,ind in s_first.groupby(s_first).groups.items():
            ind=np.asarray(list(ind)); chunks.append(rng.choice(ind,size=target,replace=len(ind)<target))
        idx=np.concatenate(chunks)
    x=prep.X_train.iloc[idx]; y=prep.y_train.iloc[idx]
    from lightgbm import LGBMClassifier
    model=LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=31,subsample=.9,colsample_bytree=.9,random_state=seed,verbosity=-1,n_jobs=1)
    model.fit(x,y); pred=model.predict(prep.X_test); prob=model.predict_proba(prep.X_test)[:,1]
    base={'dataset':ds_name, 'policy':policy, 'seed':seed,'accuracy':float(accuracy_score(prep.y_test,pred)),'f1':float(f1_score(prep.y_test,pred,zero_division=0)),'auc':float(roc_auc_score(prep.y_test,prob))}
    rows=[]
    ytest=prep.y_test.reset_index(drop=True)
    for attr in cfg.sensitive_attributes:
        train_s=prep.sensitive_train[attr].reset_index(drop=True).astype(str)
        attr_counts=train_s.value_counts(); attr_rare=set(attr_counts[attr_counts<min_group_size].index) if policy=='merge' else set()
        test_s=prep.sensitive_test[attr].reset_index(drop=True).astype(str)
        if policy=='merge': test_s=collapse(test_s,attr_rare)
        fair=full_fairness_report(ytest,pd.Series(pred).reset_index(drop=True),{attr:test_s})
        stat=pd.DataFrame({'group':test_s,'y':ytest}).groupby('group')['y'].agg(['size','sum'])
        stat['negative_count']=stat['size']-stat['sum']
        for group,r in stat.iterrows():
            one=dict(base); one.update({'sensitive_attr':attr,'group':group,'n':int(r['size']),'positive_count':int(r['sum']),'negative_count':int(r['negative_count']),'positive_rate':float(r['sum']/r['size']) if r['size'] else np.nan,'sparse_flag':bool(r['size']<min_group_size or r['sum']==0 or r['negative_count']==0),'spd_range':fair.get(f'{attr}__spd_spd_range',np.nan),'eod_tpr_range':fair.get(f'{attr}__eod_eod_tpr_range',np.nan),'disparate_impact':fair.get(f'{attr}__di_disparate_impact',np.nan),'rare_train_groups':';'.join(sorted(attr_rare))})
            rows.append(one)
    return rows

def run(seeds,datasets,out_dir,min_group_size):
    settings=Settings(); out=ensure_dir(Path(out_dir)); rows=[]
    for seed in seeds:
        settings.raw["random_state"]=seed
        for ds in datasets:
            cfg=settings.dataset(ds); loaded=DataLoader(random_state=seed).load(cfg)
            if ds in {'sbo','sbo_withheld'} and len(loaded.data)>15000: loaded.data=loaded.data.sample(n=15000,random_state=seed).reset_index(drop=True)
            prep=DataPreprocessor(dataset_cfg=cfg,test_size=settings.test_size,random_state=seed).prepare(loaded.data)
            for policy in ['original','merge','resample']:
                rows.extend(fit_eval(prep,cfg,ds,seed,policy,min_group_size))
    df=pd.DataFrame(rows); write_dataframe(df,out/'sparse_group_experiments.csv')
    summary=df.groupby(['dataset','policy','sensitive_attr'],as_index=False).agg(test_groups=('group','nunique'),min_n=('n','min'),min_positive=('positive_count','min'),min_negative=('negative_count','min'),accuracy=('accuracy','mean'),f1=('f1','mean'),auc=('auc','mean'),spd_range=('spd_range','mean'),eod_tpr_range=('eod_tpr_range','mean'))
    write_dataframe(summary,out/'sparse_group_summary.csv'); write_markdown('# Sparse-group sensitivity\n\n'+summary.to_markdown(index=False,floatfmt='.4f')+'\n',out/'SPARSE_GROUPS.md'); print(f'[sparse] wrote {out}')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]); ap.add_argument('--datasets',nargs='+',default=['sbo','acs_national_2019','acs_ma_2019','acs_tx_2019']); ap.add_argument('--out-dir',default=str(ARTIFACTS_DIR/'w3_sparse')); ap.add_argument('--min-group-size',type=int,default=30); a=ap.parse_args(); run(a.seeds,a.datasets,a.out_dir,a.min_group_size)
