"""Synthetic Data Fairness W3 repeated-seed evaluation, confidence intervals and 5D Pareto."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from w3_fairness import W3Fairness
from fairness_lab.data.loader import DataLoader
from fairness_lab.data.preprocessing import DataPreprocessor
from fairness_lab.paths import ARTIFACTS_DIR
from fairness_lab.settings import Settings
from fairness_lab.utils.io import ensure_dir, write_dataframe, write_markdown


def metric_gap(row, suffix):
    vals=[]
    for c,v in row.items():
        if str(c).endswith(suffix) and not str(c).startswith('tstr_'):
            try:
                if pd.notna(v): vals.append(abs(float(v)))
            except (TypeError,ValueError): pass
    return max(vals) if vals else np.nan


def bootstrap_ci(values, seed, n_boot=2000):
    x=pd.to_numeric(pd.Series(values),errors='coerce').dropna().to_numpy(float)
    if len(x)==0: return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed)
    if len(x)==1: return (float(x[0]),float(x[0]),float(x[0]))
    means=rng.choice(x,size=(n_boot,len(x)),replace=True).mean(axis=1)
    return (float(x.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975)))


def group_diagnostics(seed, datasets, min_group_size=30):
    settings=Settings(); settings.raw["random_state"]=seed; rows=[]
    for ds in datasets:
        cfg=settings.dataset(ds); loaded=DataLoader(random_state=seed).load(cfg)
        if ds in {'sbo','sbo_withheld'} and len(loaded.data)>15000:
            loaded.data=loaded.data.sample(n=15000,random_state=seed).reset_index(drop=True)
        prep=DataPreprocessor(dataset_cfg=cfg,test_size=settings.test_size,random_state=seed).prepare(loaded.data)
        for split, sens_df, y in [('train',prep.sensitive_train,prep.y_train),('test',prep.sensitive_test,prep.y_test)]:
            yy=pd.Series(y).reset_index(drop=True)
            for attr in cfg.sensitive_attributes:
                s=sens_df[attr].reset_index(drop=True).astype(str)
                tab=pd.DataFrame({'group':s,'y':yy}).groupby('group')['y'].agg(['size','sum'])
                tab['negative_count']=tab['size']-tab['sum']
                for group,r in tab.iterrows():
                    rows.append({'seed':seed,'dataset':ds,'split':split,'sensitive_attr':attr,'group':group,
                                 'n':int(r['size']),'positive_count':int(r['sum']),'negative_count':int(r['negative_count']),
                                 'positive_rate':float(r['sum']/r['size']) if r['size'] else np.nan,
                                 'sparse_flag':bool(r['size']<min_group_size or r['sum']==0 or r['negative_count']==0)})
    return pd.DataFrame(rows)


def five_dim_pareto(df):
    if df.empty: return df
    x=df.copy(); x['spd_gap']=x.apply(lambda r:metric_gap(r,'__spd_spd_range'),axis=1); x['eod_tpr_gap']=x.apply(lambda r:metric_gap(r,'__eod_eod_tpr_range'),axis=1)
    keep=['dataset','method','lambda','accuracy','f1','auc','spd_gap','eod_tpr_gap','seed']
    keep=[c for c in keep if c in x.columns]
    x=x[keep].dropna(subset=['accuracy','f1','auc','spd_gap','eod_tpr_gap'])
    means=x.groupby(['dataset','method','lambda'],dropna=False,as_index=False)[['accuracy','f1','auc','spd_gap','eod_tpr_gap']].mean()
    out=[]
    for ds,g in means.groupby('dataset',dropna=False):
        vals=g.reset_index(drop=True)
        for i,r in vals.iterrows():
            dominates=((vals.accuracy>=r.accuracy)&(vals.f1>=r.f1)&(vals.auc>=r.auc)&(vals.spd_gap<=r.spd_gap)&(vals.eod_tpr_gap<=r.eod_tpr_gap))
            strict=((vals.accuracy>r.accuracy)|(vals.f1>r.f1)|(vals.auc>r.auc)|(vals.spd_gap<r.spd_gap)|(vals.eod_tpr_gap<r.eod_tpr_gap))
            if (dominates&strict).any(): continue
            out.append(r.to_dict())
    return pd.DataFrame(out)


def run(seeds, datasets, out_dir, min_group_size=30):
    out=ensure_dir(Path(out_dir)); all_rows=[]; all_groups=[]
    for seed in seeds:
        seed_dir=ensure_dir(out/f'seed_{seed}')
        runner=W3Fairness(out_dir=seed_dir,lambdas=[.1,.5,1.0])
        runner.settings.raw["random_state"]=int(seed)
        runner.run(datasets=datasets)
        p=seed_dir/'fairness_vs_accuracy.csv'
        if p.exists():
            df=pd.read_csv(p); df['seed']=int(seed); all_rows.append(df)
        all_groups.append(group_diagnostics(int(seed),datasets,min_group_size))
    if not all_rows: raise RuntimeError('No W3 repeated results were produced')
    raw=pd.concat(all_rows,ignore_index=True); groups=pd.concat(all_groups,ignore_index=True)
    raw['spd_gap']=raw.apply(lambda r:metric_gap(r,'__spd_spd_range'),axis=1); raw['eod_tpr_gap']=raw.apply(lambda r:metric_gap(r,'__eod_eod_tpr_range'),axis=1)
    write_dataframe(raw,out/'w3_repeated_raw.csv'); write_dataframe(groups,out/'group_diagnostics.csv')
    metric_cols=['accuracy','f1','auc','spd_gap','eod_tpr_gap']
    summary=[]
    for keys,g in raw.groupby(['dataset','method','lambda'],dropna=False):
        rec=dict(zip(['dataset','method','lambda'],keys))
        for j,c in enumerate(metric_cols):
            mean,lo,hi=bootstrap_ci(g[c],seed=1000+j+len(summary)); rec[f'{c}_mean']=mean; rec[f'{c}_ci_low']=lo; rec[f'{c}_ci_high']=hi; rec[f'{c}_std']=float(pd.to_numeric(g[c],errors='coerce').std(ddof=1)) if pd.to_numeric(g[c],errors='coerce').notna().sum()>1 else np.nan
        rec['n_seeds']=int(g['seed'].nunique()); summary.append(rec)
    summary=pd.DataFrame(summary); write_dataframe(summary,out/'w3_summary_ci.csv')
    pareto=five_dim_pareto(raw); write_dataframe(pareto,out/'pareto_frontier_5d.csv')
    sparse=groups.groupby(['dataset','split','sensitive_attr'],as_index=False).agg(groups=('group','nunique'),min_n=('n','min'),min_positive=('positive_count','min'),min_negative=('negative_count','min'),sparse_groups=('sparse_flag','sum'))
    write_dataframe(sparse,out/'sparse_group_summary.csv')
    lines=['# W3 repeated-seed evaluation','',f'- seeds: {list(seeds)}',f'- datasets: {list(datasets)}',f'- bootstrap: 2000 resamples, 95% percentile CI',f'- pareto objectives: maximize Accuracy/F1/AUC and minimize SPD/EOD-TPR gaps','', '## Summary with confidence intervals','', summary.head(30).to_markdown(index=False,floatfmt='.4f'),'', '## Five-dimensional Pareto frontier','', (pareto.to_markdown(index=False,floatfmt='.4f') if not pareto.empty else '(empty)'),'', '## Sparse-group summary','', sparse.to_markdown(index=False)]
    write_markdown('\n'.join(lines)+'\n',out/'W3_REPEATED.md')
    print(f'[W3 repeat] wrote {out}')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]); ap.add_argument('--datasets',nargs='+',default=['sbo','acs_national_2019','acs_ma_2019','acs_tx_2019']); ap.add_argument('--out-dir',default=str(ARTIFACTS_DIR/'w3_repeated')); ap.add_argument('--min-group-size',type=int,default=30); args=ap.parse_args(); run(args.seeds,args.datasets,args.out_dir,args.min_group_size)
