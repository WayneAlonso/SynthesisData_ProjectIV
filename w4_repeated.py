"""Synthetic Data Fairness W4 repeated-seed stability evaluation."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from w4_synth import W4Synth
from fairness_lab.paths import ARTIFACTS_DIR
from fairness_lab.utils.io import ensure_dir,write_dataframe,write_markdown

def run(seeds,datasets,out_dir,synthesizers,privacy_levels,use_gpu=False):
    out=ensure_dir(Path(out_dir)); chunks=[]
    for seed in seeds:
        seed_dir=ensure_dir(out/f'seed_{seed}')
        runner=W4Synth(out_dir=seed_dir,n_synth=synthesizers,privacy_levels=privacy_levels,use_gpu=use_gpu)
        runner.settings.raw["random_state"]=int(seed)
        runner.run(datasets=datasets)
        p=seed_dir/'synth_evaluation.csv'
        if p.exists():
            df=pd.read_csv(p); df['seed']=int(seed); chunks.append(df)
    if not chunks: raise RuntimeError('No W4 repeated results were produced')
    raw=pd.concat(chunks,ignore_index=True); write_dataframe(raw,out/'w4_repeated_raw.csv')
    metric_cols=[c for c in ['trtr_accuracy','tstr_accuracy','tstr_f1','tstr_auc','train_correlation_similarity','test_correlation_similarity','train_ks_mean','test_ks_mean'] if c in raw.columns]
    summary=raw.groupby(['dataset','synthesizer','privacy_level'],as_index=False)[metric_cols].agg(['mean','std']).reset_index()
    summary.columns=['_'.join([str(x) for x in c if str(x)!='']) if isinstance(c,tuple) else c for c in summary.columns]
    write_dataframe(summary,out/'w4_summary_mean_std.csv')
    lines=['# W4 repeated-seed stability','',f'- seeds: {list(seeds)}',f'- datasets: {list(datasets)}',f'- synthesizers: {list(synthesizers)}',f'- privacy levels: {list(privacy_levels)}','',summary.head(80).to_markdown(index=False,floatfmt='.4f')]
    write_markdown('\n'.join(lines)+'\n',out/'W4_REPEATED.md'); print(f'[W4 repeat] wrote {out}')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]); ap.add_argument('--datasets',nargs='+',default=['sbo','acs_national_2019','acs_ma_2019','acs_tx_2019']); ap.add_argument('--synthesizers',nargs='+',default=['ctgan','tvae','gaussian_copula','copula_gan']); ap.add_argument('--privacy-levels',nargs='+',type=float,default=[0.0,0.1,0.3]); ap.add_argument('--out-dir',default=str(ARTIFACTS_DIR/'w4_repeated')); ap.add_argument('--use-gpu',action='store_true'); a=ap.parse_args(); run(a.seeds,a.datasets,a.out_dir,a.synthesizers,a.privacy_levels,a.use_gpu)
