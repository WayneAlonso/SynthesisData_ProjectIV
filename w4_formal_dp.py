"""Formal-DP synthetic-data experiment for Synthetic Data Fairness.

This is separate from the legacy privacy-level noise proxy in w4_synth.py.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from w4_synth import W4Synth
from formal_dp_synthesizer import FormalDPHistogramSynthesizer
from fairness_lab.paths import ARTIFACTS_DIR
from fairness_lab.utils.io import ensure_dir,write_dataframe,write_json,write_markdown

class FormalDPW4(W4Synth):
    def run_formal(self,datasets,epsilons):
        rows=[]; metadata=[]
        for ds in datasets:
            try:
                loaded,prepared,cfg=self._load_prepared(ds)
                for eps in epsilons:
                    synth=FormalDPHistogramSynthesizer(epsilon=eps,random_state=self.settings.random_state,categorical_columns=cfg.categorical_columns)
                    synth_df=synth.fit_sample(prepared.train_df,num_rows=len(prepared.train_df))
                    quality_train=self._sdmetrics_quality(prepared.train_df,synth_df)
                    quality_test=self._sdmetrics_quality(prepared.test_df,synth_df)
                    quality={f'train_{k}':v for k,v in quality_train.items()}; quality.update({f'test_{k}':v for k,v in quality_test.items()})
                    tstr=self._tstr(synth_df,prepared,cfg)
                    rows.append({'dataset':ds,'mechanism':'formal_dp_independent_marginals','epsilon':eps,'delta':0.0,'seed':self.settings.random_state,'n_synth':len(synth_df),**quality,**tstr})
                    metadata.append({'dataset':ds,'epsilon':eps,'seed':self.settings.random_state,**synth.metadata.__dict__})
            except Exception as exc:
                rows.append({'dataset':ds,'mechanism':'formal_dp_independent_marginals','seed':self.settings.random_state,'error':str(exc)})
        df=pd.DataFrame(rows); write_dataframe(df,self.out_dir/'formal_dp_evaluation.csv'); write_json(metadata,self.out_dir/'formal_dp_metadata.json')
        write_markdown('# Formal DP W4\n\n'+(df.to_markdown(index=False,floatfmt='.4f') if not df.empty else '(empty)')+'\n',self.out_dir/'FORMAL_DP_W4.md'); print(f'[formal DP] wrote {self.out_dir}')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]); ap.add_argument('--epsilons',nargs='+',type=float,default=[0.5,1.0,2.0]); ap.add_argument('--datasets',nargs='+',default=['sbo','acs_national_2019','acs_ma_2019','acs_tx_2019']); ap.add_argument('--out-dir',default=str(ARTIFACTS_DIR/'w4_formal_dp')); a=ap.parse_args(); out=ensure_dir(Path(a.out_dir)); allp=[]
    for seed in a.seeds:
        run=FormalDPW4(out_dir=out/f'seed_{seed}',n_synth=[],privacy_levels=[]); run.settings.raw["random_state"]=seed; run.run_formal(a.datasets,a.epsilons); p=out/f'seed_{seed}'/'formal_dp_evaluation.csv';
        if p.exists():
            d=pd.read_csv(p); d['seed']=seed; allp.append(d)
    if allp: write_dataframe(pd.concat(allp,ignore_index=True),out/'formal_dp_evaluation_all_seeds.csv')
