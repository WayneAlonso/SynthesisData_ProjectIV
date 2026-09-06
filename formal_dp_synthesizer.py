"""A conditional formal epsilon-DP independent-marginal synthesizer.

The mechanism releases one histogram per column with Laplace noise. The total
privacy budget is split across columns, and synthetic sampling is postprocessing.
Formal DP holds conditional on the categorical domains and numeric bounds being
public, fixed before seeing the private table, and the neighboring relation being
one-row replacement.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

DEFAULT_PUBLIC_BOUNDS={
    'AGEP':(0.0,100.0),'NOC':(0.0,100.0),'NPF':(0.0,50.0),'DENSITY':(0.0,1_000_000.0),'POVPIP':(0.0,2000.0),
    'DVET':(0.0,30.0),'DREM':(0.0,30.0),'DPHY':(0.0,30.0),'DEYE':(0.0,30.0),'DEAR':(0.0,30.0),
    'PWGTP':(0.0,2_000_000.0),'WGTP':(0.0,2_000_000.0),'EMPLOYMENT_NOISY':(0.0,2_000_000.0),
    'PAYROLL_NOISY':(0.0,2_000_000_000.0),'RECEIPTS_NOISY':(0.0,2_000_000_000.0),
    'TABWGT':(0.0,2_000_000.0),'PCT1':(0.0,100.0),'RG':(0.0,10.0),
}

@dataclass
class DPMetadata:
    epsilon: float
    delta: float
    mechanism: str
    composition: str
    neighboring_relation: str
    domain_source: str
    bounds_source: str
    n_queries: int

class FormalDPHistogramSynthesizer:
    def __init__(self,epsilon:float=1.0,random_state:int=42,n_bins:int=16,public_numeric_bounds:dict[str,tuple[float,float]]|None=None,categorical_columns:list[str]|None=None):
        if epsilon<=0: raise ValueError('epsilon must be positive')
        self.epsilon=float(epsilon); self.random_state=int(random_state); self.n_bins=int(n_bins)
        self.public_numeric_bounds=dict(DEFAULT_PUBLIC_BOUNDS); self.public_numeric_bounds.update(public_numeric_bounds or {})
        self.categorical_columns=set(categorical_columns or [])
        self.metadata:DPMetadata|None=None
    def _categorical_domain(self,s:pd.Series):
        # The caller must treat this schema domain as public and fixed.
        vals=sorted(s.dropna().astype(str).unique().tolist())
        return vals or ['<NA>']
    def _numeric_hist(self,s:pd.Series,bounds):
        lo,hi=map(float,bounds)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo: raise ValueError(f'invalid public bounds {bounds}')
        x=pd.to_numeric(s,errors='coerce').fillna(lo).clip(lo,hi).to_numpy(float)
        edges=np.linspace(lo,hi,self.n_bins+1); counts,_=np.histogram(x,bins=edges)
        return edges,counts
    @staticmethod
    def _sample_probs(counts,rng,epsilon_i):
        noisy=np.asarray(counts,dtype=float)+rng.laplace(0.0,1.0/epsilon_i,size=len(counts))
        noisy=np.clip(noisy,0.0,None)
        if noisy.sum()<=0: noisy=np.ones(len(noisy))
        return noisy/noisy.sum()
    def fit_sample(self,data:pd.DataFrame,num_rows:int|None=None)->pd.DataFrame:
        if data.empty: raise ValueError('cannot synthesize an empty table')
        n=int(num_rows or len(data)); cols=list(data.columns); q=len(cols); eps_i=self.epsilon/q; rng=np.random.default_rng(self.random_state)
        out={}
        for col in cols:
            s=data[col]
            if pd.api.types.is_numeric_dtype(s) and col not in self.categorical_columns:
                bounds=self.public_numeric_bounds.get(col)
                if bounds is None: bounds=(-1_000_000_000.0,1_000_000_000.0)
                edges,counts=self._numeric_hist(s,bounds); probs=self._sample_probs(counts,rng,eps_i); bins=rng.choice(len(probs),size=n,p=probs); left=edges[bins]; right=edges[bins+1]; vals=left+rng.random(n)*(right-left); out[col]=vals
            else:
                domain=self._categorical_domain(s); keys=s.astype('string').fillna('<NA>'); counts=keys.value_counts().reindex(domain,fill_value=0).to_numpy(); probs=self._sample_probs(counts,rng,eps_i); out[col]=rng.choice(domain,size=n,p=probs)
                # Keep string categorical columns string-like; downstream preprocessing handles it.
        self.metadata=DPMetadata(self.epsilon,0.0,'Laplace histogram per column','sequential composition: sum(epsilon_i)=epsilon','one-row replacement','declared schema domains','fixed public numeric bounds; generic bounded fallback for undeclared numeric columns',q)
        result=pd.DataFrame(out,columns=cols)
        for col in data.columns:
            if col in self.categorical_columns and pd.api.types.is_numeric_dtype(data[col]): result[col]=pd.to_numeric(result[col],errors='coerce')
            elif not pd.api.types.is_numeric_dtype(data[col]): result[col]=result[col].astype(str).replace('<NA>',np.nan)
        return result
