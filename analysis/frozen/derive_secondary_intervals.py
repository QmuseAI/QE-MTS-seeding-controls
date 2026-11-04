"""Reproduce A0 slope and parity intervals from the frozen Q50 draws."""
from pathlib import Path
import numpy as np,json
o=Path(__file__).resolve().parent;d=np.load(o/'bootstrap_q50.npz');out={}
for low,high in [(27,37),(38,48)]:
 ns=np.arange(low,high+1);x=ns-ns.mean();slope=x/(x@x)
 q=np.array([float(d[f'A0__N{n}__point']) for n in ns])
 lo=np.array([np.log10(d[f'A0__N{n}__lower']) for n in ns]);hi=np.array([np.log10(d[f'A0__N{n}__upper']) for n in ns])
 odd=ns%2==1;w=odd/odd.sum()-(~odd)/(~odd).sum();w=w-x*(w@x)/(x@x)
 for name,c in [('slope',slope),('parity',w)]:
  lb=np.sum(np.where(c[:,None]>=0,c[:,None]*lo,c[:,None]*hi),axis=0)
  ub=np.sum(np.where(c[:,None]>=0,c[:,None]*hi,c[:,None]*lo),axis=0)
  vals=[float(c@np.log10(q)),float(np.percentile(lb,2.5)),float(np.percentile(ub,97.5))]
  if name=='slope':vals=[10**v for v in vals]
  out[f'{name}_{low}_{high}']=vals
(o/'derived_bootstrap.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
