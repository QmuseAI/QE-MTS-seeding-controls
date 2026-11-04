"""Reconstruct median-based intervals only; no search or quantum simulation.
Run from any directory. Archived raw records are read without extraction.
"""
from pathlib import Path
import numpy as np
import scipy, scipy.stats as st
import tarfile,io,re,csv,json,hashlib,sys,platform,shutil
from concurrent.futures import ThreadPoolExecutor
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
B=2000
SEED=20261102
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rng_for(tag):
 h=hashlib.sha256(f'{SEED}:{tag}'.encode()).digest()
 return np.random.default_rng(np.random.SeedSequence(np.frombuffer(h[:16],dtype='<u4').tolist()))
sources=['step2_cluster/run_main/data.tar.gz','step2_cluster/run_ext/data.tar.gz','step2_cluster/run_ext/analysis_ext/gaps_ext.csv','step2_cluster/comment/out/table_gap_perN.csv','step2_cluster/code/analyze_ext.py','step2_cluster/comment/out/comment_numbers.json']
manifest={'B':B,'seed':SEED,'rng':'PCG64; SHA256-derived separate stream per arm and N; separate stream per window','python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'inputs':{p:digest(ROOT/p) for p in sources},'script_sha256':digest(Path(__file__))}
(OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with open(ROOT/sources[2]) as f: grows=list(csv.DictReader(f))
with open(ROOT/sources[3]) as f: trows=list(csv.DictReader(f));tfields=list(trows[0])
keys={(r['arm'],int(r['N'])) for r in grows}|{(r['ref'],int(r['N'])) for r in grows}
for r in trows:
 n=int(r['N']);keys.add(('A0',n))
 for c in tfields:
  if c.startswith('D_') and not c.endswith(('_lo','_hi')) and c!='D_paper' and r[c]:keys.add((c[2:],n))
raw={}; origins={}
for fn in sources[:2]:
 with tarfile.open(ROOT/fn, mode='r|gz') as tf:
  for m in tf:
   match=re.match(r'(.+)_N(\d+)\.txt$',Path(m.name).name)
   if not match or not m.isfile() or m.size<4:continue
   k=(match[1],int(match[2]))
   if k not in keys:continue
   buf=tf.extractfile(m).read();a=np.loadtxt(io.BytesIO(buf),dtype=np.int64,ndmin=2)
   a=a[np.lexsort((a[:,1],a[:,0]))];ids=np.unique(a[:,0]);counts=[np.sum(a[:,0]==r) for r in ids]
   assert len(set(counts))==1,(k,counts)
   v=np.stack([a[a[:,0]==r,2] for r in ids]);assert v.shape==(100,100),(k,v.shape)
   raw[k]=v;origins[f'{k[0]}:{k[1]}']={'archive':fn,'member':m.name,'sha256':hashlib.sha256(buf).hexdigest(),'shape':list(v.shape)}
assert keys<=raw.keys(),keys-raw.keys()
(OUT/'record_manifest.json').write_text(json.dumps(origins,indent=2),encoding='utf-8')
print(f'Loaded {len(raw)} arm-size datasets; bootstrap B={B}',flush=True)
def calc(k):
 v=raw[k];lo=np.where(v<0,-v+1,np.maximum(v,1)).astype(float);hi=np.where(v<0,np.inf,np.maximum(v,1)).astype(float)
 qlo=np.median(np.median(lo,axis=1));qhi=np.median(np.median(hi,axis=1));assert qlo==qhi,(k,qlo,qhi)
 g=rng_for(f'arm:{k[0]}:{k[1]}');bl=np.empty(B);bh=np.empty(B)
 for start in range(0,B,50):
  stop=min(start+50,B);nb=stop-start;ir=g.integers(0,100,(nb,100));ic=g.integers(0,100,(nb,100,100))
  bl[start:stop]=np.median(np.median(lo[ir[:,:,None],ic],axis=2),axis=1)
  bh[start:stop]=np.median(np.median(hi[ir[:,:,None],ic],axis=2),axis=1)
 assert np.isfinite(bh).all(),k
 return k,(qlo,bl,bh)
res={}
with ThreadPoolExecutor(max_workers=4) as pool:
 for i,(k,v) in enumerate(pool.map(calc,sorted(raw))):
  res[k]=v
  if i%20==0:print(f'Completed {i+1}/{len(raw)}: {k}',flush=True)
np.savez_compressed(OUT/'bootstrap_q50.npz',**{f'{a}__N{n}__{kind}':val for (a,n),(q,l,h) in res.items() for kind,val in [('point',np.array(q)),('lower',l),('upper',h)]})
costs={'A1':lambda n:0,'A2k':lambda n:1e3,'A2':lambda n:1e5,'A2m':lambda n:1e6,'A3':lambda n:1e7,'A5':lambda n:1e5,'A6':lambda n:1e7,'A8':lambda n:1000*n*n}
for a in ['M2m_k1','P2m_d8']:costs[a]=lambda n:1e6
for a in ['M3_k1','P3_d8']:costs[a]=lambda n:1e7
def gap(a,ref,n,cost=0):
 q,l,h=res[a,n];qr,lr,hr=res[ref,n];c=costs[a](n) if cost else 0
 return np.log10(q+c)-np.log10(qr),np.log10(l+c)-np.log10(hr),np.log10(h+c)-np.log10(lr)
diagnostics=[]
for r in grows:
 d,l,h=gap(r['arm'],r['ref'],int(r['N']),int(r['seed_cost_counted']));ci=np.percentile(h,[2.5,97.5]);cil=np.percentile(l,[2.5,97.5])
 # conservative envelope of interval endpoints over unknown completion times
 r.update(D=d,lo=min(ci[0],cil[0]),hi=max(ci[1],cil[1]))
 diagnostics.append({'arm':r['arm'],'ref':r['ref'],'N':int(r['N']),'cost':int(r['seed_cost_counted']),'unknown_draws':int(np.sum(l!=h)),'endpoint_delta':float(np.max(abs(ci-cil)))})
for r in trows:
 n=int(r['N']);r['log10_Q50_A0']=np.log10(res['A0',n][0])
 for col in tfields:
  if col.startswith('D_') and not col.endswith(('_lo','_hi')) and col!='D_paper' and r[col]:
   d,l,h=gap(col[2:],'A0',n);r[col]=d;r[col+'_lo']=np.percentile(l,2.5);r[col+'_hi']=np.percentile(h,97.5)
def writecsv(name,rows,fields=None):
 with open(OUT/name,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
writecsv('gaps_ext.csv',grows);writecsv('table_gap_perN.csv',trows,tfields);writecsv('censoring_diagnostics.csv',diagnostics)
windows=[]
combos=sorted({(r['arm'],r['ref'],int(r['seed_cost_counted'])) for r in grows}|{(k[0],'A0',0) for k in raw if not k[0].startswith(('H0','H2','I')) and k[0]!='A0'})
for a,ref,cost in combos:
 for low,high in [(27,33),(27,37),(38,40),(38,42),(38,48)]:
  ns=[n for n in range(low,high+1) if (a,n) in res and (ref,n) in res]
  if not ns:continue
  g=rng_for(f'window:{a}:{ref}:{cost}:{ns}');dl=[];dh=[];pts=[]
  for n in ns:
   d,l,h=gap(a,ref,n,cost);idx=g.integers(B,size=B);dl.append(l[idx]);dh.append(h[idx]);pts.append(d)
  l=np.mean(dl,axis=0);h=np.mean(dh,axis=0)
  windows.append(dict(arm=a,ref=ref,cost=cost,requested_lo=low,requested_hi=high,N=','.join(map(str,ns)),mean=np.mean(pts),lo=np.percentile(l,2.5),hi=np.percentile(h,97.5),unknown_draws=int(np.sum(l!=h)),endpoint_delta=float(np.max(abs(np.percentile(l,[2.5,97.5])-np.percentile(h,[2.5,97.5]))))))
writecsv('window_intervals.csv',windows)
print('DONE: immutable-source candidate outputs written; no manuscript edits.',flush=True)
