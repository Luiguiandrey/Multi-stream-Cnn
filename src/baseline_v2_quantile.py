"""
baseline_v2_quantile.py — QUANTILE REGRESSION wrapper around the tested V2 model.
Reuses baseline_v2.Model streams unchanged; only swaps the final head for a
monotone multi-quantile head and trains with pinball loss.
Outputs 7 quantiles (5/10/25/50/75/90/95) per target, matching DRF's Q05/Q95.
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch, torch.nn as nn, torch.optim as optim, numpy as np, pandas as pd, pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import importlib.util, sys

# ---- import the TESTED V2 module (streams, dataset, loaders, scalers) ----
V2DIR='/projects/oga96/cti99/BASELINE_V2'
spec=importlib.util.spec_from_file_location("bv2", f"{V2DIR}/baseline_v2.py")
bv2=importlib.util.module_from_spec(spec); spec.loader.exec_module(bv2)

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

TARGETS=bv2.TARGETS
QUANTILES=[0.05,0.10,0.25,0.50,0.75,0.90,0.95]
NQ=len(QUANTILES); QMED=QUANTILES.index(0.50)
CAP={'v_wac':1300,'ba_wac':120,'qmd_wac':1.1,'H0_wac':50,'pv_wac':45}

OUT='/projects/oga96/cti99/BASELINE_V2_QUANTILE'
CALVAL=f'{OUT}/data_used/calval_used.csv'
TESTL=f'{OUT}/data_used/test_used.csv'
EPOCHS=80; BATCH=32; LR=1e-4

# ---- Quantile model: V2 backbone, new monotone head ----
class QuantileModel(nn.Module):
    def __init__(self, na):
        super().__init__()
        base = bv2.Model(len(TARGETS), na)          # the tested V2 model
        # steal every stream + aux exactly as built
        self.chm=base.chm; self.dtm=base.dtm; self.ir=base.ir
        self.rgb=base.rgb; self.sen=base.sen; self.aux=base.aux
        # rebuild fusion up to the penultimate layer, replace final Linear
        fus=list(base.fus.children())
        self.fus_body=nn.Sequential(*fus[:-1])      # everything except final Linear(256,nt)
        in_feat=fus[-1].in_features                 # 256
        self.head=nn.Linear(in_feat, len(TARGETS)*NQ)
    def forward(self,chm,dtm,ir,rgb,sen,aux):
        f=[self.chm(chm),self.dtm(dtm),self.ir(ir),self.rgb(rgb),self.sen(sen),self.aux(aux)]
        raw=self.head(self.fus_body(torch.cat(f,1)))          # (B, nt*NQ)
        B=raw.shape[0]; raw=raw.view(B,len(TARGETS),NQ)       # (B, nt, NQ)
        med=raw[:,:,QMED]                                     # (B, nt)
        gaps=torch.nn.functional.softplus(raw)                # positive
        q=torch.zeros_like(raw)
        q[:,:,QMED]=med
        for k in range(QMED+1,NQ): q[:,:,k]=q[:,:,k-1]+gaps[:,:,k]
        for k in range(QMED-1,-1,-1): q[:,:,k]=q[:,:,k+1]-gaps[:,:,k]
        return q                                              # (B, nt, NQ), monotone

def main():
    fold=f'{OUT}/results/fold_full'; os.makedirs(fold, exist_ok=True)
    dev=torch.device('cuda')
    print("="*70); print("  V2 QUANTILE — 7 quantiles, pinball loss, fold_full"); print("="*70, flush=True)

    # ---- data: identical pipeline to V2 ----
    nfi=pd.read_csv(bv2.NFI, low_memory=False); nfi['npp']=nfi['npp'].astype(str).str.strip()
    tgt=nfi[['npp']+TARGETS].dropna(subset=TARGETS)
    cv=pd.read_csv(CALVAL); cv['npp']=cv['npp'].astype(str).str.strip()
    cv=pd.merge(cv[['npp','Ftype','Altitude']],tgt,on='npp',how='inner')
    testl=pd.read_csv(TESTL); testl['npp']=testl['npp'].astype(str).str.strip()
    tf=bv2.build_test_factors()
    te=pd.merge(testl,tf,on='npp',how='left'); te=pd.merge(te,tgt,on='npp',how='inner').dropna(subset=['Ftype','Altitude'])
    cal,val=train_test_split(cv,test_size=0.10,random_state=SEED)
    print(f"  cal={len(cal)} val={len(val)} test={len(te)}", flush=True)

    tsc={}
    for c in TARGETS:
        s=StandardScaler(); s.fit(cal[c].values.reshape(-1,1))
        if s.scale_[0]==0: s.scale_[0]=1.0
        tsc[c]=s
    print("  Fitting aux scaler...", flush=True)
    arows=[]
    for _,r in cal.iterrows():
        chm=bv2.load_band(f"{bv2.DATA}/CHM/{r['npp']}.tif",bv2.BAND_MAX['CHM'],bv2.PATCH['CHM'])
        a=bv2.chm_stats(chm)+[1. if r['Ftype']==k else 0. for k in bv2.FTYPE]+[1. if r['Altitude']==k else 0. for k in bv2.ALTI]
        arows.append(a)
    asc=StandardScaler(); asc.fit(np.array(arows,dtype=np.float32))
    pickle.dump({'tsc':tsc,'asc':asc,'quantiles':QUANTILES}, open(f'{fold}/scalers.pkl','wb'))
    print("  Scalers ready.", flush=True)

    trl=DataLoader(bv2.DS(cal,tsc,asc,'train'),BATCH,shuffle=True,num_workers=6,pin_memory=True)
    vll=DataLoader(bv2.DS(val,tsc,asc,'val'),BATCH,shuffle=False,num_workers=4,pin_memory=True)
    na=6+len(bv2.FTYPE)+len(bv2.ALTI)
    model=QuantileModel(na).to(dev)
    print(f"  aux dim {na} | params {sum(p.numel() for p in model.parameters()):,}", flush=True)

    qs=torch.tensor(QUANTILES,device=dev).view(1,1,NQ)
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2,eta_min=1e-6)

    def pinball(pred,y):                     # pred (B,nt,NQ), y (B,nt)
        e=y.unsqueeze(-1)-pred
        return torch.maximum(qs*e,(qs-1)*e)

    best=float('inf'); H=[]
    for ep in range(EPOCHS):
        model.train(); tl=0
        for b in trl:
            chm,dtm,ir,rgb,sen,aux,y=[x.to(dev) for x in b]
            opt.zero_grad(); out=model(chm,dtm,ir,rgb,sen,aux)
            m=~torch.isnan(y); loss=pinball(out,y)[m].mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item()
        model.eval(); vl=0
        with torch.no_grad():
            for b in vll:
                chm,dtm,ir,rgb,sen,aux,y=[x.to(dev) for x in b]
                out=model(chm,dtm,ir,rgb,sen,aux); m=~torch.isnan(y); vl+=pinball(out,y)[m].mean().item()
        atr,ava=tl/len(trl),vl/len(vll); H.append({'epoch':ep+1,'train':atr,'val':ava}); sch.step()
        if (ep+1)%5==0: print(f"  Ep {ep+1}/{EPOCHS} | train {atr:.5f} | val {ava:.5f}", flush=True)
        if ava<best: best=ava; torch.save(model.state_dict(), f'{fold}/best.pth')
    pd.DataFrame(H).to_csv(f'{fold}/loss_history.csv',index=False)
    plt.figure(figsize=(10,6)); dfh=pd.DataFrame(H)
    plt.plot(dfh['epoch'],dfh['train'],label='Train'); plt.plot(dfh['epoch'],dfh['val'],label='Val')
    plt.legend(); plt.grid(alpha=.3); plt.xlabel('Epoch'); plt.ylabel('Pinball loss'); plt.title('V2 Quantile')
    plt.savefig(f'{fold}/loss_curve.png',dpi=200,bbox_inches='tight')

    # ---- predict: all quantiles, unscaled, per split ----
    model.load_state_dict(torch.load(f'{fold}/best.pth'))
    from sklearn.metrics import r2_score, mean_squared_error
    def predict(meta,name):
        dl=DataLoader(bv2.DS(meta,tsc,asc,'test'),BATCH,shuffle=False,num_workers=4)
        model.eval(); P=[]; O=[]
        with torch.no_grad():
            for b in dl:
                chm,dtm,ir,rgb,sen,aux=[x.to(dev) for x in b[:6]]; y=b[6]
                o=model(chm,dtm,ir,rgb,sen,aux).cpu().numpy()      # (B,nt,NQ)
                P.append(o); O.append(y.numpy())
        P=np.vstack(P); O=np.vstack(O)                             # P (N,nt,NQ), O (N,nt)
        out=pd.DataFrame({'npp':meta['npp'].values})
        for i,c in enumerate(TARGETS):
            s=tsc[c]
            out[f'{c}_obs']=O[:,i]*s.scale_[0]+s.mean_[0]
            for qi,qv in enumerate(QUANTILES):
                out[f'{c}_q{int(qv*100):02d}']=np.clip(P[:,i,qi]*s.scale_[0]+s.mean_[0],0,CAP[c])
            out[f'{c}_pred']=out[f'{c}_q50']
        out.to_csv(f'{fold}/predictions_{name}.csv',index=False)
        print(f"\n  {name.upper()}:", flush=True)
        for i,c in enumerate(TARGETS):
            obs=out[f'{c}_obs'].values; med=out[f'{c}_q50'].values
            lo=out[f'{c}_q05'].values; hi=out[f'{c}_q95'].values
            r2=r2_score(obs,med); cov=((obs>=lo)&(obs<=hi)).mean(); w=(hi-lo).mean()
            print(f"    {c:8s} R2={r2:.3f} | 90%cov={cov*100:.1f}% width={w:.2f}", flush=True)
    for meta,name in [(cal,'cal'),(val,'val'),(te,'test')]: predict(meta,name)
    print("\n  DONE ->",fold, flush=True)

if __name__=='__main__': main()
