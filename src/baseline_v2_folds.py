"""
baseline_v2.py — BATCH 1 REBUILD
6-stream fusion, ResNet-v1 blocks (Conv->GroupNorm->ReLU, post-activation, no attention),
LayerNorm in aux/fusion, dropout 0.1. Aux = 6 CHM stats + Ftype(3) + Altitude(3) = 12.
11K augmentation (gain/noise on RGB/IR only). GroupNorm throughout.
CHM+S2 native size; DTM/RGB/IR downsampled to 256px. Batch 32, 150 epochs, best ckpt.
Reads frozen calval_used.csv / test_used.csv (leakage already removed).
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch, random, pickle, warnings
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio, numpy as np, pandas as pd, geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
warnings.filterwarnings('ignore')

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

DATA='/projects/oga96/cti99/data/plots_100m'
NFI='/projects/oga96/cti99/data/xy_nfistats_24.csv'
GPKG='/projects/oga96/cti99/data/h5_plots_pva_dtm_assigned.gpkg'
OUT='/projects/oga96/cti99/BASELINE_V2'
CALVAL=f'{OUT}/data_used/calval_used.csv'
TESTL=f'{OUT}/data_used/test_used.csv'
EXPORT=f'{OUT}/results_folds'
TARGETS=['v_wac','ba_wac','qmd_wac','H0_wac','pv_wac']

BAND_MAX={'CHM':50.0,'DTM':500.0,'ORTHO':255.0,'SEN':10000.0}
PATCH={'CHM':500,'DTM':256,'RGB':256,'IR':256,'SEN':16}
FTYPE=['D','C','M']; ALTI=['Faible','Moyen','Élevé']
GN_GROUPS=8
BATCH=32; LR=1e-4; EPOCHS=80; DROPOUT=0.1

def load_band(path,max_val,px,bands=None):
    with rasterio.open(path) as s:
        d=s.read().astype(np.float32) if bands is None else s.read(bands).astype(np.float32)
    d=np.nan_to_num(d,nan=0.,posinf=0.,neginf=0.); d=np.clip(d,0,max_val)/max_val
    c,h,w=d.shape
    if h>px or w>px:
        sh=(h-px)//2 if h>px else 0; sw=(w-px)//2 if w>px else 0
        d=d[:,sh:sh+px,sw:sw+px]
    c,h,w=d.shape
    if h<px or w<px:
        d=np.pad(d,((0,0),((px-h)//2,px-h-(px-h)//2),((px-w)//2,px-w-(px-w)//2)),mode='constant')
    return np.ascontiguousarray(d)

def chm_stats(chm):
    cf=chm[0]; vp=cf[cf>0]
    if len(vp)>0:
        dm=maximum_filter(cf,size=5); tree=float(np.sum(((dm>2.)&(cf>2.))&(cf==dm)))
        return [np.mean(vp),np.std(vp),np.max(vp),np.percentile(vp,90),np.percentile(vp,50),tree]
    return [0.,0.,0.,0.,0.,0.]

class DS(Dataset):
    def __init__(self,meta,tsc=None,asc=None,split='train'):
        self.m=meta.reset_index(drop=True);self.tsc=tsc;self.asc=asc
        self.aug=(split=='train')
    def __len__(self): return len(self.m)
    def __getitem__(self,i):
        r=self.m.iloc[i];npp=r['npp']
        chm=load_band(f"{DATA}/CHM/{npp}.tif",BAND_MAX['CHM'],PATCH['CHM'])
        dtm=load_band(f"{DATA}/DTM/{npp}.tif",BAND_MAX['DTM'],PATCH['DTM'])
        rgb=load_band(f"{DATA}/ORTHO/{npp}.tif",BAND_MAX['ORTHO'],PATCH['RGB'],bands=[1,2,3])
        ir =load_band(f"{DATA}/ORTHO/{npp}.tif",BAND_MAX['ORTHO'],PATCH['IR'],bands=[4,3,2])
        sen=load_band(f"{DATA}/S2/{npp}.tif",BAND_MAX['SEN'],PATCH['SEN'])
        aux=chm_stats(chm)
        aux+=[1. if r['Ftype']==k else 0. for k in FTYPE]
        aux+=[1. if r['Altitude']==k else 0. for k in ALTI]
        aux=np.array(aux,dtype=np.float32)
        tv=np.array([r[c] for c in TARGETS],dtype=np.float32)
        if self.aug:
            hf,vf=random.random()>.5,random.random()>.5;kr=random.randint(0,3)
            if hf: chm,dtm,ir,rgb=[np.flip(b,2).copy() for b in [chm,dtm,ir,rgb]]
            if vf: chm,dtm,ir,rgb=[np.flip(b,1).copy() for b in [chm,dtm,ir,rgb]]
            if kr: chm,dtm,ir,rgb=[np.rot90(b,kr,(1,2)).copy() for b in [chm,dtm,ir,rgb]]
            if random.random()>.5:
                g=random.uniform(.9,1.1); rgb=np.clip(rgb*g,0,1); ir=np.clip(ir*g,0,1)
            if random.random()>.5:
                n=np.random.normal(0,.01,rgb.shape).astype(np.float32); rgb=np.clip(rgb+n,0,1); ir=np.clip(ir+n,0,1)
            if hf: sen=np.flip(sen,2).copy()
            if vf: sen=np.flip(sen,1).copy()
            if kr: sen=np.rot90(sen,kr,(1,2)).copy()
        if self.asc is not None: aux=self.asc.transform(aux.reshape(1,-1))[0]
        if self.tsc is not None:
            tv=np.array([(tv[j]-self.tsc[c].mean_[0])/self.tsc[c].scale_[0] if not np.isnan(tv[j]) else np.nan
                         for j,c in enumerate(TARGETS)],dtype=np.float32)
        return (torch.tensor(chm,dtype=torch.float32),torch.tensor(dtm,dtype=torch.float32),
                torch.tensor(np.ascontiguousarray(ir),dtype=torch.float32),
                torch.tensor(np.ascontiguousarray(rgb),dtype=torch.float32),
                torch.tensor(sen,dtype=torch.float32),torch.tensor(aux,dtype=torch.float32),
                torch.tensor(tv,dtype=torch.float32))

def GN(c): return nn.GroupNorm(GN_GROUPS,c)

class Block(nn.Module):
    """ResNet-v1 basic block: Conv->GN->ReLU->Conv->GN, add skip, ReLU."""
    def __init__(self,ic,oc,stride=1):
        super().__init__()
        self.c1=nn.Conv2d(ic,oc,3,stride,1,bias=False);self.n1=GN(oc);self.r=nn.ReLU(inplace=True)
        self.c2=nn.Conv2d(oc,oc,3,1,1,bias=False);self.n2=GN(oc)
        self.ds=None
        if stride!=1 or ic!=oc:
            self.ds=nn.Sequential(nn.Conv2d(ic,oc,1,stride,bias=False),GN(oc))
    def forward(self,x):
        idn=x; o=self.r(self.n1(self.c1(x))); o=self.n2(self.c2(o))
        if self.ds is not None: idn=self.ds(x)
        return self.r(o+idn)

def stream(ic):
    return nn.Sequential(
        nn.Conv2d(ic,64,7,2,3,bias=False),GN(64),nn.ReLU(inplace=True),nn.MaxPool2d(3,2,1),
        Block(64,64),Block(64,64),
        Block(64,128,2),Block(128,128),
        Block(128,256,2),Block(256,256),
        Block(256,512,2),Block(512,512),
        nn.AdaptiveAvgPool2d(1),nn.Flatten())

def sen_stream():
    return nn.Sequential(
        nn.Conv2d(12,32,3,1,1,bias=False),GN(32),nn.ReLU(inplace=True),
        nn.Conv2d(32,64,3,1,1,bias=False),GN(64),nn.ReLU(inplace=True),nn.MaxPool2d(2,2),
        nn.Conv2d(64,128,3,1,1,bias=False),GN(128),nn.ReLU(inplace=True),Block(128,128),
        nn.Conv2d(128,256,3,1,1,bias=False),GN(256),nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),nn.Flatten())

class Model(nn.Module):
    def __init__(self,nt,na):
        super().__init__()
        self.chm=stream(1);self.dtm=stream(1);self.ir=stream(3);self.rgb=stream(3);self.sen=sen_stream()
        self.aux=nn.Sequential(nn.Linear(na,64),nn.LayerNorm(64),nn.ReLU(inplace=True),
                               nn.Dropout(DROPOUT),nn.Linear(64,128),nn.ReLU(inplace=True))
        fin=512*4+256+128
        self.fus=nn.Sequential(
            nn.Linear(fin,1024),nn.LayerNorm(1024),nn.ReLU(inplace=True),nn.Dropout(DROPOUT),
            nn.Linear(1024,512),nn.LayerNorm(512),nn.ReLU(inplace=True),nn.Dropout(DROPOUT),
            nn.Linear(512,256),nn.LayerNorm(256),nn.ReLU(inplace=True),nn.Linear(256,nt))
    def forward(self,chm,dtm,ir,rgb,sen,aux):
        return self.fus(torch.cat([self.chm(chm),self.dtm(dtm),self.ir(ir),
                                    self.rgb(rgb),self.sen(sen),self.aux(aux)],1))

def build_test_factors():
    g=gpd.read_file(GPKG); g['npp']=g['npp'].astype(str).str.strip(); g=g.drop_duplicates('npp')
    g['comp3_r']=pd.to_numeric(g['comp3_r'],errors='coerce')
    g['meanDTM']=pd.to_numeric(g['meanDTM'],errors='coerce')
    g['Ftype']=g['comp3_r'].map({1:'D',2:'C',3:'M',4:'M'})
    def al(x):
        if pd.isna(x) or x<-100: return np.nan
        return 'Faible' if x<=200 else 'Moyen' if x<=500 else 'Élevé'
    g['Altitude']=g['meanDTM'].apply(al)
    return g[['npp','Ftype','Altitude']]

def metrics(o,p,cap):
    k=~(np.isnan(o)|np.isnan(p));o,p=o[k],p[k];p=np.clip(p,0,cap)
    rmse=np.sqrt(mean_squared_error(o,p));mn=o.mean()
    return {'R2':r2_score(o,p),'RMSE':rmse,'RMSE%':rmse/mn*100,
            'MAE':mean_absolute_error(o,p),'Bias%':(p-o).mean()/mn*100}

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True,choices=[1,2,3])
    a=ap.parse_args(); FOLD=a.fold
    os.makedirs(EXPORT,exist_ok=True)
    print("="*70);print(f"  V2 SPATIAL CV — fold {FOLD} held out");print("="*70,flush=True)
    nfi=pd.read_csv(NFI,low_memory=False);nfi['npp']=nfi['npp'].astype(str).str.strip()
    tgt=nfi[['npp']+TARGETS].dropna(subset=TARGETS)

    cv=pd.read_csv(CALVAL);cv['npp']=cv['npp'].astype(str).str.strip()
    cv=pd.merge(cv[['npp','Ftype','Altitude','fold']],tgt,on='npp',how='inner')
    print(f"  CalVal: {len(cv)} folds {dict(cv['fold'].value_counts().sort_index())}",flush=True)

    testl=pd.read_csv(TESTL);testl['npp']=testl['npp'].astype(str).str.strip()
    tf=build_test_factors()
    te=pd.merge(testl,tf,on='npp',how='left')
    te=pd.merge(te,tgt,on='npp',how='inner').dropna(subset=['Ftype','Altitude'])
    print(f"  Test: {len(te)}",flush=True)

    cal=cv[cv['fold']!=FOLD].copy(); val=cv[cv['fold']==FOLD].copy()
    print(f"  cal={len(cal)} val={len(val)} (fold {FOLD}) test={len(te)}",flush=True)

    fold=os.path.join(EXPORT,f'fold_{FOLD}');os.makedirs(fold,exist_ok=True)
    tsc={}
    for c in TARGETS:
        s=StandardScaler();s.fit(cal[c].values.reshape(-1,1))
        if s.scale_[0]==0:s.scale_[0]=1.
        tsc[c]=s
    # aux scaler
    print("  Fitting aux scaler...",flush=True)
    arows=[]
    for _,r in cal.iterrows():
        chm=load_band(f"{DATA}/CHM/{r['npp']}.tif",BAND_MAX['CHM'],PATCH['CHM'])
        a=chm_stats(chm)+[1. if r['Ftype']==k else 0. for k in FTYPE]+[1. if r['Altitude']==k else 0. for k in ALTI]
        arows.append(a)
    asc=StandardScaler();asc.fit(np.array(arows,dtype=np.float32))
    pickle.dump({'tsc':tsc,'asc':asc},open(os.path.join(fold,'scalers.pkl'),'wb'))
    print("  Scalers ready.",flush=True)

    dev=torch.device('cuda')
    trl=DataLoader(DS(cal,tsc,asc,'train'),BATCH,shuffle=True,num_workers=6,pin_memory=True)
    vll=DataLoader(DS(val,tsc,asc,'val'),BATCH,shuffle=False,num_workers=4,pin_memory=True)
    na=6+len(FTYPE)+len(ALTI)
    model=Model(len(TARGETS),na).to(dev)
    print(f"  aux dim {na} | params {sum(p.numel() for p in model.parameters()):,}",flush=True)

    crit=nn.SmoothL1Loss();opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2,eta_min=1e-6)
    best=float('inf');H=[]
    for ep in range(EPOCHS):
        model.train();tl=0
        for b in trl:
            chm,dtm,ir,rgb,sen,aux,y=[x.to(dev) for x in b]
            opt.zero_grad();out=model(chm,dtm,ir,rgb,sen,aux)
            m=~torch.isnan(y);loss=crit(out[m],y[m])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();tl+=loss.item()
        model.eval();vl=0
        with torch.no_grad():
            for b in vll:
                chm,dtm,ir,rgb,sen,aux,y=[x.to(dev) for x in b]
                out=model(chm,dtm,ir,rgb,sen,aux);m=~torch.isnan(y);vl+=crit(out[m],y[m]).item()
        atr,ava=tl/len(trl),vl/len(vll);H.append({'epoch':ep+1,'train':atr,'val':ava});sch.step()
        if (ep+1)%5==0: print(f"  Ep {ep+1}/{EPOCHS} | train {atr:.5f} | val {ava:.5f}",flush=True)
        if ava<best: best=ava; torch.save(model.state_dict(),os.path.join(fold,'best.pth'))
    pd.DataFrame(H).to_csv(os.path.join(fold,'loss_history.csv'),index=False)
    plt.figure(figsize=(10,6))
    dfh=pd.DataFrame(H);plt.plot(dfh['epoch'],dfh['train'],label='Train',lw=2)
    plt.plot(dfh['epoch'],dfh['val'],label='Val',lw=2);plt.legend();plt.grid(alpha=.3)
    plt.xlabel('Epoch');plt.ylabel('SmoothL1');plt.title(f'V2 fold {FOLD}')
    plt.savefig(os.path.join(fold,'loss_curve.png'),dpi=200,bbox_inches='tight')

    model.load_state_dict(torch.load(os.path.join(fold,'best.pth')))
    CAP={'v_wac':1300,'ba_wac':120,'qmd_wac':1.1,'H0_wac':50,'pv_wac':45}
    def predict(meta,name):
        dl=DataLoader(DS(meta,tsc,asc,'test'),BATCH,shuffle=False,num_workers=4)
        model.eval();P=[];T=[]
        with torch.no_grad():
            for b in dl:
                chm,dtm,ir,rgb,sen,aux=[x.to(dev) for x in b[:6]];y=b[6]
                o=model(chm,dtm,ir,rgb,sen,aux).cpu().numpy()
                pb=np.zeros_like(o);tb=np.zeros_like(o)
                for i,c in enumerate(TARGETS):
                    s=tsc[c];pb[:,i]=o[:,i]*s.scale_[0]+s.mean_[0]
                    tb[:,i]=y[:,i].numpy()*s.scale_[0]+s.mean_[0]
                P.append(pb);T.append(tb)
        P=np.vstack(P);T=np.vstack(T)
        out=pd.DataFrame({'npp':meta['npp'].values})
        for i,c in enumerate(TARGETS): out[f'{c}_obs']=T[:,i];out[f'{c}_pred']=P[:,i]
        out.to_csv(os.path.join(fold,f'predictions_{name}.csv'),index=False)
        print(f"\n  {name.upper()}:",flush=True)
        for i,c in enumerate(TARGETS):
            m=metrics(T[:,i],P[:,i],CAP[c])
            print(f"    {c:8s} R2={m['R2']:.3f} RMSE={m['RMSE']:.2f} Bias%={m['Bias%']:+.1f}",flush=True)
    for meta,name in [(cal,'cal'),(val,'val'),(te,'test')]: predict(meta,name)
    print("\n  DONE ->",fold,flush=True)

if __name__=='__main__': main()
