import pathlib,json,hashlib
import scipy.io as sio
import numpy as np
root=pathlib.Path(__file__).resolve().parent
records=[]
for name in ['synthetic3d','handwritten','Mfeat','MNIST_fea','cifar10','cifar100']:
 p=pathlib.Path.home()/'Desktop'/(name+'.mat');d=sio.loadmat(str(p))
 raw=d['data' if 'data' in d else 'X'].ravel()
 y=(d['truelabel'].ravel()[0] if 'truelabel' in d else d['Y']).ravel()
 _,y=np.unique(y,return_inverse=True)
 dst=root/'data'/name;dst.mkdir(parents=True,exist_ok=True)
 np.save(dst/'labels.npy',y)
 shapes=[]
 for i,x in enumerate(raw):
  if x.shape[0]!=len(y):x=x.T
  assert x.shape[0]==len(y) and np.isfinite(x).all()
  x=x.astype('float32');lo=x.min(0);scale=x.max(0)-lo;scale[scale==0]=1
  x=(x-lo)/scale
  np.save(dst/f'view{i}.npy',x);shapes.append(list(x.shape))
 rec={'dataset':name,'samples':len(y),'classes':len(np.unique(y)),'shapes':shapes,'preprocessing':'per-feature min-max to [0,1], all stored feature rows; no artificial missingness/noise','source':str(p)}
 if 'MissingStatus' in d:rec['MissingStatus_column_sums']=d['MissingStatus'].sum(0).tolist()
 (dst/'metadata.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');records.append(rec);print(name,shapes,flush=True)
(root/'dataset_audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
