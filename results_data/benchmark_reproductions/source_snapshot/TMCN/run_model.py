import os
os.environ['OMP_NUM_THREADS']='2'
import sys,pathlib,json,time,random,argparse,traceback
import numpy as np,torch
from sklearn.cluster import KMeans
from metrics_common import evaluate
ROOT=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--method',required=True);parser.add_argument('--dataset',required=True);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
out=ROOT/'runs'/args.method/args.dataset/str(args.seed)
if args.smoke:out=ROOT/'smoke'/args.method/args.dataset/str(args.seed)
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);np.random.seed(args.seed);random.seed(args.seed);torch.backends.cudnn.deterministic=True
dst=ROOT/'data'/args.dataset;y=np.load(dst/'labels.npy');views=[torch.from_numpy(np.load(p)) for p in sorted(dst.glob('view*.npy'))];dims=[x.shape[1] for x in views];nv=len(views);k=len(np.unique(y))
device=torch.device('cuda');start=time.time()
def log(x):print(json.dumps(x),flush=True)
def batches(bs,shuffle=True):
 idx=torch.randperm(len(y)) if shuffle else torch.arange(len(y))
 for ids in idx.split(bs):
  if shuffle and len(ids)<bs:continue
  yield [v[ids].to(device) for v in views]
try:
 if args.method in ['STCMC-UR','PGCL-DCL']:
  sys.path.insert(0,str(ROOT/'adapted'/args.method))
  from config import defaults as C
  common=dict(backbone_configs=tuple(C.MLP(input_size=(d,)) for d in dims),projector_config=None,optimizer_config=C.Optimizer(scheduler_step_size=50,scheduler_gamma=0.1))
  if args.method=='STCMC-UR':
   from models.STCMC_UR import STCMC_UR
   cfg=C.STCMC_UR(**common,fusion_config=C.Numberviews(n_views=nv),cm_config=C.ObtainEvidence(n_clusters=k),loss_config=C.Loss(funcs='LC1|LC2|LC3|Alignment|LCE',delta=20.0,n_clusters=k));model=STCMC_UR(cfg).to(device)
  else:
   from models.PGCLDCL import PGCLDCL
   cfg=C.PGCLDCL(**common,fusion_config=C.Fusion(method='weighted_mean',n_views=nv),cm_config=C.DDC(n_clusters=k),loss_config=C.Loss(funcs='ddc_1|ddc_2|ddc_3|Alignment|Clu|mi_fused',delta=20.0,n_clusters=k));model=PGCLDCL(cfg).to(device)
  epochs=2 if args.smoke else 100;bs=100
  for epoch in range(epochs):
   model.train();total=0
   for it,x in enumerate(batches(bs)):
    losses=model.train_step(x,epoch,it,len(y)//bs);val=losses['tot'].item()
    if not np.isfinite(val):raise ValueError('non-finite loss')
    total+=val
   log({'epoch':epoch+1,'epochs':epochs,'loss':total/(len(y)//bs),'seconds':time.time()-start})
  model.eval()
  with torch.no_grad():p=np.concatenate([model(x).argmax(1).cpu().numpy() for x in batches(256,False)])
 elif args.method=='TMCN':
  sys.path.insert(0,str(ROOT/'sources/TMCN'))
  from network import TMCN
  from loss import Loss
  if args.dataset!='synthetic3d':
   from network import MambaBlock
   from tmcn_fast_scan import selective_scan
   MambaBlock.selective_scan=selective_scan
  model=TMCN(nv,dims,512,128,device).to(device);opt=torch.optim.Adam(model.parameters(),lr=.0003);crit=Loss(256,.5,device).to(device)
  pre=1 if args.smoke else 200;fine=1 if args.smoke else {'cifar10':10,'cifar100':200}.get(args.dataset,100);epochs=pre+fine;bs=256
  for epoch in range(epochs):
   model.train();total=0
   for x in batches(bs):
    opt.zero_grad();xr,_,h=model(x);loss=sum(torch.nn.functional.mse_loss(a,b) for a,b in zip(x,xr))
    if epoch>=pre:
     z,S=model.TMCNF(x);loss+=sum(crit.Structure_guided_Contrastive_Loss(a,z,S) for a in h)
    if not torch.isfinite(loss):raise ValueError('non-finite loss')
    loss.backward();opt.step();total+=loss.item()
   log({'epoch':epoch+1,'epochs':epochs,'loss':total/(len(y)//bs),'seconds':time.time()-start})
  model.eval()
  with torch.no_grad():z=np.concatenate([model.TMCNF(x)[0].cpu().numpy() for x in batches(256,False)])
  p=KMeans(n_clusters=k,n_init=100,random_state=args.seed).fit_predict(z)
 elif args.method=='AICN-MLM':
  import types
  sys.path.insert(0,str(ROOT/'sources/AICN-MLM'))
  class InputData(torch.utils.data.Dataset):
   def __len__(self):return len(y)
   def __getitem__(self,i):return [v[i] for v in views],y[i],i
  loader_module=types.ModuleType('dataloader');loader_module.load_data=lambda name:(InputData(),dims,nv,len(y),k);sys.modules['dataloader']=loader_module
  import metric
  original_eval=metric.evaluate
  saved={}
  def capture(labels,pred):
   saved['pred']=pred.copy()
   return original_eval(labels.copy(),pred)
  metric.evaluate=capture
  source=(ROOT/'sources/AICN-MLM/train.py').read_text(encoding='utf-8')
  source=source.replace("Dataname = 'KUST_CE'","Dataname = 'CUSTOM'")
  source=source.replace('setup_seed(seed)\n',f'setup_seed({args.seed})\n')
  # The public loop recalculates every view's prototypes once per ordered
  # view-pair although the result depends only on the current batch/model.
  # Cache the identical tensor list once per batch (same graph and loss).
  marker='            for v in range(view):\n                for w in range(view):'
  source=source.replace(marker,'            ps = model.forward_prototype(xs)\n            for v in range(view):\n                for w in range(view):')
  source=source.replace("                    ps = model.forward_prototype(xs)\n                    '''prototype", "                    '''prototype")
  if args.smoke:
   source=source.replace('default=100)', 'default=1)').replace('default=50)', 'default=1)').replace('default=4)', 'default=1)')
  # Save exact executed adapter for audit; the published training loops are retained.
  (out/'executed_train.py').write_text(source,encoding='utf-8')
  old_argv=sys.argv;sys.argv=['train.py'];old_cwd=os.getcwd();os.chdir(out)
  scope={'__name__':'__main__','__file__':str(out/'executed_train.py')}
  try:exec(compile(source,str(out/'executed_train.py'),'exec'),scope)
  finally:sys.argv=old_argv;os.chdir(old_cwd)
  model=scope['model'];epochs=scope['args'].mse_epochs*3+scope['Total_con_epochs'];bs=scope['args'].batch_size
  # Use the published fused Z representation, with its specified KMeans variant.
  from sklearn.cluster import MiniBatchKMeans
  model.eval()
  with torch.no_grad():z=np.concatenate([torch.cat(model(x)[0],dim=1).cpu().numpy() for x in batches(256,False)])
  km=MiniBatchKMeans(n_clusters=k,batch_size=5000,n_init=100,random_state=args.seed) if len(y)>10000 else KMeans(n_clusters=k,n_init=100,random_state=args.seed)
  p=km.fit_predict(z)
 elif args.method in ['ROLL','CorreGen']:
  from run_two_view import run
  model,p,epochs,bs=run(args.method,views,k,args.seed,args.smoke,log);nv=2
 else:raise ValueError('Unsupported method')
 metrics=evaluate(y,p);np.savez_compressed(out/'predictions.npz',true_labels=y,predicted_labels=p);torch.save(model.state_dict(),out/'model.pt')
 result={'method':args.method,'dataset':args.dataset,'seed':args.seed,'status':'smoke_only' if args.smoke else 'completed','epochs':epochs,'batch_size':bs,'views':nv,'samples':len(y),'seconds':time.time()-start,'metrics':metrics}
 (out/'result.json').write_text(json.dumps(result,indent=2));log(result)
except Exception as e:
 (out/'error.txt').write_text(traceback.format_exc());log({'status':'failed','error':str(e)});raise
