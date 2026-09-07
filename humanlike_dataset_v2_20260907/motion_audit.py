import argparse,json
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
m=json.loads((a.root/'manifest.json').read_text());rows=[]
human_version=next(v for v in m['versions'] if v!='baseline')
for pair in range(m['pairs']):
 for regime in ['full','partial']:
  paths={v:a.root/'trials'/f'pair_{pair:03d}_{v}_{regime}' for v in ['baseline',human_version]}
  records={v:json.loads(p.with_suffix('.json').read_text()) for v,p in paths.items()}
  if not all(r['accepted'] for r in records.values()):continue
  vals={}
  for v,path in paths.items():
   with h5py.File(path.with_suffix('.hdf5')) as f:
    d=f['data/demo_0'];act=d['actions'][:,:7];q=d['robot0_joint_pos'][:];pos=d['obs/robot0_eef_pos'][:]
    speed=np.linalg.norm(np.diff(pos,axis=0),axis=1);moving=speed[speed>1e-4]
    vals[v]=dict(steps=len(act),eef_step_cv=float(moving.std()/max(moving.mean(),1e-12)),stationary_fraction=float(np.mean(speed<=1e-4)),joint_residual_rms=float(np.sqrt(np.mean((act-q)**2))),joint_target_second_diff_rms=float(np.sqrt(np.mean(np.diff(act,n=2,axis=0)**2))))
  rows.append(dict(pair=pair,regime=regime,values=vals))
summary=dict(matched_successful_cells=len(rows),scope='Same pair/regime accepted under both policies; descriptive, no human-equivalence claim',means={v:{k:float(np.mean([r['values'][v][k] for r in rows])) for k in rows[0]['values'][v]} for v in ['baseline',human_version]})
(a.root/'motion_audit.json').write_text(json.dumps(dict(summary=summary,cells=rows),indent=2));print(json.dumps(summary,indent=2))
