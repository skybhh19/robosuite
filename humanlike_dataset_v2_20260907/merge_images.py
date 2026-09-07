import argparse,json,hashlib
from pathlib import Path
import numpy as np,h5py
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
s=json.loads((a.root/'summary.json').read_text());m=json.loads((a.root/'manifest.json').read_text());human_version=next(v for v in m['versions'] if v!='baseline') if 'baseline' in m['versions'] else m['versions'][0];out=a.root/'dataset_image84.hdf5'
assert s['complete'] and not out.exists()
with h5py.File(a.root/'dataset_state.hdf5') as state,h5py.File(out.with_suffix('.partial.hdf5'),'w') as f:
 for k in state:state.copy(k,f)
 for k,v in state.attrs.items():f.attrs[k]=v
 checks=[]
 for name in f['data']:
  d=f['data'][name];pair=int(d.attrs['pair_id']);regime=d.attrs['observability']
  src=a.root/'rendered'/f'pair_{pair:03d}_{human_version}_{regime}.hdf5'
  with h5py.File(src) as images:
   reference=images['data/demo_0'];n=len(d['actions']);assert np.array_equal(d['actions'][:],reference['actions'][:n]);assert np.array_equal(d['states'][:],reference['states'][:n])
   for camera in ['agentview_image','robot0_eye_in_hand_image']:
    d['obs'].create_dataset(camera,data=reference['obs/'+camera][:n],compression='lzf',chunks=(1,84,84,3))
    arr=d['obs'][camera];assert arr.shape==(len(d['actions']),84,84,3) and arr.dtype==np.uint8
    # All frames have spatial variation; reject blank frames, not legitimate holds.
    deviations=np.asarray([np.std(frame) for frame in arr]);assert np.min(deviations)>1
    checks.append(dict(demo=name,camera=camera,min_frame_std=float(deviations.min())))
 # Pair-level train/valid isolation.
 pairs={key:{int(f['data'][x.decode()].attrs['pair_id']) for x in f['mask'][key][:]} for key in ['train','valid']}
 assert not pairs['train']&pairs['valid']
out.with_suffix('.partial.hdf5').replace(out)
s['image_dataset_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();s['image_audit']=dict(passed=True,arrays=len(checks),episode_action_state_parity=True,pair_split_isolation=True)
(a.root/'summary.json').write_text(json.dumps(s,indent=2));(a.root/'image_audit.json').write_text(json.dumps(checks,indent=2));print(out,flush=True)
