import argparse,json,hashlib
from pathlib import Path
import numpy as np,h5py
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();m=json.loads((a.root/'manifest.json').read_text());human_version=next(v for v in m['versions'] if v!='baseline') if 'baseline' in m['versions'] else m['versions'][0]
s=json.loads((a.root/'summary.json').read_text())
for fname,sha_key in [('dataset_state.hdf5','dataset_sha256'),('dataset_image84.hdf5','image_dataset_sha256')]:
 path=a.root/fname
 with h5py.File(path,'r+') as f:
  for name,d in f['data'].items():
   pair=int(d.attrs['pair_id']);r=d.attrs['observability'];record=json.loads((a.root/'trials'/f'pair_{pair:03d}_{human_version}_{r}.json').read_text());stats=record['stats']
   d.attrs['source_trial']=f'trials/pair_{pair:03d}_{human_version}_{r}.hdf5'
   d.attrs['action_representation']='absolute_joint_position_plus_gripper'
   if 'source_step_index' not in d:d.create_dataset('source_step_index',data=np.arange(len(d['actions']),dtype=np.int32))
   if 'actions_absolute_joint_position' not in d:d['actions_absolute_joint_position']=d['actions']
   if 'action_dict' not in d:
    ad=d.create_group('action_dict');ad['actions_absolute_joint_position']=d['actions'];ad['actions_joint_delta']=d['actions_joint_delta']
   if m['kind']=='toolhang':
    d.attrs['label_regime']='full_visible' if r=='full' else 'partial_hidden';d.attrs['label_motion_style']=stats['variation']['motion_style'];d.attrs['label_grasp_bin_index']=m['entries'][pair]['bin']
   else:d.attrs['target_grasp_angle_deg']=stats['target_grasp_angle_deg']
  masks=f['mask']
  if 'all' not in masks:masks.create_dataset('all',data=np.asarray(list(f['data']),dtype='S'))
  if m['kind']=='toolhang':
   for r,label in [('full','full_visible'),('partial','partial_hidden')]:
    if label not in masks:masks[label]=masks[r]
    for split in ['train','valid']:
     key=split+'_'+label
     if key not in masks:masks.create_dataset(key,data=np.asarray(sorted(set(masks[r][:])&set(masks[split][:])),dtype='S'))
  assert len(f['data'])==s['selected_episodes']
 s[sha_key]=hashlib.sha256(path.read_bytes()).hexdigest()
(a.root/'summary.json').write_text(json.dumps(s,indent=2))
(a.root/'SHA256SUMS').write_text(s['dataset_sha256']+'  dataset_state.hdf5\n'+s['image_dataset_sha256']+'  dataset_image84.hdf5\n')
print(a.root,'FINALIZED',flush=True)
