"""Exclude explicitly logged terminal validation from training, keeping raw trials."""
import argparse,json,hashlib
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();m=json.loads((a.root/'manifest.json').read_text());human_version=next(v for v in m['versions'] if v!='baseline') if 'baseline' in m['versions'] else m['versions'][0]
original=a.root/'dataset_state_with_validation_tail.hdf5';current=a.root/'dataset_state.hdf5'
if not original.exists():current.rename(original)
temporary=a.root/'dataset_state_trimmed_pending.hdf5'
assert not temporary.exists()
removed=0;total=0
with h5py.File(original) as src,h5py.File(temporary,'w') as out:
 for k in src:src.copy(k,out)
 for k,v in src.attrs.items():out.attrs[k]=v
 for name in out['data']:
  d=out['data'][name];pair=int(d.attrs['pair_id']);regime=d.attrs['observability']
  record=json.loads((a.root/'trials'/f'pair_{pair:03d}_{human_version}_{regime}.json').read_text())
  checks=record['stats']['validation_only_steps'];assert set(checks)=={'post_release_persistence'}
  validation=int(checks['post_release_persistence'])
  stage=next(x for x in record['stats']['stage_checks'] if x['name']=='release_retreat')
  required=int(stage['persistent_success_required']);history=stage['persistent_success_history']
  assert len(history)==validation and all(history[-required:]) and required>=2
  # Retain the first observed successful transition in the final stable run.
  tail=required-1;before=len(d['actions']);n=before-tail;assert 0<tail<n
  d.attrs['kept_validation_steps_until_first_stable_success']=validation-tail
  datasets=[]
  d.visititems(lambda k,v:datasets.append(k) if isinstance(v,h5py.Dataset) and v.ndim and v.shape[0]==before else None)
  for k in datasets:
   values=d[k][:n];attrs=dict(d[k].attrs);del d[k];ds=d.create_dataset(k,data=values)
   for key,value in attrs.items():ds.attrs[key]=value
  d['rewards'][-1]=1;d['dones'][-1]=1;d.attrs['num_samples']=n;d.attrs['excluded_terminal_validation_steps']=tail
  assert np.array_equal(d['states'][1:],d['next_states'][:-1]);total+=n;removed+=tail
 out['data'].attrs['total']=total
if current.exists():
 backup=a.root/'dataset_state_trim_all_tail_intermediate.hdf5'
 assert not backup.exists();current.rename(backup)
temporary.replace(current)
s=json.loads((a.root/'summary.json').read_text());s['raw_selected_steps']=s.get('raw_selected_steps',s['dataset_steps']);s['dataset_steps']=total;s['excluded_terminal_validation_steps']=removed;s['dataset_sha256']=hashlib.sha256(current.read_bytes()).hexdigest();(a.root/'summary.json').write_text(json.dumps(s,indent=2));print(total,removed)
