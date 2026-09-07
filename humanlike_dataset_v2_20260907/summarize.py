import argparse,json,hashlib
from pathlib import Path
import numpy as np,h5py
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--allow-incomplete',action='store_true');p.add_argument('--target-pairs',type=int);a=p.parse_args()
m=json.loads((a.root/'manifest.json').read_text());rows=[]
for path in sorted((a.root/'trials').glob('*.json')):rows.append(json.loads(path.read_text()))
expected={(e['pair'],v,r) for e in m['entries'] for v in m['versions'] for r in m['regimes']}
actual={(e['pair'],e['version'],e['regime']) for e in rows}
assert actual<=expected and len(actual)==len(rows)
summary=dict(complete=actual==expected,completed=len(rows),expected=len(expected),results={})
for v in m['versions']:
 summary['results'][v]={}
 for r in m['regimes']:
  ss=[e for e in rows if e['version']==v and e['regime']==r];n=len(ss)
  summary['results'][v][r]=dict(n=n,physical=sum(e['physical_success'] for e in ss),accepted=sum(e['accepted'] for e in ss),exceptions=sum(e['exception'] is not None for e in ss))
print(json.dumps(summary,indent=2),flush=True)
if not summary['complete']:
 if a.allow_incomplete:raise SystemExit(0)
 raise ValueError('incomplete collection')
lookup={(e['pair'],e['version'],e['regime']):e for e in rows}
human_version=next(v for v in m['versions'] if v!='baseline') if 'baseline' in m['versions'] else m['versions'][0]
for pair in range(m['pairs']):
 hashes={e['initial_state_sha256'] for e in rows if e['pair']==pair and 'initial_state_sha256' in e}
 assert len(hashes)==1,(pair,hashes)
summary['paired_initial_state_checks']='passed'
for v in m['versions']:
 x=np.array([int(lookup[(i,v,'full')]['physical_success'])-int(lookup[(i,v,'partial')]['physical_success']) for i in range(m['pairs'])])
 boot=np.random.default_rng(719).choice(x,size=(10000,len(x)),replace=True).mean(1)
 summary['results'][v]['full_minus_partial']=dict(point=float(x.mean()),paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),discordant=int(np.count_nonzero(x)))
selected=[i for i in range(m['pairs']) if all(lookup[(i,human_version,r)]['accepted'] for r in m['regimes'])]
eligible_pairs=list(selected)
if a.target_pairs is not None:
 assert len(selected)>=a.target_pairs,(len(selected),a.target_pairs)
 selected=selected[:a.target_pairs]
summary['eligible_pairs']=eligible_pairs
summary['selected_pairs']=selected;summary['selection_rule']=f'both {human_version} regimes accepted on the same frozen initial state; fixed pair order; no smoothness ranking or score selection'
summary['selected_episodes']=len(selected)*2
(a.root/'summary.json').write_text(json.dumps(summary,indent=2));(a.root/'selection.json').write_text(json.dumps(selected))
out=a.root/'dataset_state.hdf5'
if out.exists():raise FileExistsError(out)
valid=set(np.random.default_rng(720).permutation(selected)[:max(1,round(.2*len(selected)))])
masks={k:[] for k in ['train','valid','all','full','partial','fully_observable','partially_observable']}
with h5py.File(out,'w') as f:
 data=f.create_group('data');data.attrs['env_args']=json.dumps(m['env_args']);total=0
 for pair in selected:
  for regime in m['regimes']:
   name=f'demo_{len(data)}';src=a.root/'trials'/f'pair_{pair:03d}_{human_version}_{regime}.hdf5'
   with h5py.File(src) as source:source.copy('data/demo_0',data,name=name)
   d=data[name];n=len(d['actions']);total+=n
   if m['kind']=='threading' and 'stage' in d:
    d.move('stage','last_bounded_stage_hint');d['last_bounded_stage_hint'].attrs['warning']='Last bounded stage only; not exact per-step phase labels. Use policy stats for stage-level outcomes.'
   rewards=np.zeros(n,dtype=np.float32);rewards[-1]=1
   dones=np.zeros(n,dtype=np.int64);dones[-1]=1
   d.create_dataset('rewards',data=rewards);d.create_dataset('dones',data=dones)
   d.attrs['reward_convention']='Derived sparse terminal-success reward; dones marks recorded episode boundary.'
   assert d['actions'].shape==(n,8) and np.isfinite(d['actions'][:]).all()
   assert np.max(np.abs(d['actions'][:,:7]-d['robot0_joint_pos'][:]-d['actions_joint_delta'][:,:7]))<1e-10
   assert np.array_equal(d['states'][1:],d['next_states'][:-1])
   masks['all'].append(name);masks[regime].append(name)
   masks['fully_observable' if regime == 'full' else 'partially_observable'].append(name)
   masks['valid' if pair in valid else 'train'].append(name)
 data.attrs['total']=total
 mask=f.create_group('mask')
 for k,v in masks.items():mask.create_dataset(k,data=np.asarray(v,dtype='S'))
 f.attrs['description']=f'Human-calibrated scripted production dataset ({human_version}). Privileged-geometry expert. Paired-success subset; use summary.json for fixed-budget success rates.'
summary['dataset_steps']=total;summary['dataset_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
(a.root/'summary.json').write_text(json.dumps(summary,indent=2));print('DATASET',out,len(selected)*2,total,flush=True)
