import argparse,json
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();m=json.loads((a.root/'manifest.json').read_text());human_version=next(v for v in m['versions'] if v!='baseline') if 'baseline' in m['versions'] else m['versions'][0]
paired_profiles=0;modified_selected=0;s=json.loads((a.root/'summary.json').read_text())
for pair in range(m['pairs']):
 records={r:json.loads((a.root/'trials'/f'pair_{pair:03d}_{human_version}_{r}.json').read_text()) for r in ['full','partial']}
 stats=[records[r]['stats'] for r in ['full','partial']]
 if m['kind']=='threading':profiles=[x.get('human_motion_v2',{}) for x in stats]
 else:profiles=[{k:v for k,v in x.get('variation',{}).items() if k.startswith('human_')} for x in stats]
 common=set(profiles[0])&set(profiles[1])
 for k in common:assert profiles[0][k]==profiles[1][k],(pair,k)
 paired_profiles+=bool(common)
 if pair in s['selected_pairs']:
  for r in ['full','partial']:
   hp=a.root/'trials'/f'pair_{pair:03d}_{human_version}_{r}.hdf5'
   with h5py.File(hp) as h:
    x=h['data/demo_0/actions'][:];assert np.isfinite(x).all()
   bp=a.root/'trials'/f'pair_{pair:03d}_baseline_{r}.hdf5'
   if bp.exists():
    with h5py.File(bp) as b:
     y=b['data/demo_0/actions'][:];assert x.shape!=y.shape or not np.allclose(x,y,atol=1e-8,rtol=0)
   modified_selected+=1
result=dict(passed=True,paired_profiles_compared=paired_profiles,selected_episodes_with_changed_action_stream=modified_selected,expected_selected=s['selected_episodes'],all_selected_finite=True)
assert modified_selected==s['selected_episodes'];(a.root/'integrity_audit.json').write_text(json.dumps(result,indent=2));print(result)
