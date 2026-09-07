import argparse,json
from pathlib import Path
import numpy as np,h5py
import robosuite as suite
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
rows=[]
with h5py.File(a.root/'dataset_state.hdf5') as f:
 meta=json.loads(f['data'].attrs['env_args'])
 for regime in ['full','partial']:
  name=next(n for n in f['data'] if f['data'][n].attrs['observability']==regime);d=f['data'][name]
  env=suite.make(meta['env_name'],**meta['env_kwargs'])
  try:
   env.reset();env.reset_from_xml_string(env.edit_model_xml(d.attrs['model_file']));env.sim.reset()
   first=d['states'][0]
   if hasattr(env,'load_phase2_reference_state'):env.load_phase2_reference_state(first)
   else:env.sim.set_state_from_flattened(first);env.sim.forward()
   env.robots[0].composite_controller.update_state();env.robots[0].composite_controller.reset()
   errors=[]
   for action,expected in zip(d['actions'],d['next_states']):
    env.step(action);actual=np.asarray(env.sim.get_state().flatten());errors.append(float(np.max(np.abs(actual[1:8]-expected[1:8]))))
   rows.append(dict(demo=name,regime=regime,steps=len(errors),max_joint_error_rad=max(errors),final_native_success=bool(env._check_success()),passed=max(errors)<1e-4 and bool(env._check_success())))
  finally:env.close()
(a.root/'replay_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
