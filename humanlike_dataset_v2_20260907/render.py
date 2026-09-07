"""Render existing pre-action simulator states; never resimulate or relabel actions."""
import argparse,json
from pathlib import Path
import numpy as np,h5py,imageio.v2 as imageio
import robosuite as suite
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args()
selected=json.loads((a.root/'selection.json').read_text())
if a.index not in selected:raise SystemExit(0)
m=json.loads((a.root/'manifest.json').read_text());human_version=next(v for v in m['versions'] if v!='baseline') if 'baseline' in m['versions'] else m['versions'][0];kw=m['env_args']['env_kwargs'].copy();kw.update(has_offscreen_renderer=True,has_renderer=False,use_camera_obs=False)
env=suite.make(m['env_args']['env_name'],**kw)
try:
 for regime in ['full','partial']:
  src=a.root/'trials'/f'pair_{a.index:03d}_{human_version}_{regime}.hdf5';out=a.root/'rendered'/src.name;out.parent.mkdir(exist_ok=True)
  if out.exists():raise FileExistsError(out)
  with h5py.File(src) as f,h5py.File(out.with_suffix('.partial.hdf5'),'w') as dest:
   f.copy('data',dest);d=dest['data/demo_0'];env.reset();env.reset_from_xml_string(env.edit_model_xml(d.attrs['model_file']));env.sim.reset()
   n=len(d['states']);ds={c:d['obs'].create_dataset(c+'_image',shape=(n,84,84,3),dtype='uint8',compression='lzf',chunks=(1,84,84,3)) for c in ['agentview','robot0_eye_in_hand']}
   writer=None
   if a.index in selected[:3]:
    review=a.root/'review';review.mkdir(exist_ok=True);writer=imageio.get_writer(review/(src.stem+'.mp4'),fps=env.control_freq/3)
   try:
    for t,state in enumerate(d['states']):
     if hasattr(env,'load_phase2_reference_state'):env.load_phase2_reference_state(state)
     else:env.sim.set_state_from_flattened(state);env.sim.forward()
     for c,arr in ds.items():arr[t]=env.sim.render(camera_name=c,width=84,height=84,depth=False)[::-1]
     if writer is not None and t%3==0:
      frame=np.concatenate([env.sim.render(camera_name=c,width=256,height=256,depth=False)[::-1] for c in ds],axis=1);writer.append_data(frame)
      if t==3*(n//6):imageio.imwrite(review/(src.stem+'.png'),frame)
   finally:
    if writer:writer.close()
   for arr in ds.values():
    assert arr.dtype==np.uint8 and arr.shape==(n,84,84,3)
    assert np.std(arr[0])>1 and np.std(arr[-1])>1
  out.with_suffix('.partial.hdf5').replace(out);print(out,flush=True)
finally:env.close()
