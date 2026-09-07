"""Frozen, fixed-budget baseline/human-inspired Full/Partial pilot."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
import numpy as np
import h5py
from human_motion import patch_threading


def safe(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 raise TypeError(type(v).__name__)

def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,default=safe,indent=2));tmp.replace(p)

def digest(x):return hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()

class Capture:
 def __init__(self,env):
  self.env=env;self.record_joint_position_fields=True;self.human_residual_gain=None;self.human_command_alpha=None;self.clear()
 def clear(self):
  self.actions=[];self.states=[];self.q=[];self.observations=[];self.stages=[];self.stage='unknown';self.xml=None;self.human_previous_action=None
 def __getattr__(self,k):return getattr(self.env,k)
 @property
 def unwrapped(self):return self.env
 @property
 def t(self):return len(self.actions)
 def reset(self):self.clear();return self.env.reset()
 def step(self,action):
  if self.xml is None:self.xml=self.sim.model.get_xml()
  action=np.asarray(action).copy()
  if self.human_residual_gain is not None and self.stage in ('aim_continuous','lift_arc'):
   current=np.asarray(self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes])
   low,high=self.env.action_spec
   action[:7]=np.clip(current+self.human_residual_gain*(action[:7]-current),np.asarray(low)[:7],np.asarray(high)[:7])
   if self.human_previous_action is not None:
    alpha=self.human_command_alpha
    action[:7]=alpha*action[:7]+(1-alpha)*self.human_previous_action[:7]
   self.human_previous_action=action.copy()
  else:self.human_previous_action=None
  obs=self.env._get_observations(force_update=True)
  self.observations.append({k:np.asarray(v).copy() for k,v in obs.items() if not k.endswith('_image')})
  self.states.append(np.asarray(self.sim.get_state().flatten()).copy())
  self.q.append(np.asarray(self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes]).copy())
  self.actions.append(np.asarray(action).copy());self.stages.append(self.stage)
  return self.env.step(action)

def components(kind,seed):
 if kind=='threading':
  from robosuite.scripts.collect_threading_balanced_initial_states import make_env,sample_initial_state,restore_initial_state
  from robosuite.scripts.collect_threading_scripted_grasp_angle import make_controller_config,ThreadingScriptedPolicy
  cfg=make_controller_config('Panda','joint_position')
  env=make_env('Threading_D06_Hard',cfg,seed,1000)
  kwargs=dict(robots=['Panda'],controller_configs=cfg,ignore_done=True,use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,use_object_obs=True,horizon=1000)
  return env,dict(env_name='Threading_D06_Hard',type=1,env_kwargs=kwargs)
 from robosuite.scripts.collect_tool_hang_balanced_state_retries import make_env
 from robosuite.scripts.collect_tool_hang_wrench_joint import make_controller_config
 meta=dict(env_name='ToolHangWrenchOnly',type=1,env_kwargs=dict(robots=['Panda'],controller_configs=make_controller_config('Panda','joint_position'),initialization_noise=None,ignore_done=True,use_camera_obs=False,use_object_obs=True,has_renderer=False,has_offscreen_renderer=False,camera_names=['agentview','robot0_eye_in_hand'],horizon=700,hard_reset=False))
 return make_env(seed,84,84,True,'joint_position'),meta

def prepare(a):
 env,meta=components(a.kind,a.seed)
 try:
  if a.kind=='threading':
   from robosuite.scripts.collect_threading_balanced_initial_states import sample_initial_state
   entries=[dict(pair=i,initial=sample_initial_state(env,i,i,0),bin=i%5) for i in range(a.pairs)]
  else:
   from robosuite.scripts.collect_tool_hang_balanced_state_retries import generate_state_pool
   pool,_=generate_state_pool(env,a.pairs,a.seed+1)
   entries=[dict(pair=i,initial=e['reset_variation'],bin=e['grasp_bin_index'],style=e['motion_style']) for i,e in enumerate(pool)]
 finally:env.close()
 versions=[a.human_version] if a.production else ['baseline',a.human_version]
 write(a.root/'manifest.json',dict(kind=a.kind,seed=a.seed,pairs=a.pairs,entries=entries,env_args=meta,versions=versions,regimes=['full','partial'],attempts_per_cell=1,protocol='frozen_reset_paired_fixed_budget',production=bool(a.production)))

def trial(a,m,e,version,regime):
 target=a.root/a.output_dir/f"pair_{e['pair']:03d}_{version}_{regime}"
 if target.with_suffix('.json').exists():raise FileExistsError(target)
 target.parent.mkdir(parents=True,exist_ok=True)
 seed=m['seed']+10000+e['pair']+1000000*a.attempt
 env,_=components(a.kind,m['seed']);cap=Capture(env)
 stats={};success=False;accepted=False;error=None
 try:
  if a.kind=='threading':
   from robosuite.scripts.collect_threading_balanced_initial_states import restore_initial_state,PANDA_JOINT_MIN,PANDA_JOINT_MAX
   from robosuite.scripts.collect_threading_scripted_grasp_angle import ThreadingScriptedPolicy
   cls=ThreadingScriptedPolicy if version=='baseline' else patch_threading()
   action_noise_std = .01 if version == 'human_v8' else .02
   p=cls(np.random.RandomState(seed),action_noise_std=action_noise_std,control_mode='joint_position');p.human_seed=seed
   if version in ('human_v3','human_v4','human_v5','human_v6','human_v7','human_v8'):
    from human_motion import profile
    command_profile=profile(seed,'command_lead');cap.human_residual_gain=command_profile['residual_gain'];cap.human_command_alpha=command_profile['command_alpha']
   p.rollout(
    cap,
    target_grasp_angle=float(
     (95.0 + (0.4 if version == 'human_v6' else 1.0) * e['bin'])
     if version in ('human_v6','human_v7','human_v8') and regime == 'full'
     else ((96 if regime == 'full' else 80) + e['bin'])
    ),
    full_quality_mode=(version in ('human_v5','human_v6','human_v7','human_v8') and regime == 'full'),
    post_reset_callback=lambda c:restore_initial_state(c,e['initial']),
   )
   stats=p.stats[-1];success=bool(stats['policy_success'] and env._check_success())
   q=np.vstack([cap.q,np.asarray(env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes])]);margin=float(np.min(np.minimum(q-PANDA_JOINT_MIN,PANDA_JOINT_MAX-q)))
   contact=stats.get('pregrasp_contact_check',{});accepted=success and margin>=.05 and contact.get('passed',False) and not contact.get('detected',True)
   stats['joint_margin_rad']=margin
  else:
   from robosuite.scripts.collect_tool_hang_wrench_joint import GeometricJointPolicy,ROBUST_JOINT_OPTIONS,VideoRecorder,collection_acceptance
   lo=(-.005 if regime=='full' else .045)+e['bin']*.002
   opts=dict(seed=seed,variation=True,grasp_profile='full_visible' if regime=='full' else 'partial_hidden',robot_start_mode='threading_continuous',motion_style=e['style'],grasp_offset_range=(lo,lo+.002),controller_backend='joint_position',threading_pregrasp_frames=50,insertion_correction_steps=8,joint_line_correction_gain=.65)
   opts.update(ROBUST_JOINT_OPTIONS);p=GeometricJointPolicy(**opts);p.human_enabled=version!='baseline'
   success,stats=p.rollout(cap,VideoRecorder(None),reset_variation_override=e['initial'],allow_reset_resample=False)
   checks,accepted=collection_acceptance(success,stats,'any',True);stats['acceptance_checks']=checks
  if not cap.actions:raise RuntimeError('no recorded actions')
 except Exception as ex:error=dict(type=type(ex).__name__,message=str(ex),traceback=traceback.format_exc())
 finally:
  row=dict(pair=e['pair'],version=version,regime=regime,attempt=a.attempt,seed=seed,physical_success=bool(success),accepted=bool(accepted),exception=error,steps=len(cap.actions),stats=stats)
  if cap.actions:
   actions=np.asarray(cap.actions);q=np.asarray(cap.q);states=np.asarray(cap.states)
   if a.kind=='toolhang':
    for stage in stats.get('stage_checks',[]):
     for j in range(stage['start_step'],min(stage['end_step'],len(cap.stages))):cap.stages[j]=stage['name']
   row['initial_state_sha256']=digest(states[0]);row['q_sha256']=digest(q);row['action_sha256']=digest(actions)
   with h5py.File(target.with_suffix('.hdf5'),'w') as f:
    g=f.create_group('data');g.attrs['env_args']=json.dumps(m['env_args']);d=g.create_group('demo_0')
    for k,v in dict(actions=actions,states=states,robot0_joint_pos=q,actions_joint_delta=np.column_stack([actions[:,:7]-q,actions[:,7]]),next_states=np.vstack([states[1:],np.asarray(env.sim.get_state().flatten())])).items():d.create_dataset(k,data=v)
    o=d.create_group('obs');keys=set.intersection(*(set(x) for x in cap.observations))
    for k in sorted(keys):
     if k.startswith('robot0_') or k in ('object','object-state'):o.create_dataset('object' if k=='object-state' else k,data=np.asarray([x[k] for x in cap.observations]))
    d.create_dataset('stage',data=np.asarray(cap.stages,dtype='S'))
    d.attrs['model_file']=cap.xml;d.attrs['num_samples']=len(actions);d.attrs['observability']=regime;d.attrs['pair_id']=e['pair'];d.attrs['version']=version
    d.attrs['accepted']=bool(accepted);d.attrs['physical_success']=bool(success)
  write(target.with_suffix('.json'),row);env.close()
 print(json.dumps({k:row[k] for k in ['pair','version','regime','physical_success','accepted','steps','exception']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--kind',choices=['threading','toolhang'],required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=202609071);p.add_argument('--pairs',type=int,default=40);p.add_argument('--index',type=int,default=0);p.add_argument('--shards',type=int,default=40);p.add_argument('--production',action='store_true');p.add_argument('--human-version',choices=['human_v2','human_v3','human_v4','human_v5','human_v6','human_v7','human_v8'],default='human_v2');p.add_argument('--attempt',type=int,default=0);p.add_argument('--output-dir',default='trials');p.add_argument('--retry-regime',choices=['full','partial']);p.add_argument('--failed-from',type=Path);a=p.parse_args()
 if a.mode=='prepare':
  assert not (a.root/'manifest.json').exists();prepare(a)
 else:
  m=json.loads((a.root/'manifest.json').read_text());assert m['kind']==a.kind
  for e in m['entries'][a.index::a.shards]:
   for v in m['versions']:
    regimes=[a.retry_regime] if a.retry_regime else m['regimes']
    for r in regimes:
     if a.failed_from is not None:
      previous=a.failed_from/f"pair_{e['pair']:03d}_{v}_{r}.json"
      if json.loads(previous.read_text())['physical_success']:continue
     trial(a,m,e,v,r)
