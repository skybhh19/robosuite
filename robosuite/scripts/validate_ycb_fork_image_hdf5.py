#!/usr/bin/env python3
"""Fail-closed structural and image audit for the final fork VLA dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def decode(values): return [x.decode() if isinstance(x,bytes) else str(x) for x in values]
def sha256(path):
    d=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): d.update(b)
    return d.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True,type=Path); p.add_argument("--report",required=True,type=Path); a=p.parse_args()
    result={}
    with h5py.File(a.dataset) as f:
        names=sorted(f["data"],key=lambda x:int(x.rsplit("_",1)[1])); full=set(decode(f["mask/full"][:])); partial=set(decode(f["mask/partial"][:]))
        train=set(decode(f["mask/train"][:])); valid=set(decode(f["mask/valid"][:])); all_names=set(decode(f["mask/all"][:]))
        if len(names)!=300 or len(full)!=150 or len(partial)!=150 or full&partial or full|partial!=all_names or all_names!=set(names): raise RuntimeError("mask/count audit failed")
        if len(train)!=240 or len(valid)!=60 or train&valid or train|valid!=all_names: raise RuntimeError("split audit failed")
        combos={"full":set(),"partial":set()}; total=0; min_std=float("inf")
        for name in names:
            d=f["data"][name]; n=len(d["actions"]); total+=n; regime=d.attrs["observability"]
            if d["actions"].shape!=(n,8) or d["obs/robot0_joint_pos"].shape!=(n,7) or d["obs/robot0_gripper_qpos"].shape!=(n,2): raise RuntimeError(f"{name}: state/action shape")
            if np.max(np.abs(d["actions"][:,:7]-d["obs/robot0_joint_pos"][:]-d["actions_joint_delta"][:,:7]))>=1e-8: raise RuntimeError(f"{name}: action identity")
            variation=json.loads(d.attrs["variation"]); combos[regime].add((variation["motion_style"],variation["style_variant"],variation["side_sign"]))
            for key in ("agentview_image","robot0_eye_in_hand_image"):
                x=d["obs"][key]
                if x.shape!=(n,256,256,3) or x.dtype!=np.uint8: raise RuntimeError(f"{name}/{key}: shape")
                for start in range(0,n,32): min_std=min(min_std,float(np.std(x[start:start+32],axis=(1,2,3)).min()))
        if any(len(x)!=100 for x in combos.values()): raise RuntimeError(f"variation coverage failed: { {k:len(v) for k,v in combos.items()} }")
        if min_std<=1: raise RuntimeError("blank image detected")
        result={"passed":True,"episodes":300,"full":150,"partial":150,"train":240,"valid":60,"frames":total,"image_shape":[256,256,3],"minimum_frame_std":min_std,"unique_path_families":{"full":100,"partial":100}}
    result["sha256"]=sha256(a.dataset); a.report.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
