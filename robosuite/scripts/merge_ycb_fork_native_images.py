#!/usr/bin/env python3
"""Merge native fork RGB/proprio shards into a validated training HDF5."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")


def copy_attrs(source, target):
    for key, value in source.attrs.items(): target.attrs[key] = value


def clone(source, target, path=""):
    copy_attrs(source, target)
    for name, item in source.items():
        child = target.create_group(name) if isinstance(item, h5py.Group) else None
        if child is not None: clone(item, child, f"{path}/{name}")
        else:
            kwargs = {"compression": item.compression} if item.compression else {}
            created = target.create_dataset(name, data=item[...], **kwargs); copy_attrs(item, created)


def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source",required=True,type=Path); parser.add_argument("--shards",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path); parser.add_argument("--height",type=int,default=256); parser.add_argument("--width",type=int,default=256)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    fd,tmpname=tempfile.mkstemp(prefix=args.output.name+".tmp.",dir=args.output.parent); os.close(fd); tmp=Path(tmpname)
    try:
        with h5py.File(args.source) as source, h5py.File(tmp,"w") as output:
            clone(source,output)
            for name in source["data"]:
                with h5py.File(args.shards/f"{name}.hdf5") as shard:
                    if shard.attrs["demo"] != name: raise RuntimeError(f"{name}: shard mismatch")
                    demo=output["data"][name]; obs=demo["obs"]; length=len(demo["actions"])
                    for key in CAMERAS:
                        values=shard[key]
                        if values.shape!=(length,args.height,args.width,3) or values.dtype!=np.uint8: raise RuntimeError(f"{name}/{key}: invalid")
                        obs.create_dataset(key,data=values,compression="lzf",chunks=(1,args.height,args.width,3))
                    for key,shape in (("robot0_gripper_qpos",(length,2)),):
                        values=shard[key]
                        if values.shape!=shape or not np.isfinite(values[:]).all(): raise RuntimeError(f"{name}/{key}: invalid")
                        obs.create_dataset(key,data=values)
                    if float(shard.attrs["joint_state_max_error"])>1e-8: raise RuntimeError(f"{name}: joint parity")
            output.attrs["native_image_height"]=args.height; output.attrs["native_image_width"]=args.width
            output.attrs["native_image_source"]="stored pre-action simulator states"
        os.replace(tmp,args.output)
    finally:
        if tmp.exists(): tmp.unlink()
    print(json.dumps({"output":str(args.output),"sha256":sha256(args.output)},indent=2))


if __name__=="__main__": main()
