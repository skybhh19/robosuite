"""Opt-in byte-identical cache for a macOS cloud-offloaded XML input.

Only the exact original mount path is intercepted. MujocoXML still sees its
original filename/folder, so relative mesh resolution and physics are unchanged.
Ordinary users without the environment variable keep the standard parser.
"""
import hashlib
import io
import os
from pathlib import Path
import xml.etree.ElementTree as ET


_installed = None


def enable_from_environment():
    global _installed
    cache = os.environ.get('ROBOSUITE_FORK_RETHINK_MOUNT_CACHE')
    if not cache:
        return
    if _installed is not None:
        if _installed != cache:
            raise RuntimeError('Different XML cache already installed')
        return
    expected = os.environ.get('ROBOSUITE_FORK_RETHINK_MOUNT_BLOB')
    if not expected:
        raise RuntimeError('XML cache requires the verified original Git blob hash')
    content = Path(cache).read_bytes()
    digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    if digest != expected:
        raise RuntimeError('XML input cache differs from the verified original')
    original = Path(__file__).resolve().parents[1] / 'models/assets/bases/rethink_mount.xml'
    parse = ET.parse

    def cached_parse(source, parser=None):
        if isinstance(source, (str, os.PathLike)) and Path(source).resolve() == original:
            return parse(io.BytesIO(content), parser=parser)
        return parse(source, parser=parser)

    ET.parse = cached_parse
    _installed = cache
