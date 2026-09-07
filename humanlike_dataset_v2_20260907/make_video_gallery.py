#!/usr/bin/env python3
"""Create a deterministic paired Full/Partial video gallery from final 14 Hz datasets."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATASETS = {
    "threading": {
        "title": "Threading D06 Hard",
        "path": "/iris/u/jasonyan/data/humanlike_dataset_v2_20260907/production_v8/threading/dataset_image84_14hz.hdf5",
        "sha256": "c46f029d7e71b8e12251ee7dc52e3efc6d95609144a3248b689f33df3d9e6cf5",
    },
    "toolhang": {
        "title": "Tool Hang Wrench Only",
        "path": "/iris/u/jasonyan/data/humanlike_dataset_v2_20260907/production/toolhang/dataset_image84_14hz.hdf5",
        "sha256": "af6925a835a701c16fa1de431e0c4fada95c18398cc83b098836f6a57e839382",
    },
}
PAIR_POSITIONS = (0, 100, 199)


def frame_for(demo, step, title, regime, pair_id):
    canvas = Image.new("RGB", (360, 192), "#0b1220")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    left = Image.fromarray(np.asarray(demo["obs/agentview_image"][step], dtype=np.uint8)).resize((168, 168), Image.Resampling.BILINEAR)
    right = Image.fromarray(np.asarray(demo["obs/robot0_eye_in_hand_image"][step], dtype=np.uint8)).resize((168, 168), Image.Resampling.BILINEAR)
    canvas.paste(left, (8, 20))
    canvas.paste(right, (184, 20))
    draw.text((8, 4), f"{title} | pair {pair_id} | {regime.upper()} | {step / 14:.1f}s", fill="white", font=font)
    draw.text((10, 178), "agentview", fill="#cbd5e1", font=font)
    draw.text((186, 178), "eye-in-hand", fill="#cbd5e1", font=font)
    return np.asarray(canvas)


def render_dataset(key, spec, output):
    records = []
    with h5py.File(spec["path"], "r") as dataset:
        demos = sorted(dataset["data"], key=lambda x: int(x.rsplit("_", 1)[1]))
        pairs = {}
        for name in demos:
            demo = dataset["data"][name]
            pairs.setdefault(int(demo.attrs["pair_id"]), {})[str(demo.attrs["observability"])] = name
        ordered = sorted(pairs)
        for position in PAIR_POSITIONS:
            pair_id = ordered[position]
            for regime in ("full", "partial"):
                name = pairs[pair_id][regime]
                demo = dataset["data"][name]
                filename = f"{key}_pair{pair_id:03d}_{regime}.mp4"
                path = output / "videos" / filename
                with imageio.get_writer(
                    path, fps=14, codec="libx264", quality=7, macro_block_size=None,
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                ) as writer:
                    for step in range(len(demo["actions"])):
                        writer.append_data(frame_for(demo, step, spec["title"], regime, pair_id))
                records.append({
                    "environment": key,
                    "title": spec["title"],
                    "pair_id": pair_id,
                    "regime": regime,
                    "demo": name,
                    "steps": len(demo["actions"]),
                    "seconds": len(demo["actions"]) / 14,
                    "video": f"videos/{filename}",
                })
    return records


def make_html(records, output):
    sections = []
    for key, spec in DATASETS.items():
        cards = []
        env_records = [row for row in records if row["environment"] == key]
        for pair_id in sorted({row["pair_id"] for row in env_records}):
            videos = []
            for row in [x for x in env_records if x["pair_id"] == pair_id]:
                videos.append(f"""
                <article class="card {html.escape(row['regime'])}">
                  <h3>{html.escape(row['regime'].title())}</h3>
                  <video controls preload="metadata" playsinline src="{html.escape(row['video'])}"></video>
                  <p>{html.escape(row['demo'])} · {row['steps']} frames · {row['seconds']:.1f}s · 14 Hz</p>
                </article>""")
            cards.append(f'<section class="pair"><h2>Matched pair {pair_id}</h2><div class="grid">{"".join(videos)}</div></section>')
        sections.append(f"""
        <section class="environment">
          <h1>{html.escape(spec['title'])}</h1>
          <p class="path"><strong>Training HDF5:</strong> <code>{html.escape(spec['path'])}</code></p>
          <p class="path"><strong>SHA-256:</strong> <code>{spec['sha256']}</code></p>
          {''.join(cards)}
        </section>""")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Humanlike Dataset Video Review</title>
<style>
:root{{color-scheme:light dark;--bg:#f5f7fb;--card:#fff;--ink:#111827;--muted:#64748b;--border:#dbe2ea}}
@media(prefers-color-scheme:dark){{:root{{--bg:#08101d;--card:#111827;--ink:#e5edf7;--muted:#9aa9bc;--border:#263449}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:32px 24px 64px}}header{{margin-bottom:36px}}h1{{margin:0 0 8px}}h2{{margin:30px 0 12px}}h3{{margin:0 0 10px;text-transform:capitalize}}p{{color:var(--muted)}}
.environment{{margin-top:48px;padding-top:30px;border-top:1px solid var(--border)}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px;box-shadow:0 5px 18px #0001}}video{{display:block;width:100%;aspect-ratio:15/8;background:#020617;border-radius:9px}}code{{overflow-wrap:anywhere}}.path{{margin:4px 0}}
.full h3{{color:#2563eb}}.partial h3{{color:#d97706}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}main{{padding:22px 14px 48px}}}}
</style></head><body><main>
<header><h1>Humanlike Dataset Video Review</h1><p>Three deterministic matched pairs per environment. Each Full/Partial pair starts from the same saved simulator state. Both stored RGB cameras are shown side by side at the training rate.</p></header>
{''.join(sections)}
</main></body></html>"""
    (output / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "videos").mkdir(exist_ok=True)
    records = []
    for key, spec in DATASETS.items():
        records.extend(render_dataset(key, spec, args.output))
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    make_html(records, args.output)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
