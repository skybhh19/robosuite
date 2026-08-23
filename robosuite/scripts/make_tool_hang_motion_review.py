"""Build a minimal motion-style review page for successful ToolHang demos."""

import argparse
import html
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main():
    output_dir = parse_args().output_dir.resolve()
    summary = json.loads(
        (output_dir / "tool_hang_wrench_joint_summary.json").read_text()
    )
    rows = sorted(summary["rollouts"], key=lambda row: int(row["state_id"]))
    styles = ("direct_low", "high_arc", "left_sweep", "right_sweep", "vertical_first")
    cards_by_style = []
    for style in styles:
        cards = []
        for row in rows:
            variation = row["variation"]
            if variation["motion_style"] != style:
                continue
            state_id = int(row["state_id"])
            regime = "FULL" if row["assigned_regime"] == "full_visible" else "PARTIAL"
            grasp_x_mm = float(variation["grasp_offset_local_x_m"]) * 1000.0
            cards.append(
                f'''<article class="card {regime.lower()}">
<div class="label"><span>Demo #{state_id:03d}</span><b>Motion: {html.escape(style)}</b></div>
<video controls muted loop playsinline preload="metadata" src="videos/rollout_{state_id:03d}.mp4"></video>
<div class="meta"><strong>Grasp: {regime}</strong><span>Grasp x: {grasp_x_mm:.1f} mm</span></div>
</article>'''
            )
        cards_by_style.append(
            f'<h2>{html.escape(style)} <small>{len(cards)} demos</small></h2>'
            f'<section class="grid">{"".join(cards)}</section>'
        )

    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ToolHang Motion Variations</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;padding:24px;background:#f5f5f3;color:#171717;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
h1{{margin:0 0 6px;font-size:25px}}p{{margin:0 0 34px;color:#666}}h2{{margin:38px 0 14px;font-size:21px;text-transform:capitalize}}h2 small{{font-size:13px;color:#777;font-weight:500;margin-left:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}}.card{{overflow:hidden;background:#fff;border:1px solid #d8d8d5;border-radius:9px}}
.label,.meta{{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:11px 12px;font-size:14px}}.label{{border-bottom:1px solid #ddd}}.label b{{text-transform:capitalize}}.meta{{border-top:1px solid #ddd}}.full .meta strong{{color:#087342}}.partial .meta strong{{color:#a14d00}}video{{display:block;width:100%;background:#000}}
</style></head><body>
<h1>ToolHang · {len(rows)} Successful Demos</h1>
<p>Five hook-approach motion variations. Grasp x is relative to the center of the black handle.</p>
{"".join(cards_by_style)}
</body></html>'''
    (output_dir / "index.html").write_text(document)
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
