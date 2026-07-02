"""Build a static annotation tool for Threading partial / full / unused labels."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


def load_records(dataset_dir, render_dir, output_dir):
    records = []
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for idx, stats_path in enumerate(sorted(dataset_dir.glob("ep_*/policy_stats.json")), start=1):
        ep_dir = stats_path.parent
        with open(stats_path, "r") as f:
            stats = json.load(f)

        video_name = f"{ep_dir.name}_agentview_wrist.mp4"
        src_video = render_dir / video_name
        if not src_video.exists():
            raise FileNotFoundError(f"Missing rendered video for {ep_dir.name}: {src_video}")

        dst_video = videos_dir / f"demo_{idx:03d}.mp4"
        if not dst_video.exists() or dst_video.stat().st_size != src_video.stat().st_size:
            shutil.copy2(src_video, dst_video)

        actual_angle = stats.get("actual_insert_angle_deg")
        script_label = "partial" if actual_angle is not None and actual_angle < 95.0 else "full"
        records.append(
            {
                "dataset": dataset_dir.name,
                "dataset_name": dataset_dir.name,
                "demo_idx": idx,
                "name": f"demo_{idx:03d}",
                "episode": ep_dir.name,
                "video": f"videos/{dst_video.name}",
                "script_label": script_label,
                "actual_insert_angle_deg": actual_angle,
                "motion_style": stats.get("motion_style"),
                "style_variant": stats.get("style_variant"),
                "steps": stats.get("steps"),
                "policy_success": bool(stats.get("policy_success")),
                "env_success": bool(stats.get("env_success")),
                "smooth_pass": bool(stats.get("smooth_filter", {}).get("passed", True)),
            }
        )
    return records


def build_html(records, storage_key):
    payload = json.dumps(records, separators=(",", ":"))
    storage_key_json = json.dumps(storage_key)
    dataset_name = html.escape(records[0]["dataset_name"] if records else "threading")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threading Partial / Full / Unused Annotation</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f8;
      color: #17202a;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 18px;
      background: #101820;
      color: #fff;
      border-bottom: 1px solid #26323d;
    }}
    header h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .top-actions {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button, label.import {{
      border: 1px solid #c7d0da;
      background: #fff;
      color: #17202a;
      border-radius: 6px;
      padding: 8px 11px;
      font-size: 14px;
      cursor: pointer;
      line-height: 1;
    }}
    button:hover, label.import:hover {{ background: #eef3f7; }}
    button.primary {{ background: #1f6feb; border-color: #1f6feb; color: #fff; }}
    button.partial.active {{ background: #f2c94c; border-color: #d9ae2f; }}
    button.full.active {{ background: #27ae60; border-color: #1f8f4d; color: #fff; }}
    button.unsure.active {{ background: #8e7cc3; border-color: #6f5aa8; color: #fff; }}
    button.unused.active {{ background: #eb5757; border-color: #ca3f3f; color: #fff; }}
    input[type=file] {{ display: none; }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      min-height: calc(100vh - 57px);
    }}
    .stage {{
      padding: 18px;
      min-width: 0;
    }}
    .video-wrap {{
      background: #0b1117;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0,0,0,0.14);
    }}
    video {{
      display: block;
      width: 100%;
      max-height: calc(100vh - 250px);
      background: #000;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .metric {{
      background: #fff;
      border: 1px solid #d8e0e8;
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .metric b {{
      display: block;
      font-size: 12px;
      color: #607080;
      margin-bottom: 4px;
      font-weight: 600;
    }}
    .metric span {{
      font-size: 15px;
      font-weight: 600;
    }}
    .controls {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .controls button {{
      min-width: 108px;
      padding: 11px 14px;
      font-weight: 650;
    }}
    textarea {{
      width: 100%;
      margin-top: 12px;
      border: 1px solid #c7d0da;
      border-radius: 8px;
      padding: 10px;
      min-height: 74px;
      resize: vertical;
      font: inherit;
    }}
    aside {{
      border-left: 1px solid #d8e0e8;
      background: #fff;
      overflow: auto;
      max-height: calc(100vh - 57px);
    }}
    .progress {{
      padding: 14px;
      border-bottom: 1px solid #d8e0e8;
      background: #fbfcfd;
    }}
    .bar {{
      height: 8px;
      background: #e7edf3;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 9px;
    }}
    .fill {{ height: 100%; background: #1f6feb; width: 0%; }}
    .list {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 6px;
      padding: 12px;
    }}
    .tile {{
      border: 1px solid #d8e0e8;
      background: #fff;
      border-radius: 6px;
      min-height: 44px;
      font-size: 13px;
      cursor: pointer;
    }}
    .tile.current {{ outline: 2px solid #1f6feb; }}
    .tile.partial {{ background: #fff4cc; border-color: #e0bd3a; }}
    .tile.full {{ background: #dff5e8; border-color: #2ea35a; }}
    .tile.unsure {{ background: #eee8ff; border-color: #8e7cc3; }}
    .tile.unused {{ background: #ffe1e1; border-color: #d45a5a; }}
    .hint {{
      color: #607080;
      font-size: 13px;
      margin-top: 8px;
      line-height: 1.35;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ max-height: none; border-left: 0; border-top: 1px solid #d8e0e8; }}
      .meta {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <header>
      <h1>Threading Partial / Full / Unused Annotation · {dataset_name}</h1>
    <div class="top-actions">
      <button id="downloadCsv">Download CSV</button>
      <button id="downloadJson">Download JSON</button>
      <label class="import">Import CSV<input id="importCsv" type="file" accept=".csv,text/csv"></label>
      <button id="resetLabels">Reset</button>
    </div>
  </header>
  <main>
    <section class="stage">
      <div class="video-wrap">
        <video id="video" controls autoplay muted loop playsinline></video>
      </div>
      <div class="meta">
        <div class="metric"><b>Demo</b><span id="demoName"></span></div>
        <div class="metric"><b>Label</b><span id="labelText"></span></div>
        <div class="metric"><b>Progress</b><span id="indexText"></span></div>
        <div class="metric"><b>Episode</b><span id="episodeText"></span></div>
      </div>
      <div class="controls">
        <button class="partial" id="partialBtn">1 Partial</button>
        <button class="full" id="fullBtn">2 Full</button>
        <button class="unused" id="unusedBtn">3 Unused</button>
        <button class="unsure" id="unsureBtn">4 Unsure</button>
        <button id="prevBtn">← Prev</button>
        <button id="nextBtn" class="primary">Next →</button>
      </div>
      <textarea id="note" placeholder="Optional note"></textarea>
      <div class="hint">Keyboard: 1 partial, 2 full, 3 unused, 4 unsure, ←/→ navigate, space play/pause. Labels are saved in this browser automatically; export CSV when done.</div>
    </section>
    <aside>
      <div class="progress">
        <div id="summary"></div>
        <div class="bar"><div class="fill" id="fill"></div></div>
      </div>
      <div class="list" id="list"></div>
    </aside>
  </main>
  <script>
    const records = {payload};
    const storageKey = {storage_key_json};
    let state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    let current = 0;

    function emptyEntry() {{ return {{ label: "", note: "", updated_at: "" }}; }}
    function entry(record) {{
      if (!state[record.name]) state[record.name] = emptyEntry();
      return state[record.name];
    }}
    function save() {{ localStorage.setItem(storageKey, JSON.stringify(state)); }}
    function escCsv(value) {{
      const text = String(value ?? "");
      return /[",\\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
    }}
    function setLabel(label) {{
      const rec = records[current];
      const e = entry(rec);
      e.label = label;
      e.note = document.getElementById("note").value;
      e.updated_at = new Date().toISOString();
      save();
      if (current < records.length - 1) current += 1;
      render();
    }}
    function render() {{
      const rec = records[current];
      const e = entry(rec);
      const video = document.getElementById("video");
      if (!video.src.endsWith(rec.video)) {{
        video.src = rec.video;
        video.playbackRate = 1.0;
      }}
      document.getElementById("demoName").textContent = rec.name;
      document.getElementById("labelText").textContent = e.label || "unlabeled";
      document.getElementById("indexText").textContent = `${{current + 1}} / ${{records.length}}`;
      document.getElementById("episodeText").textContent = rec.episode;
      document.getElementById("note").value = e.note || "";
      for (const name of ["partial", "full", "unused", "unsure"]) {{
        document.getElementById(name + "Btn").classList.toggle("active", e.label === name);
      }}
      renderList();
      updateSummary();
      save();
    }}
    function renderList() {{
      const list = document.getElementById("list");
      list.innerHTML = "";
      records.forEach((rec, idx) => {{
        const e = entry(rec);
        const b = document.createElement("button");
        b.className = `tile ${{e.label || ""}} ${{idx === current ? "current" : ""}}`;
        b.textContent = idx + 1;
        b.onclick = () => {{ current = idx; render(); }};
        list.appendChild(b);
      }});
    }}
    function updateSummary() {{
      const counts = {{ partial: 0, full: 0, unused: 0, unsure: 0, unlabeled: 0 }};
      for (const rec of records) {{
        const label = entry(rec).label || "unlabeled";
        counts[label] = (counts[label] || 0) + 1;
      }}
      const done = records.length - counts.unlabeled;
      document.getElementById("summary").textContent =
        `Done ${{done}}/${{records.length}} | partial ${{counts.partial}} | full ${{counts.full}} | unused ${{counts.unused}} | unsure ${{counts.unsure}}`;
      document.getElementById("fill").style.width = `${{100 * done / records.length}}%`;
    }}
    function rows() {{
      return records.map((rec) => {{
        const e = entry(rec);
        return {{
          dataset: rec.dataset,
          demo_idx: rec.demo_idx,
          name: rec.name,
          episode: rec.episode,
          label: e.label || "",
          note: e.note || "",
          updated_at: e.updated_at || "",
          video: rec.video,
          script_label: rec.script_label,
          actual_insert_angle_deg: rec.actual_insert_angle_deg,
          motion_style: rec.motion_style,
          style_variant: rec.style_variant,
          steps: rec.steps,
          policy_success: rec.policy_success,
          env_success: rec.env_success,
          smooth_pass: rec.smooth_pass,
        }};
      }});
    }}
    function download(filename, text, type) {{
      const blob = new Blob([text], {{ type }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }}
    document.getElementById("partialBtn").onclick = () => setLabel("partial");
    document.getElementById("fullBtn").onclick = () => setLabel("full");
    document.getElementById("unusedBtn").onclick = () => setLabel("unused");
    document.getElementById("unsureBtn").onclick = () => setLabel("unsure");
    document.getElementById("prevBtn").onclick = () => {{ current = Math.max(0, current - 1); render(); }};
    document.getElementById("nextBtn").onclick = () => {{ current = Math.min(records.length - 1, current + 1); render(); }};
    document.getElementById("note").oninput = () => {{
      const e = entry(records[current]);
      e.note = document.getElementById("note").value;
      e.updated_at = new Date().toISOString();
      save();
    }};
    document.getElementById("downloadCsv").onclick = () => {{
      const r = rows();
      const header = Object.keys(r[0]);
      const csv = [header.join(","), ...r.map(row => header.map(k => escCsv(row[k])).join(","))].join("\\n");
      download("threading_partial_full_unused_annotations.csv", csv, "text/csv");
    }};
    document.getElementById("downloadJson").onclick = () => {{
      download("threading_partial_full_unused_annotations.json", JSON.stringify(rows(), null, 2), "application/json");
    }};
    document.getElementById("resetLabels").onclick = () => {{
      if (confirm("Clear all labels stored in this browser?")) {{
        state = {{}};
        save();
        render();
      }}
    }};
    document.getElementById("importCsv").onchange = async (event) => {{
      const file = event.target.files[0];
      if (!file) return;
      const text = await file.text();
      const lines = text.trim().split(/\\r?\\n/);
      const header = lines.shift().split(",");
      const idxName = header.indexOf("name");
      const idxLabel = header.indexOf("label");
      const idxNote = header.indexOf("note");
      const idxUpdated = header.indexOf("updated_at");
      for (const line of lines) {{
        const cols = line.match(/("([^"]|"")*"|[^,]*)/g).filter((_, i) => i % 2 === 0).map(v => v.replace(/^"|"$/g, "").replaceAll('""', '"'));
        const name = cols[idxName];
        if (!name) continue;
        state[name] = {{
          label: cols[idxLabel] || "",
          note: idxNote >= 0 ? cols[idxNote] || "" : "",
          updated_at: idxUpdated >= 0 ? cols[idxUpdated] || "" : new Date().toISOString(),
        }};
      }}
      save();
      render();
    }};
    document.addEventListener("keydown", (event) => {{
      if (event.target.tagName === "TEXTAREA") return;
      if (event.key === "1") setLabel("partial");
      else if (event.key === "2") setLabel("full");
      else if (event.key === "3") setLabel("unused");
      else if (event.key === "4") setLabel("unsure");
      else if (event.key === "ArrowLeft") {{ current = Math.max(0, current - 1); render(); }}
      else if (event.key === "ArrowRight") {{ current = Math.min(records.length - 1, current + 1); render(); }}
      else if (event.key === " ") {{
        event.preventDefault();
        const v = document.getElementById("video");
        if (v.paused) v.play(); else v.pause();
      }}
    }});
    render();
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.dataset_dir, args.render_dir, args.output_dir)
    storage_hash = hashlib.sha1(str(args.dataset_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    storage_key = f"threading_partial_full_unused_annotations_v2_{storage_hash}"
    index_path = args.output_dir / "index.html"
    index_path.write_text(build_html(records, storage_key))
    print(f"wrote {index_path}")
    print(f"videos={len(records)}")


if __name__ == "__main__":
    main()
