#!/usr/bin/env python3
"""
Build parts catalog metadata and generate canonical 3D isometric thumbnails
using LDraw and OSMesa for the LEGO Sorter web GUI.
"""

import os
import sys
import tempfile
import random
import json
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

WORKSPACE = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
LDVIEW_BIN = WORKSPACE / "tools" / "ldview-osmesa"
LDRAW_DIR = WORKSPACE / "ldraw_lib" / "ldraw"
THUMBS_DIR = APP_DIR / "static" / "thumbnails"
CATALOG_FILE = APP_DIR / "data" / "parts_catalog.json"

# Categorization mapping helper
def categorize_part(name: str, ldraw_header: str = "") -> str:
    text = (name + " " + ldraw_header).lower()
    if any(k in text for k in ["technic", "gear", "axle", "liftarm", "bush", "driv"]):
        return "Technic"
    elif any(k in text for k in ["tile", "grille"]):
        return "Tiles"
    elif any(k in text for k in ["slope", "wedge", "curved slope", "roof"]):
        return "Slopes & Wedges"
    elif any(k in text for k in ["plate"]):
        return "Plates"
    elif any(k in text for k in ["brick"]):
        return "Bricks"
    elif any(k in text for k in ["wheel", "tyre", "tire", "rim"]):
        return "Wheels & Tires"
    elif any(k in text for k in ["window", "door", "glass", "windshield", "pane"]):
        return "Windows & Doors"
    elif any(k in text for k in ["round", "cone", "cylinder", "dome"]):
        return "Round & Cylinders"
    elif any(k in text for k in ["minifig", "helmet", "hair", "head", "torso", "weapon"]):
        return "Minifigures"
    elif any(k in text for k in ["hinge", "clip", "handle", "bracket", "turntable"]):
        return "Hinges, Clips & Brackets"
    else:
        return "Special & Other"


def render_thumb_worker(task):
    part_id, out_png, color_code = task
    tmp_ldr = tempfile.mktemp(suffix=".ldr")
    # Canonical CAD presentation: color code 71 (Light Bluish Gray) or 14 (Yellow)
    ldr_content = (
        f"0 Part {part_id}\n"
        f"1 {color_code} 0 0 0 1 0 0 0 1 0 0 0 1 {part_id}.dat\n"
    )
    with open(tmp_ldr, "w") as f:
        f.write(ldr_content)

    cmd = [
        str(LDVIEW_BIN),
        f"-LDrawDir={str(LDRAW_DIR)}",
        tmp_ldr,
        f"-SaveSnapshot={out_png}",
        "-SaveWidth=160",
        "-SaveHeight=160",
        "-SaveZoomToFit=1",
        "-SaveAlpha=1",
        "-DefaultLatLong=30,45",
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        return os.path.exists(out_png) and os.path.getsize(out_png) > 0
    except Exception:
        return False
    finally:
        if os.path.exists(tmp_ldr):
            try: os.remove(tmp_ldr)
            except OSError: pass


def main():
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    
    parts_metadata = {}

    # 0. Seed from existing parts_catalog.json if present
    if CATALOG_FILE.exists():
        try:
            with open(CATALOG_FILE, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    pid = item.get("id", "").strip().lower()
                    if pid:
                        parts_metadata[pid] = item
        except Exception:
            pass

    # 1. Read detailed metadata from next_1000_parts_detailed.tsv if available
    det_tsv = WORKSPACE / "next_1000_parts_detailed.tsv"
    if det_tsv.exists():
        with open(det_tsv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    pid = parts[0].strip().lower()
                    cat = parts[1].strip()
                    name = parts[2].strip()
                    parts_metadata[pid] = {"id": pid, "name": name, "category": cat}

    # 2. Add parts from top_2000_parts.txt
    top_file = WORKSPACE / "top_2000_parts.txt"
    all_target_ids = []
    if top_file.exists():
        with open(top_file, "r") as f:
            for line in f:
                pid = line.strip().lower()
                if pid:
                    all_target_ids.append(pid)

    # 3. Add existing dataset folders
    dataset_dir = WORKSPACE / "dataset"
    if dataset_dir.exists():
        for d in dataset_dir.iterdir():
            if d.is_dir():
                all_target_ids.append(d.name.lower())

    all_target_ids = sorted(list(set(all_target_ids)))

    # Fetch names from LDraw headers for any missing parts
    catalog = []
    for pid in all_target_ids:
        dat_path = LDRAW_DIR / "parts" / f"{pid}.dat"
        if not dat_path.exists():
            continue

        if pid in parts_metadata:
            item = parts_metadata[pid]
            category = categorize_part(item["name"], item["category"])
            name = item["name"]
        else:
            # Read first line of LDraw dat file
            try:
                with open(dat_path, "r", encoding="utf-8", errors="ignore") as fp:
                    header = fp.readline().strip().lstrip("0").strip()
            except Exception:
                header = f"Part {pid}"
            name = header if header else f"Part {pid}"
            category = categorize_part(name)

        catalog.append({
            "id": pid,
            "name": name,
            "category": category,
            "thumbnail": f"/static/thumbnails/{pid}.png"
        })

    # Save catalog
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"Saved catalog metadata for {len(catalog)} parts in {CATALOG_FILE}")

    # Render missing thumbnails
    tasks = []
    for item in catalog:
        pid = item["id"]
        out_png = THUMBS_DIR / f"{pid}.png"
        if not out_png.exists() or out_png.stat().st_size == 0:
            tasks.append((pid, str(out_png), 71)) # 71 = Light Bluish Gray

    print(f"Rendering {len(tasks)} missing thumbnails in parallel with OSMesa...")
    if tasks:
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
            futures = [executor.submit(render_thumb_worker, t) for t in tasks]
            with tqdm(total=len(tasks), desc="Thumbnails", unit="part") as pbar:
                for f in as_completed(futures):
                    pbar.update(1)

    print(f"All {len(catalog)} canonical thumbnails ready in {THUMBS_DIR}!")

if __name__ == "__main__":
    main()
