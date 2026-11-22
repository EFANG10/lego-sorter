#!/usr/bin/env python3
"""
LDraw LEGO Synthetic Dataset Generator for Machine Vision & Sorting Machines

Generates photorealistic synthetic training datasets of LEGO parts with:
- Top 1,000 most common LEGO parts (or all 23,800+ parts in LDraw)
- Dynamic multi-color backgrounds across the entire color wheel (prevents color memorization)
- Full-spectrum brick colors (official LEGO colors + arbitrary 24-bit RGB hex colors)
- Diverse 3D poses (canonical conveyor resting poses + full 3D SO(3) tumbling poses)
- Variable camera elevations, azimuths, and machine vision lighting/sensor augmentations
- Multi-process parallel rendering, automatic resume, and manifest generation
"""

import os
import sys
import time
import math
import random
import argparse
import subprocess
import tempfile
import re
import json
import colorsys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

DEFAULT_WORKSPACE = Path(__file__).resolve().parent
OSMESA_BIN = DEFAULT_WORKSPACE / "tools" / "ldview-osmesa"
STD_LDVIEW = DEFAULT_WORKSPACE / "tools" / "ldview-dist" / "usr" / "bin" / "LDView"
DEFAULT_LDVIEW = OSMESA_BIN if OSMESA_BIN.exists() else STD_LDVIEW
DEFAULT_LDRAW_DIR = DEFAULT_WORKSPACE / "ldraw_lib" / "ldraw"
TOP_2000_FILE = DEFAULT_WORKSPACE / "top_2000_parts.txt"
ALL_PARTS_FILE = DEFAULT_WORKSPACE / "all_parts.txt"

# Standard popular LEGO color codes
POPULAR_LEGO_COLORS = [
    0,   # Black
    1,   # Blue
    2,   # Green
    4,   # Red
    5,   # Dark Pink
    6,   # Brown
    7,   # Light Grey
    8,   # Dark Grey
    10,  # Bright Green
    14,  # Yellow
    15,  # White
    19,  # Tan
    25,  # Orange
    27,  # Lime
    28,  # Medium Dark Pink
    29,  # Bright Pink
    70,  # Reddish Brown
    71,  # Light Bluish Grey
    72,  # Dark Bluish Grey
    84,  # Medium Dark Flesh
    85,  # Medium Lilac / Dark Purple
    191, # Bright Light Orange
    212, # Medium Azure
    272, # Dark Blue
    320, # Dark Red
    288, # Dark Green
    179, # Flat Silver
    297, # Pearl Gold
    47,  # Trans-Clear
    36,  # Trans-Red
    46,  # Trans-Yellow
]


def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable duration."""
    if seconds <= 0:
        return "0s"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if d > 0:
        return f"{d}d {h}h {m}m"
    elif h > 0:
        return f"{h}h {m:02d}m"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def load_ldraw_colors(ldconfig_path: Path):
    """Parses color codes from LDConfig.ldr."""
    if not ldconfig_path.exists():
        return POPULAR_LEGO_COLORS
    colors = []
    with open(ldconfig_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.match(r"^0\s+!COLOUR\s+(\S+)\s+CODE\s+(\d+)", line)
            if match:
                colors.append(int(match.group(2)))
    return colors if colors else POPULAR_LEGO_COLORS


def discover_all_parts(ldraw_parts_dir: Path):
    """Scans and filters all valid renderable parts from LDraw library."""
    if ALL_PARTS_FILE.exists():
        with open(ALL_PARTS_FILE, "r", encoding="utf-8") as f:
            cached = [line.strip() for line in f if line.strip()]
        if cached:
            return cached

    valid = []
    for dat_file in sorted(ldraw_parts_dir.glob("*.dat")):
        if dat_file.name.startswith("~"):
            continue
        try:
            with open(dat_file, "r", encoding="utf-8", errors="ignore") as fp:
                first_line = fp.readline()
                if "~Moved to" in first_line or "=Moved to" in first_line:
                    continue
            valid.append(dat_file.stem)
        except Exception:
            continue

    with open(ALL_PARTS_FILE, "w", encoding="utf-8") as f:
        for p in valid:
            f.write(p + "\n")

    return valid


def get_canonical_resting_rotation():
    """
    Returns a 3x3 rotation matrix for a brick resting on one of its 6 primary faces
    with random in-plane rotation and realistic tilt jitter.
    """
    primary_faces = [
        (0.0, 0.0, 0.0),          # Base flat / studs up
        (math.pi, 0.0, 0.0),      # Upside down / studs down
        (math.pi / 2, 0.0, 0.0),  # Lying on front face
        (-math.pi / 2, 0.0, 0.0), # Lying on back face
        (0.0, 0.0, math.pi / 2),  # Lying on right face
        (0.0, 0.0, -math.pi / 2), # Lying on left face
    ]
    pitch0, yaw0, roll0 = random.choice(primary_faces)
    yaw = random.uniform(0, 2 * math.pi)
    tilt_x = random.uniform(-0.21, 0.21)
    tilt_z = random.uniform(-0.21, 0.21)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch0 + tilt_x), -math.sin(pitch0 + tilt_x)],
        [0, math.sin(pitch0 + tilt_x), math.cos(pitch0 + tilt_x)]
    ])
    Ry = np.array([
        [math.cos(yaw0 + yaw), 0, math.sin(yaw0 + yaw)],
        [0, 1, 0],
        [-math.sin(yaw0 + yaw), 0, math.cos(yaw0 + yaw)]
    ])
    Rz = np.array([
        [math.cos(roll0 + tilt_z), -math.sin(roll0 + tilt_z), 0],
        [math.sin(roll0 + tilt_z), math.cos(roll0 + tilt_z), 0],
        [0, 0, 1]
    ])

    R = Ry @ Rx @ Rz
    return R.flatten().tolist()


def get_so3_random_rotation():
    """Generates a uniform random 3D rotation from SO(3) using random quaternions."""
    u1, u2, u3 = random.random(), random.random(), random.random()
    q = [
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3)
    ]
    w, x, y, z = q
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x*x + y*y)
    return [r00, r01, r02, r10, r11, r12, r20, r21, r22]


def generate_dynamic_multicolor_background(width: int, height: int, augment: bool) -> Image.Image:
    """
    Generates a dynamically colored background covering the entire spectrum
    so the AI model never memorizes background colors or brick-background color correlations.
    """
    mode = random.choices(
        ["hsv_vibrant", "gradient", "textured", "neutral", "pastels"],
        weights=[35, 25, 20, 12, 8]
    )[0]

    if mode == "hsv_vibrant":
        # Random hue, saturation, and brightness across full RGB color wheel
        h = random.random()
        s = random.uniform(0.20, 0.95)
        v = random.uniform(0.20, 0.98)
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)]
        bg = Image.new("RGBA", (width, height), (r, g, b, 255))

    elif mode == "gradient":
        # Dual-color linear gradient (e.g. overhead factory lighting gradient)
        h1, h2 = random.random(), random.random()
        s1, s2 = random.uniform(0.2, 0.9), random.uniform(0.2, 0.9)
        v1, v2 = random.uniform(0.2, 0.95), random.uniform(0.2, 0.95)
        c1 = np.array([int(c * 255) for c in colorsys.hsv_to_rgb(h1, s1, v1)], dtype=np.float32)
        c2 = np.array([int(c * 255) for c in colorsys.hsv_to_rgb(h2, s2, v2)], dtype=np.float32)
        
        grad_dir = random.choice(["v", "h"])
        if grad_dir == "v":
            t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
            grad = (1 - t) * c1 + t * c2
            grad = np.repeat(grad, width, axis=1).astype(np.uint8)
        else:
            t = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
            grad = (1 - t) * c1 + t * c2
            grad = np.repeat(grad, height, axis=0).astype(np.uint8)
        bg = Image.fromarray(grad).convert("RGBA")

    elif mode == "textured":
        # Random colored conveyor surface with noise grain
        h = random.random()
        s = random.uniform(0.1, 0.7)
        v = random.uniform(0.15, 0.85)
        base = np.array([int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)], dtype=np.float32)
        noise = np.random.normal(0, random.uniform(4, 14), (height, width, 3))
        textured = np.clip(base + noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(textured).convert("RGBA")

    elif mode == "pastels":
        # Soft pastel / light tinted backgrounds
        h = random.random()
        s = random.uniform(0.10, 0.35)
        v = random.uniform(0.85, 0.98)
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)]
        bg = Image.new("RGBA", (width, height), (r, g, b, 255))

    else:
        # Neutral conveyor tones (charcoal rubber, graphite, medium grey, ivory, white)
        val = random.choice([
            random.randint(15, 45),    # dark rubber
            random.randint(80, 130),   # industrial slate
            random.randint(180, 215),  # light grey
            random.randint(225, 250),  # sorting chute white
        ])
        bg = Image.new("RGBA", (width, height), (val, val, val, 255))

    # Optional subtle sensor noise
    if augment:
        noise = np.random.normal(0, random.uniform(1.0, 3.0), (height, width, 3)).astype(np.float32)
        bg_arr = np.array(bg.convert("RGB"), dtype=np.float32) + noise
        bg_arr = np.clip(bg_arr, 0, 255).astype(np.uint8)
        bg = Image.fromarray(bg_arr).convert("RGBA")

    return bg


def render_worker_task(task_params):
    """
    Renders a single LEGO part image with specific 3D pose, color, and dynamic background.
    """
    (
        part_id,
        color_spec,
        out_path,
        res,
        camera_lat,
        camera_lon,
        is_resting_pose,
        bg_mode,
        augment,
        ldview_bin,
        ldraw_dir,
        worker_id,
    ) = task_params

    # Select pose
    if is_resting_pose:
        rot = get_canonical_resting_rotation()
    else:
        rot = get_so3_random_rotation()

    # Create temporary LDR file
    tmp_dir = tempfile.gettempdir()
    tmp_ldr = os.path.join(tmp_dir, f"lego_{worker_id}_{os.getpid()}_{random.randint(0, 999999)}.ldr")
    tmp_png = tmp_ldr.replace(".ldr", ".png")

    ldr_content = (
        f"0 LEGO Part {part_id}\n"
        f"1 {color_spec} 0 0 0 "
        f"{rot[0]:.4f} {rot[1]:.4f} {rot[2]:.4f} "
        f"{rot[3]:.4f} {rot[4]:.4f} {rot[5]:.4f} "
        f"{rot[6]:.4f} {rot[7]:.4f} {rot[8]:.4f} "
        f"{part_id}.dat\n"
    )

    try:
        with open(tmp_ldr, "w", encoding="utf-8") as f:
            f.write(ldr_content)

        cmd = [
            str(ldview_bin),
            f"-LDrawDir={str(ldraw_dir)}",
            tmp_ldr,
            f"-SaveSnapshot={tmp_png}",
            f"-SaveWidth={res}",
            f"-SaveHeight={res}",
            "-SaveZoomToFit=1",
            "-SaveAlpha=1",
            f"-DefaultLatLong={camera_lat:.1f},{camera_lon:.1f}",
        ]

        res_code = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode

        if res_code == 0 and os.path.exists(tmp_png):
            img = Image.open(tmp_png).convert("RGBA")
            
            if bg_mode == "transparent":
                final_img = img
            else:
                # Dynamic multi-color background
                bg = generate_dynamic_multicolor_background(img.width, img.height, augment)
                composite = Image.alpha_composite(bg, img).convert("RGB")
                
                if augment:
                    if random.random() < 0.6:
                        composite = ImageEnhance.Brightness(composite).enhance(random.uniform(0.85, 1.15))
                    if random.random() < 0.4:
                        composite = ImageEnhance.Contrast(composite).enhance(random.uniform(0.90, 1.15))
                    if random.random() < 0.15:
                        composite = composite.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
                final_img = composite
            
            # Save final image
            if out_path.endswith(".jpg") or out_path.endswith(".jpeg"):
                final_img.convert("RGB").save(out_path, quality=92)
            else:
                final_img.save(out_path)
            return True
    except Exception:
        return False
    finally:
        if os.path.exists(tmp_ldr):
            try:
                os.remove(tmp_ldr)
            except OSError:
                pass
        if os.path.exists(tmp_png):
            try:
                os.remove(tmp_png)
            except OSError:
                pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic LEGO dataset for machine vision sorting."
    )
    parser.add_argument(
        "--top-2000",
        action="store_true",
        help="Render the top 2,000 most common LEGO parts in existence (default)",
    )
    parser.add_argument(
        "--top-1000",
        action="store_true",
        help="Render the top 1,000 most common LEGO parts in existence",
    )
    parser.add_argument(
        "--all-parts",
        action="store_true",
        help="Render EVERY possible LEGO part in the LDraw library (23,800+ parts)",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        help="Space-separated list of LEGO part numbers (e.g. 3001 3002 3003 3020)",
    )
    parser.add_argument(
        "--parts-file",
        type=str,
        help="Path to file listing part numbers (e.g. top_2000_parts.txt, all_parts.txt)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=2000,
        help="Number of images to generate per part (default: 2000)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Image resolution width and height in pixels (default: 224)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Output directory path (default: dataset)",
    )
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Image format (default: jpg)",
    )
    parser.add_argument(
        "--bg-mode",
        choices=["multicolor", "transparent", "conveyor_black", "tray_white"],
        default="multicolor",
        help="Background mode (default: multicolor, changes colors on every image)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable camera lighting and sensor noise augmentations",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 8,
        help=f"Number of parallel render workers (default: {os.cpu_count() or 8})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip images that have already been rendered",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        help="Start rendering from this part ID alphabetically/numerically",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit execution to the first N parts in the target list",
    )
    parser.add_argument(
        "--ldview",
        type=str,
        default=str(DEFAULT_LDVIEW),
        help="Path to LDView executable",
    )
    parser.add_argument(
        "--ldraw-dir",
        type=str,
        default=str(DEFAULT_LDRAW_DIR),
        help="Path to LDraw parts library directory",
    )

    args = parser.parse_args()

    ldview_bin = Path(args.ldview).resolve()
    ldraw_dir = Path(args.ldraw_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not ldview_bin.exists():
        sys.exit(f"Error: LDView executable not found at {ldview_bin}")
    if not (ldraw_dir / "parts").exists():
        sys.exit(f"Error: LDraw parts library not found at {ldraw_dir / 'parts'}")

    # Discover / resolve parts
    if args.top_1000 and TOP_2000_FILE.exists():
        with open(TOP_2000_FILE, "r", encoding="utf-8") as f:
            parts = [line.strip() for line in f if line.strip() and not line.startswith("#")][:1000]
    elif args.top_2000 or (not args.all_parts and not args.parts and not args.parts_file and TOP_2000_FILE.exists()):
        # Default to top 2,000 parts
        with open(TOP_2000_FILE, "r", encoding="utf-8") as f:
            parts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    elif args.all_parts:
        if ALL_PARTS_FILE.exists():
            with open(ALL_PARTS_FILE, "r", encoding="utf-8") as f:
                parts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        else:
            parts = discover_all_parts(ldraw_dir / "parts")
    elif args.parts:
        parts = list(args.parts)
    elif args.parts_file:
        p_path = Path(args.parts_file)
        if not p_path.exists():
            sys.exit(f"Error: parts-file not found at {p_path}")
        with open(p_path, "r", encoding="utf-8") as f:
            parts = [
                line.split("#")[0].split("\t")[0].split()[0].strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    else:
        parts = discover_all_parts(ldraw_dir / "parts")

    # Filter parts starting from
    if args.start_from and args.start_from in parts:
        start_idx = parts.index(args.start_from)
        parts = parts[start_idx:]

    if args.limit and args.limit > 0:
        parts = parts[:args.limit]

    # Verify part files exist
    valid_parts = []
    for p in parts:
        dat_path = ldraw_dir / "parts" / f"{p}.dat"
        if dat_path.exists():
            valid_parts.append(p)
        else:
            if (ldraw_dir / "parts" / f"{p.lower()}.dat").exists():
                valid_parts.append(p.lower())

    if not valid_parts:
        sys.exit("No valid parts found to render.")

    # Load color library
    color_pool = load_ldraw_colors(ldraw_dir / "LDConfig.ldr")

    # Save classes.txt
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        for p in valid_parts:
            f.write(p + "\n")

    print("=" * 70)
    print(" LEGO Machine Vision Top 1,000 Parts Dataset Generator")
    print("=" * 70)
    print(f" Output Directory   : {out_dir}")
    print(f" Total Target Parts : {len(valid_parts):,} parts")
    print(f" Images per Part    : {args.num_images:,}")
    print(f" Total Images       : {len(valid_parts) * args.num_images:,}")
    print(f" Image Resolution   : {args.resolution}x{args.resolution} ({args.format})")
    print(f" Background Mode    : Dynamic Multicolor (Changes Every Image)")
    print(f" Brick Color Mode   : 70% Official Colors + 30% 24-bit Direct RGB Hex")
    print(f" Augmentation       : {'Disabled' if args.no_augment else 'Enabled (Lighting Jitter, Blur, Grain)'}")
    print(f" Parallel Workers   : {args.workers}")
    print("=" * 70)

    total_rendered = 0
    total_target_parts = len(valid_parts)
    total_target_images = total_target_parts * args.num_images

    # Calculate pre-existing images if resuming
    pre_existing_images = 0
    for p in valid_parts:
        p_folder = out_dir / p
        if p_folder.exists():
            pre_existing_images += min(args.num_images, len(list(p_folder.glob(f"*.{args.format}"))))
    
    overall_completed_images = pre_existing_images
    session_start_time = time.time()
    session_rendered = 0

    manifest = {}
    manifest_path = out_dir / "dataset_manifest.json"

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    for part_idx, part_id in enumerate(valid_parts, 1):
        part_folder = out_dir / part_id
        part_folder.mkdir(parents=True, exist_ok=True)

        tasks = []
        for img_idx in range(1, args.num_images + 1):
            out_filename = f"{part_id}_{img_idx:06d}.{args.format}"
            out_filepath = part_folder / out_filename

            if args.resume and out_filepath.exists() and out_filepath.stat().st_size > 0:
                continue

            # Full brick color randomization:
            # 70% authentic LEGO colors, 30% arbitrary 24-bit hex RGB (0x2RRGGBB)
            if random.random() < 0.70:
                color_spec = str(random.choice(color_pool if random.random() < 0.3 else POPULAR_LEGO_COLORS))
            else:
                hex_rgb = f"0x2{random.randint(0, 0xFFFFFF):06X}"
                color_spec = hex_rgb

            cam_lat = random.uniform(30.0, 85.0)
            cam_lon = random.uniform(0.0, 360.0)
            is_resting = random.random() < 0.65

            tasks.append((
                part_id,
                color_spec,
                str(out_filepath),
                args.resolution,
                cam_lat,
                cam_lon,
                is_resting,
                args.bg_mode,
                not args.no_augment,
                str(ldview_bin),
                str(ldraw_dir),
                img_idx % args.workers,
            ))

        if not tasks:
            manifest[part_id] = args.num_images
            pct_now = (overall_completed_images / total_target_images) * 100 if total_target_images > 0 else 100
            print(f"[{part_idx}/{total_target_parts}] Part {part_id}: Already complete (Overall: {pct_now:.1f}%)")
            continue

        part_success = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(render_worker_task, t) for t in tasks]
            with tqdm(total=len(tasks), desc=f"[{part_idx}/{total_target_parts}] Part {part_id}", unit="img") as pbar:
                for future in as_completed(futures):
                    if future.result():
                        part_success += 1
                        session_rendered += 1
                        overall_completed_images += 1
                    pbar.update(1)

                    elapsed = time.time() - session_start_time
                    if session_rendered > 0 and elapsed > 0:
                        cur_speed = session_rendered / elapsed
                        rem_imgs = max(0, total_target_images - overall_completed_images)
                        eta_s = rem_imgs / cur_speed if cur_speed > 0 else 0
                        pct = (overall_completed_images / total_target_images) * 100
                        pbar.set_postfix_str(f"Overall: {pct:.1f}% | Time Left: {format_duration(eta_s)} | {cur_speed:.1f} img/s")

        total_rendered += part_success
        manifest[part_id] = part_success
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    total_elapsed = time.time() - session_start_time
    avg_speed = session_rendered / total_elapsed if total_elapsed > 0 else 0
    print("\n" + "=" * 70)
    print(f" FINISHED RUN! Total images generated this session: {total_rendered:,}")
    print(f" Total time elapsed: {format_duration(total_elapsed)} ({avg_speed:.1f} imgs/sec avg)")
    print(f" Manifest written to: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
