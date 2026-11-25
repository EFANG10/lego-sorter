#!/usr/bin/env python3
"""
Main FastAPI Backend Server for LEGO Sorter Web Application
Serves REST API, MJPEG camera stream, metrics, and responsive frontend UI.
Supports dynamic bin creation/removal, drag-and-drop assignment, per-bin delays,
and clean shutdown with port unbinding.
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add workspace directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config_manager import ConfigManager
from backend.arduino_serial import ArduinoController
from backend.classifier import BrickClassifier
from backend.vision_pipeline import VisionPipeline

# Setup Log Handler for UI Logs page
class RingBufferLogHandler(logging.Handler):
    def __init__(self, capacity=500):
        super().__init__()
        self.capacity = capacity
        self.buffer = []

    def emit(self, record):
        try:
            msg = self.format(record)
            entry = {
                "timestamp": record.created,
                "time_str": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": msg
            }
            self.buffer.append(entry)
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)
        except Exception:
            pass

log_handler = RingBufferLogHandler(500)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
log_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(log_handler)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger("lego_sorter.main")

# Initialize app
app = FastAPI(title="Autonomous LEGO Sorter", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core subsystems
config_mgr = ConfigManager(base_dir=BASE_DIR)
arduino_ctl = ArduinoController(
    port=config_mgr.settings.get("arduino_port", "MOCK"),
    baud_rate=config_mgr.settings.get("arduino_baud", 115200),
    pulse_ms=config_mgr.settings.get("gate_pulse_ms", 350)
)
classifier = BrickClassifier(
    catalog_path=BASE_DIR / "data" / "parts_catalog.json",
    model_path=config_mgr.settings.get("model_path", ""),
    labels_path=config_mgr.settings.get("labels_path", "")
)
pipeline = VisionPipeline(config_mgr, classifier, arduino_ctl)

# Start vision pipeline thread
pipeline.start()

# Mount static files
static_dir = BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Pydantic schemas
class SettingsUpdate(BaseModel):
    camera_source: Optional[Any] = None
    threshold_mode: Optional[str] = None
    threshold_val: Optional[int] = None
    min_contour_area: Optional[int] = None
    max_contour_area: Optional[int] = None
    snapshot_count: Optional[int] = None
    trigger_line_x: Optional[float] = None
    arduino_port: Optional[str] = None
    arduino_baud: Optional[int] = None
    gate_pulse_ms: Optional[int] = None
    model_path: Optional[str] = None
    labels_path: Optional[str] = None
    confidence_threshold: Optional[float] = None
    top_k: Optional[int] = None
    chute_rows_per_spot: Optional[int] = None

class BinCreate(BaseModel):
    name: Optional[str] = None
    spot: Optional[int] = None
    side: Optional[str] = None
    row: Optional[int] = None
    delay_ms: Optional[int] = None

class BinSingleUpdate(BaseModel):
    id: int
    name: Optional[str] = None
    color: Optional[str] = None
    spot: Optional[int] = None
    side: Optional[str] = None
    row: Optional[int] = None
    delay_ms: Optional[int] = None
    pulse_ms: Optional[int] = None
    assigned_parts: Optional[List[str]] = None

class PresetRequest(BaseModel):
    name: str

class PartAssignment(BaseModel):
    bin_id: int
    part_id: str

class TestTrigger(BaseModel):
    bin_id: int


# Clean cleanup on process exit
def cleanup_resources():
    logger.info("Cleaning up vision pipeline and serial port...")
    pipeline.stop()
    arduino_ctl.close()

import atexit
atexit.register(cleanup_resources)


# Video Stream Generator
def mjpeg_generator():
    while True:
        frame_bytes = pipeline.get_jpeg_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033) # ~30 fps cap for stream


@app.get("/")
def serve_index():
    index_file = BASE_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"status": "Frontend loading..."})


@app.get("/api/stream")
def video_stream():
    """Live MJPEG video stream with bounding boxes and HUD overlay."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "vision_fps": pipeline.fps,
        "camera_source": config_mgr.settings.get("camera_source", 0),
        "is_camera_live": pipeline.cap is not None and pipeline.cap.isOpened(),
        "arduino_port": arduino_ctl.port_name,
        "is_arduino_connected": arduino_ctl.is_connected,
        "is_arduino_mock": arduino_ctl.is_mock,
        "classifier_type": classifier.model_type,
        "model_loaded": classifier.model_path if classifier.model_path else "Mock / Calibration Mode",
        "total_classes": len(classifier.classes),
        "total_sorted": pipeline.parts_sorted_count
    }


@app.get("/api/bins")
def get_bins():
    return {
        "bins": config_mgr.bins,
        "num_conveyor_spots": config_mgr.settings.get("num_conveyor_spots", 3),
        "confidence_threshold": config_mgr.settings.get("confidence_threshold", 0.35),
        "chute_rows_per_spot": config_mgr.settings.get("chute_rows_per_spot", 2)
    }


@app.post("/api/bins/add")
def add_new_bin(payload: BinCreate):
    new_bin = config_mgr.add_bin(
        name=payload.name,
        spot=payload.spot,
        side=payload.side,
        row=payload.row,
        delay_ms=payload.delay_ms
    )
    return {
        "status": "ok",
        "bin": new_bin,
        "bins": config_mgr.bins,
        "num_conveyor_spots": config_mgr.settings.get("num_conveyor_spots", 3)
    }


@app.delete("/api/bins/{bin_id}")
def delete_bin(bin_id: int):
    ok = config_mgr.delete_bin(bin_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot delete Bin -1 or bin does not exist")
    return {"status": "ok", "bins": config_mgr.bins}


@app.post("/api/bins/{bin_id}/clear")
def clear_single_bin(bin_id: int):
    ok = config_mgr.clear_bin(bin_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Bin not found")
    return {"status": "ok", "bins": config_mgr.bins}


@app.post("/api/bins/clear_all")
def clear_all_bins():
    config_mgr.clear_all_bins()
    return {"status": "ok", "bins": config_mgr.bins}


@app.post("/api/bins/remove_all")
def remove_all_bins():
    config_mgr.remove_all_bins()
    return {"status": "ok", "bins": config_mgr.bins}


@app.post("/api/spots/add")
def add_conveyor_spot():
    num_spots = config_mgr.add_conveyor_spot()
    return {"status": "ok", "num_conveyor_spots": num_spots, "bins": config_mgr.bins}


@app.post("/api/spots/remove")
def remove_conveyor_spot():
    num_spots = config_mgr.remove_conveyor_spot()
    return {"status": "ok", "num_conveyor_spots": num_spots, "bins": config_mgr.bins}


@app.get("/api/presets")
def get_presets():
    return {
        "presets": config_mgr.list_presets(),
        "active": config_mgr.settings.get("active_preset", "")
    }


@app.post("/api/presets/save")
def save_preset(payload: PresetRequest):
    ok = config_mgr.save_preset(payload.name)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to save preset")
    return {"status": "ok", "preset": payload.name, "presets": config_mgr.list_presets()}


@app.post("/api/presets/load")
def load_preset(payload: PresetRequest):
    ok = config_mgr.load_preset(payload.name)
    if not ok:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {
        "status": "ok",
        "preset": payload.name,
        "bins": config_mgr.bins,
        "num_conveyor_spots": config_mgr.settings.get("num_conveyor_spots", 3)
    }


@app.delete("/api/presets/{name}")
def delete_preset(name: str):
    ok = config_mgr.delete_preset(name)
    return {"status": "ok" if ok else "error", "presets": config_mgr.list_presets()}


@app.post("/api/bins/update_bin")
def update_single_bin(payload: BinSingleUpdate):
    data = payload.dict(exclude_unset=True)
    bid = data.pop("id")
    updated = config_mgr.update_bin(bid, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Bin not found")
    return {"status": "ok", "bin": updated, "bins": config_mgr.bins}


@app.post("/api/bins/assign_part")
def assign_part(payload: PartAssignment):
    pid = payload.part_id.lower().strip()
    target_bin = config_mgr.get_bin_dict(payload.bin_id)
    if not target_bin:
        raise HTTPException(status_code=404, detail="Target bin not found")

    # Remove from any existing bin first
    for b in config_mgr.bins:
        if pid in [p.lower().strip() for p in b.get("assigned_parts", [])]:
            b["assigned_parts"] = [p for p in b["assigned_parts"] if p.lower().strip() != pid]

    # Add to target bin
    if pid not in target_bin["assigned_parts"]:
        target_bin["assigned_parts"].append(pid)

    config_mgr.save_current_bins()
    return {"status": "ok", "bins": config_mgr.bins}


@app.post("/api/bins/remove_part")
def remove_part(payload: PartAssignment):
    pid = payload.part_id.lower().strip()
    target_bin = config_mgr.get_bin_dict(payload.bin_id)
    if target_bin:
        target_bin["assigned_parts"] = [p for p in target_bin.get("assigned_parts", []) if p.lower().strip() != pid]
        config_mgr.save_current_bins()
    return {"status": "ok", "bins": config_mgr.bins}


@app.post("/api/bins/reset_counts")
def reset_counts():
    config_mgr.reset_bin_counts()
    pipeline.parts_sorted_count = 0
    logger.info("Reset session bin counters to zero.")
    return {"status": "ok", "bins": config_mgr.bins}


@app.get("/api/catalog/unassigned")
def get_unassigned_catalog(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(150, ge=1, le=2000)
):
    return config_mgr.get_unassigned_parts(category=category, search=search, page=page, limit=limit)


@app.get("/api/catalog")
def get_catalog(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(48, ge=1, le=200)
):
    items = config_mgr.catalog
    if category and category.lower() != "all":
        items = [x for x in items if x.get("category", "").lower() == category.lower()]

    if search:
        s = search.lower().strip()
        items = [x for x in items if s in x.get("id", "").lower() or s in x.get("name", "").lower()]

    total = len(items)
    start = (page - 1) * limit
    end = start + limit
    paginated = items[start:end]

    return {
        "total": total,
        "page": page,
        "pages": max(1, (total + limit - 1) // limit),
        "items": paginated
    }


@app.get("/api/catalog/categories")
def get_catalog_categories():
    cat_counts = {}
    for item in config_mgr.catalog:
        c = item.get("category", "Special & Other")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    
    categories = [{"name": "All", "count": len(config_mgr.catalog)}]
    for c, count in sorted(cat_counts.items(), key=lambda x: x[0]):
        categories.append({"name": c, "count": count})
    return categories


@app.get("/api/settings")
def get_settings():
    return {
        "settings": config_mgr.settings,
        "serial_ports": arduino_ctl.get_available_ports(),
        "available_cameras": [
            {"id": 0, "name": "USB Camera 0 (/dev/video0)"},
            {"id": 1, "name": "USB Camera 1 (/dev/video1)"},
            {"id": 2, "name": "USB Camera 2 (/dev/video2)"},
            {"id": 3, "name": "USB Camera 3 (/dev/video3)"}
        ]
    }


@app.post("/api/settings")
def update_settings(payload: SettingsUpdate):
    data = payload.dict(exclude_unset=True)
    new_settings = config_mgr.save_settings(data)

    if "arduino_port" in data or "arduino_baud" in data:
        arduino_ctl.connect(
            port=new_settings.get("arduino_port", "MOCK"),
            baud=new_settings.get("arduino_baud", 115200)
        )

    if "camera_source" in data:
        pipeline._open_camera()

    if "model_path" in data or "labels_path" in data:
        if new_settings.get("model_path"):
            success, msg = classifier.load_model(
                new_settings["model_path"],
                new_settings.get("labels_path", "")
            )
            logger.info(f"Model load attempt: {msg}")

    return {"status": "ok", "settings": new_settings}


@app.post("/api/test/trigger_bin")
def test_trigger_bin(payload: TestTrigger):
    bin_dict = config_mgr.get_bin_dict(payload.bin_id)
    delay_ms = bin_dict.get("delay_ms", 0) if bin_dict else 0
    pulse_ms = bin_dict.get("pulse_ms", 350) if bin_dict else 350
    spot = bin_dict.get("spot", 0) if bin_dict else 0
    row = bin_dict.get("row", 0) if bin_dict else 0

    ok, msg = arduino_ctl.send_bin(payload.bin_id, delay_ms=0, pulse_ms=pulse_ms, spot=spot, row=row)
    return {"status": "ok" if ok else "error", "message": msg, "bin_id": payload.bin_id}


@app.get("/api/metrics")
def get_metrics():
    uptime_sec = max(1.0, time.time() - pipeline.start_time)
    ppm = round((pipeline.parts_sorted_count / uptime_sec) * 60.0, 1)

    bin_counts = {b["id"]: b.get("count", 0) for b in config_mgr.bins}
    reject_count = bin_counts.get(-1, 0)
    rejection_rate = round((reject_count / max(1, pipeline.parts_sorted_count)) * 100.0, 1)

    return {
        "fps": pipeline.fps,
        "inference_latency_ms": round(classifier.last_inference_time_ms, 1),
        "total_sorted": pipeline.parts_sorted_count,
        "ppm": ppm,
        "uptime_seconds": round(uptime_sec, 0),
        "rejection_rate_percent": rejection_rate,
        "bin_distribution": bin_counts,
        "recent_detections": pipeline.recent_detections[:20]
    }


@app.get("/api/logs")
def get_logs(level: Optional[str] = Query(None)):
    logs = log_handler.buffer
    if level and level.upper() != "ALL":
        lvl = level.upper()
        logs = [e for e in logs if e["level"] == lvl]
    return {"logs": logs[-150:]}


@app.get("/api/browse_path")
def browse_path(path: str = Query("/home/ethan/Documents/lego sorter")):
    p = Path(path).resolve()
    if not p.exists():
        p = Path.home()
    if p.is_file():
        p = p.parent

    entries = []
    try:
        for item in sorted(p.iterdir()):
            entries.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0
            })
    except PermissionError:
        pass

    return {
        "current_path": str(p),
        "parent_path": str(p.parent) if p.parent != p else None,
        "entries": entries
    }


@app.post("/api/control/shutdown")
def shutdown_system(background_tasks: BackgroundTasks):
    """
    Shuts down camera capture, closes serial connection, unbinds the port,
    and terminates the process.
    """
    logger.info("Shutdown command received. Terminating services and releasing network port...")
    cleanup_resources()

    def terminate_proc():
        time.sleep(0.6)
        logger.info("Unbinding port and exiting.")
        os.kill(os.getpid(), signal.SIGTERM)

    background_tasks.add_task(terminate_proc)
    return {"status": "ok", "message": "System shutdown initiated. Port unbinding in progress."}


if __name__ == "__main__":
    import uvicorn
    from backend.window_launcher import launch_native_window

    parser = argparse.ArgumentParser(description="Autonomous LEGO Sorter Server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind (default: 0.0.0.0)")
    parser.add_argument("--no-window", action="store_true", help="Disable launching standalone GUI window (server-only mode)")
    args = parser.parse_args()

    if not args.no_window:
        # Launch dedicated native GUI window once the server is ready
        threading.Thread(
            target=launch_native_window,
            kwargs={"port": args.port, "host": args.host},
            daemon=True
        ).start()

    logger.info(f"Starting server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)
