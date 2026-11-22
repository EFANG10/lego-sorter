#!/usr/bin/env python3
"""
Vision Pipeline and Conveyor Tracker for LEGO Sorter
- Captures from physical USB camera (/dev/video*)
- Uses OpenCV for contour-based object localization and bounding box extraction
- Collects multiple snapshot crops as a brick passes the trigger line
- Dispatches crops to the classifier for top-5 averaged confidence inference
- Sends target bin actuation command to Arduino with per-bin hardware travel delay
- Emits real-time MJPEG video stream with bounding boxes and detection overlays
"""

import cv2
import time
import math
import base64
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path

logger = logging.getLogger("lego_sorter.vision")


class TrackedBrick:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], first_seen: float):
        self.track_id = track_id
        self.bbox = bbox # (x, y, w, h)
        self.centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.history: List[Tuple[int, int]] = [self.centroid]
        self.snapshots: List[np.ndarray] = []
        self.classified = False
        self.trigger_crossed = False
        self.result: Optional[Dict[str, Any]] = None


class VisionPipeline:
    def __init__(self, config_manager, classifier, arduino):
        self.config = config_manager
        self.classifier = classifier
        self.arduino = arduino

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated: Optional[np.ndarray] = None
        self.fps = 0.0
        self.parts_sorted_count = 0
        self.recent_detections: List[Dict[str, Any]] = []

        self.next_track_id = 1
        self.tracked_objects: Dict[int, TrackedBrick] = {}
        
        # Performance stats
        self.total_frames = 0
        self.start_time = time.time()
        self.last_cam_retry_time = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Vision Pipeline started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.5)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("Vision Pipeline stopped and camera released.")

    def _open_camera(self):
        source = self.config.settings.get("camera_source", 0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        try:
            device_idx = int(source) if str(source).isdigit() else str(source)
            logger.info(f"Connecting to physical USB camera: {device_idx}...")
            cap = cv2.VideoCapture(device_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.cap = cap
                    logger.info(f"Successfully connected to USB camera device {device_idx} ({test_frame.shape[1]}x{test_frame.shape[0]})")
                    return
            logger.warning(f"Could not read from camera {device_idx}. Will keep trying to connect...")
            cap.release()
        except Exception as e:
            logger.error(f"Failed to initialize camera source {source}: {e}")
        self.cap = None

    def _get_no_signal_frame(self) -> np.ndarray:
        """Renders clean standby frame when webcam is disconnected or warming up."""
        frame = np.full((480, 640, 3), 20, dtype=np.uint8)
        # Subtle crosshair
        cv2.line(frame, (320, 0), (320, 480), (35, 35, 35), 1)
        cv2.line(frame, (0, 240), (640, 240), (35, 35, 35), 1)
        # Warning text
        source = self.config.settings.get("camera_source", 0)
        cv2.putText(frame, "NO CAMERA SIGNAL", (185, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"Connect USB Webcam (Device {source}) or select port in Settings", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        return frame

    def _capture_loop(self):
        self._open_camera()
        prev_time = time.time()
        frame_counter = 0

        while self.running:
            t0 = time.time()

            # 1. Grab raw frame from camera
            has_frame = False
            frame = None

            if self.cap is not None and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame
                    has_frame = True
                else:
                    # Camera lost connection
                    self.cap.release()
                    self.cap = None
                    logger.warning("Camera disconnected. Retrying in background...")

            # Periodic camera reconnect attempt if disconnected
            if not has_frame:
                if t0 - self.last_cam_retry_time > 3.0:
                    self.last_cam_retry_time = t0
                    self._open_camera()
                frame = self._get_no_signal_frame()

            frame_counter += 1
            dt = t0 - prev_time
            if dt >= 1.0:
                self.fps = round(frame_counter / dt, 1)
                frame_counter = 0
                prev_time = t0

            # 2. Process frame with OpenCV (only if live frame is present)
            if has_frame:
                annotated = self._process_frame(frame)
            else:
                annotated = frame.copy()
                cv2.putText(annotated, f"FPS: {self.fps} | Standby", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

            with self.lock:
                self.latest_frame = frame
                self.latest_annotated = annotated
                self.total_frames += 1

            # Cap loop to ~30 FPS
            elapsed = time.time() - t0
            sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. Preprocess for OpenCV contour detection (Method 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh_val = self.config.settings.get("threshold_val", 120)
        mode = self.config.settings.get("threshold_mode", "adaptive")

        if mode == "adaptive":
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 6)
        else:
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        dilated = cv2.dilate(opened, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.config.settings.get("min_contour_area", 800)
        max_area = self.config.settings.get("max_contour_area", 120000)
        snapshot_target = self.config.settings.get("snapshot_count", 5)

        detected_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < w * 0.9 and bh < h * 0.9:
                    detected_bboxes.append((x, y, bw, bh))

        # 2. Centroid tracking and snapshot collection
        current_time = time.time()
        updated_tracks = set()

        for (x, y, bw, bh) in detected_bboxes:
            cx = x + bw // 2
            cy = y + bh // 2

            # Match to existing track by nearest centroid
            best_id = None
            min_dist = 85.0

            for tid, track in self.tracked_objects.items():
                if tid in updated_tracks:
                    continue
                tcx, tcy = track.centroid
                dist = math.hypot(cx - tcx, cy - tcy)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid

            if best_id is None:
                # New track
                best_id = self.next_track_id
                self.next_track_id += 1
                self.tracked_objects[best_id] = TrackedBrick(best_id, (x, y, bw, bh), current_time)

            track = self.tracked_objects[best_id]
            track.bbox = (x, y, bw, bh)
            track.centroid = (cx, cy)
            track.last_seen = current_time
            track.history.append((cx, cy))
            updated_tracks.add(best_id)

            # Check if entire brick bounding box is fully inside the frame
            margin = 12
            is_fully_in_frame = (x >= margin and (x + bw) <= (w - margin) and y >= margin and (y + bh) <= (h - margin))

            # Extract snapshot crop if brick has good margin and is fully inside the frame
            pad = 6
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            crop = frame[y1:y2, x1:x2].copy()

            if is_fully_in_frame and not track.classified:
                if crop.shape[0] > 10 and crop.shape[1] > 10 and len(track.snapshots) < snapshot_target:
                    track.snapshots.append(crop)

                # Once target number of snapshots (e.g. 5) are collected while fully in frame, classify!
                if len(track.snapshots) >= snapshot_target:
                    self._classify_and_route(track)
            elif not track.classified and len(track.snapshots) >= 2 and (x + bw) >= (w - margin):
                # Brick is about to exit view frame -> classify with collected snapshots
                self._classify_and_route(track)

            # 3. Draw bounding boxes & tracking info
            color = (0, 255, 0) if not track.classified else (255, 180, 0)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
            
            # Tracking history line
            if len(track.history) > 1:
                pts = np.array(track.history[-15:], np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], False, (0, 200, 255), 1)

            # Label overlay
            if track.result:
                label = f"#{track.result['part_id']} -> Bin {track.result['bin_id']} ({track.result['confidence_pct']}%)"
                cv2.putText(annotated, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            else:
                label = f"ID:{track.track_id} (Captured {len(track.snapshots)}/{snapshot_target} snaps)"
                cv2.putText(annotated, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Remove stale tracks that disappeared
        stale_ids = [tid for tid, t in self.tracked_objects.items() if current_time - t.last_seen > 1.2]
        for tid in stale_ids:
            del self.tracked_objects[tid]

        # Draw HUD banner
        source = self.config.settings.get("camera_source", 0)
        cv2.putText(annotated, f"FPS: {self.fps} | Sorted: {self.parts_sorted_count}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"CAMERA: /dev/video{source}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1)

        return annotated

    def _classify_and_route(self, track: TrackedBrick):
        """
        Classifies the collected snapshots, averages confidences, routes to bin,
        and dispatches USB command to Arduino with per-bin hardware travel delay.
        """
        if not track.snapshots:
            return

        track.classified = True
        
        active_parts = []
        for b in self.config.bins:
            active_parts.extend(b.get("assigned_parts", []))

        # Run multi-crop averaged classifier
        top_k = self.config.settings.get("top_k", 5)
        results, latency_ms = self.classifier.classify_multi_crop_averaged(
            track.snapshots,
            top_k=top_k,
            assigned_parts=active_parts
        )

        if not results:
            return

        winner = results[0]
        part_id = winner["part_id"]
        confidence = winner["confidence"]

        # Route to physical bin or Bin -1
        bin_id = self.config.get_bin_for_part(part_id, confidence)
        self.config.increment_bin_count(bin_id)
        self.parts_sorted_count += 1

        bin_dict = self.config.get_bin_dict(bin_id)
        delay_ms = bin_dict.get("delay_ms", 0) if bin_dict else 0
        pulse_ms = bin_dict.get("pulse_ms", 350) if bin_dict else 350
        spot = bin_dict.get("spot", 0) if bin_dict else 0
        side = bin_dict.get("side", "top") if bin_dict else "top"
        row = bin_dict.get("row", 0) if bin_dict else 0

        # Dispatch Arduino gate control with per-bin hardware travel delay
        serial_ok, serial_msg = self.arduino.send_bin(
            bin_id,
            delay_ms=delay_ms,
            pulse_ms=pulse_ms,
            spot=spot,
            row=row
        )

        # Encode the actual camera snapshots sent to classifier as base64 JPEG data URLs
        captured_crops_b64 = []
        for sn in track.snapshots[:5]:
            if sn is not None and sn.size > 0:
                try:
                    _, buf = cv2.imencode('.jpg', sn, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    b64 = base64.b64encode(buf).decode('ascii')
                    captured_crops_b64.append(f"data:image/jpeg;base64,{b64}")
                except Exception:
                    pass

        detection_event = {
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S"),
            "track_id": track.track_id,
            "part_id": part_id,
            "name": winner["name"],
            "category": winner["category"],
            "confidence": confidence,
            "confidence_pct": winner["confidence_percent"],
            "bin_id": bin_id,
            "spot": spot,
            "side": side,
            "row": row,
            "delay_ms": delay_ms,
            "thumbnail": winner["thumbnail"],
            "captured_crops": captured_crops_b64,
            "top_5": results,
            "snapshots_taken": len(track.snapshots),
            "inference_time_ms": round(latency_ms, 1),
            "serial_ok": serial_ok,
            "serial_msg": serial_msg
        }

        track.result = {
            "part_id": part_id,
            "bin_id": bin_id,
            "confidence_pct": winner["confidence_percent"]
        }

        self.recent_detections.insert(0, detection_event)
        if len(self.recent_detections) > 80:
            self.recent_detections.pop()

        logger.info(
            f"SORTED: Part {part_id} ({winner['name']}) -> Bin {bin_id} (Spot {spot}, Row {row}) "
            f"| Conf: {winner['confidence_percent']}% | Delay: {delay_ms}ms | Latency: {latency_ms:.1f}ms"
        )

    def get_jpeg_frame(self) -> bytes:
        with self.lock:
            if self.latest_annotated is not None:
                _, buffer = cv2.imencode('.jpg', self.latest_annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buffer.tobytes()
            # Standby blank frame
            blank = self._get_no_signal_frame()
            _, buffer = cv2.imencode('.jpg', blank)
            return buffer.tobytes()
