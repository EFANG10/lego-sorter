#!/usr/bin/env python3
"""
Multithreaded Motion-Triggered Vision Pipeline for LEGO Sorter
- Thread 1 (Camera Grabber): Continuously reads frames from USB camera without blocking
- Thread 2 (Motion Tracker): Uses MOG2 background subtraction to detect moving bricks,
  collects spaced-out multi-frame snapshots (with spatial/time variance), draws HUD,
  and pre-encodes MJPEG frames for zero-stutter streaming
- Thread 3 (Classifier Worker): Runs PyTorch/ONNX batch inference and Arduino gate
  actuation asynchronously so inference never stalls the live video feed
"""

import cv2
import time
import math
import queue
import fcntl
import struct
import base64
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger("lego_sorter.vision")


class TrackedBrick:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], first_seen: float):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        self.start_centroid = self.centroid
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.history: List[Tuple[int, int]] = [self.centroid]
        self.snapshots: List[np.ndarray] = []
        self.last_snap_centroid: Optional[Tuple[int, int]] = None
        self.last_snap_time: float = 0.0
        self.classified = False
        self.result: Optional[Dict[str, Any]] = None


class VisionPipeline:
    def __init__(self, config_manager, classifier, arduino):
        self.config = config_manager
        self.classifier = classifier
        self.arduino = arduino

        self.running = False
        self.grab_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        self.classify_thread: Optional[threading.Thread] = None

        self.lock = threading.Lock()
        self._raw_lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._classify_queue: queue.Queue = queue.Queue(maxsize=16)

        self.cap: Optional[cv2.VideoCapture] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._raw_has_frame: bool = False
        self._raw_seq: int = 0

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated: Optional[np.ndarray] = None
        self.latest_jpeg: Optional[bytes] = None

        self.fps = 0.0
        self.parts_sorted_count = 0
        self.recent_detections: List[Dict[str, Any]] = []

        self.next_track_id = 1
        self.tracked_objects: Dict[int, TrackedBrick] = {}

        # Motion detector (MOG2)
        self.bg_subtractor = self._create_bg_subtractor()
        self.bg_warmup_frames = 0

        # Performance stats
        self.total_frames = 0
        self.start_time = time.time()
        self.last_cam_retry_time = 0.0
        self.reconnect_requested = False

    @staticmethod
    def _create_bg_subtractor():
        return cv2.createBackgroundSubtractorMOG2(
            history=250,
            varThreshold=36,
            detectShadows=False
        )

    def request_camera_reconnect(self):
        """Thread-safe signal to reopen camera on the grabber thread."""
        self.reconnect_requested = True

    def start(self):
        if self.running:
            return
        self.running = True
        self.grab_thread = threading.Thread(target=self._camera_grab_loop, name="CamGrabber", daemon=True)
        self.process_thread = threading.Thread(target=self._vision_process_loop, name="MotionTracker", daemon=True)
        self.classify_thread = threading.Thread(target=self._classification_worker, name="ClassifierWorker", daemon=True)

        self.grab_thread.start()
        self.process_thread.start()
        self.classify_thread.start()
        logger.info("Multithreaded Vision Pipeline started (Grabber + MotionTracker + ClassifierWorker).")

    def stop(self):
        self.running = False
        self._new_frame_event.set()
        try:
            self._classify_queue.put_nowait(None)
        except queue.Full:
            pass

        for t in (self.grab_thread, self.process_thread, self.classify_thread):
            if t and t.is_alive():
                t.join(timeout=1.5)

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

        # Reset motion background model when switching or reconnecting cameras
        self.bg_subtractor = self._create_bg_subtractor()
        self.bg_warmup_frames = 0
        self.tracked_objects.clear()

        try:
            device_idx = int(source) if str(source).isdigit() else str(source)
            logger.info(f"Connecting to physical USB camera: {device_idx}...")

            cap = cv2.VideoCapture(device_idx, cv2.CAP_V4L2) if isinstance(device_idx, int) else cv2.VideoCapture(device_idx)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(device_idx)

            # Force MJPG before resolution/FPS so USB 2.0 webcams (e.g. Logitech Brio) run at full 30 FPS instead of 7.5 FPS YUYV
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                self._disable_v4l2_dynamic_framerate(device_idx)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.cap = cap
                    fc = int(cap.get(cv2.CAP_PROP_FOURCC))
                    fc_str = "".join([chr((fc >> 8 * i) & 0xFF) for i in range(4)])
                    rep_fps = cap.get(cv2.CAP_PROP_FPS)
                    logger.info(
                        f"Successfully connected to USB camera device {device_idx} "
                        f"({test_frame.shape[1]}x{test_frame.shape[0]} @ {rep_fps:.0f} FPS, {fc_str})"
                    )
                    return
            logger.warning(f"Could not read from camera {device_idx}. Will keep trying to connect...")
            cap.release()
        except Exception as e:
            logger.error(f"Failed to initialize camera source {source}: {e}")
        self.cap = None

    @staticmethod
    def _disable_v4l2_dynamic_framerate(device_idx):
        """
        Disables V4L2_CID_EXPOSURE_AUTO_PRIORITY (exposure_dynamic_framerate = 0) on Linux
        so UVC webcams (e.g., Logitech Brio 105) do not drop from 30 FPS down to 7.5 FPS
        in auto-exposure mode.
        """
        try:
            dev_path = f"/dev/video{device_idx}" if isinstance(device_idx, int) else str(device_idx)
            if not dev_path.startswith("/dev/video"):
                return
            vidioc_s_ctrl = 0xC008561C
            cid_exposure_auto_priority = 0x009A0903
            with open(dev_path, "rb+", buffering=0) as fd:
                fcntl.ioctl(fd, vidioc_s_ctrl, struct.pack("Ii", cid_exposure_auto_priority, 0))
            logger.info(f"Disabled V4L2 dynamic framerate throttling on {dev_path} (locked to target FPS).")
        except Exception:
            pass

    def _get_no_signal_frame(self) -> np.ndarray:
        """Renders clean standby frame when webcam is disconnected or warming up."""
        frame = np.full((480, 640, 3), 20, dtype=np.uint8)
        cv2.line(frame, (320, 0), (320, 480), (35, 35, 35), 1)
        cv2.line(frame, (0, 240), (640, 240), (35, 35, 35), 1)
        source = self.config.settings.get("camera_source", 0)
        cv2.putText(frame, "NO CAMERA SIGNAL", (185, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"Connect USB Webcam (Device {source}) or select port in Settings", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        return frame

    # ----------------------------------------------------
    # Thread 1: Dedicated Camera Frame Grabber
    # ----------------------------------------------------
    def _camera_grab_loop(self):
        self._open_camera()

        while self.running:
            t0 = time.time()

            if self.reconnect_requested:
                self.reconnect_requested = False
                self._open_camera()

            has_frame = False
            frame = None

            if self.cap is not None and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame
                    has_frame = True
                else:
                    self.cap.release()
                    self.cap = None
                    logger.warning("Camera disconnected. Retrying in background...")

            if not has_frame:
                if t0 - self.last_cam_retry_time > 3.0:
                    self.last_cam_retry_time = t0
                    self._open_camera()
                frame = self._get_no_signal_frame()
                time.sleep(0.05)

            with self._raw_lock:
                self._raw_frame = frame
                self._raw_has_frame = has_frame
                self._raw_seq += 1
            self._new_frame_event.set()

    # ----------------------------------------------------
    # Thread 2: Motion Detection, Spaced Capture & Encoding
    # ----------------------------------------------------
    def _vision_process_loop(self):
        last_seq = -1
        prev_time = time.time()
        frame_counter = 0

        while self.running:
            self._new_frame_event.wait(timeout=0.2)
            self._new_frame_event.clear()
            if not self.running:
                break

            with self._raw_lock:
                if self._raw_seq == last_seq or self._raw_frame is None:
                    continue
                last_seq = self._raw_seq
                frame = self._raw_frame.copy()
                has_frame = self._raw_has_frame

            t0 = time.time()
            frame_counter += 1
            dt = t0 - prev_time
            if dt >= 1.0:
                self.fps = round(frame_counter / dt, 1)
                frame_counter = 0
                prev_time = t0

            if has_frame:
                try:
                    annotated = self._process_motion_frame(frame)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}", exc_info=True)
                    annotated = frame.copy()
                    cv2.putText(annotated, f"Vision Error: {str(e)[:45]}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            else:
                annotated = frame.copy()
                cv2.putText(annotated, f"FPS: {self.fps} | Standby", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

            # Encode JPEG once per frame for all MJPEG stream viewers
            ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = buf.tobytes() if ok else None

            with self.lock:
                self.latest_frame = frame
                self.latest_annotated = annotated
                if jpeg_bytes is not None:
                    self.latest_jpeg = jpeg_bytes
                self.total_frames += 1

    def _process_motion_frame(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. Motion-triggered foreground mask via MOG2 background subtraction
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = self.bg_subtractor.apply(blurred, learningRate=0.005)
        self.bg_warmup_frames += 1

        # Clean up noise and merge nearby brick features into solid blobs
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k_open, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=2)
        cleaned = cv2.dilate(cleaned, k_open, iterations=1)

        min_area = self.config.settings.get("min_contour_area", 800)
        max_area = self.config.settings.get("max_contour_area", 120000)
        snapshot_target = max(1, int(self.config.settings.get("snapshot_count", 5)))

        detected_bboxes = []
        # Skip contour extraction during initial 15-frame background warmup
        if self.bg_warmup_frames >= 15:
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw < w * 0.85 and bh < h * 0.85:
                        detected_bboxes.append((x, y, bw, bh))

        # 2. Centroid tracking & spaced-out multi-snapshot capture
        current_time = time.time()
        updated_tracks = set()

        # Spacing thresholds so the 5 frames have distinct belt positions/angles
        min_step_px = 18.0
        min_step_time = 0.11
        min_total_motion_px = 16.0

        for (x, y, bw, bh) in detected_bboxes:
            cx = x + bw // 2
            cy = y + bh // 2

            best_id = None
            min_dist = 90.0
            for tid, track in self.tracked_objects.items():
                if tid in updated_tracks:
                    continue
                tcx, tcy = track.centroid
                dist = math.hypot(cx - tcx, cy - tcy)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid

            if best_id is None:
                best_id = self.next_track_id
                self.next_track_id += 1
                self.tracked_objects[best_id] = TrackedBrick(best_id, (x, y, bw, bh), current_time)

            track = self.tracked_objects[best_id]
            track.bbox = (x, y, bw, bh)
            track.centroid = (cx, cy)
            track.last_seen = current_time
            track.history.append((cx, cy))
            updated_tracks.add(best_id)

            total_displacement = math.hypot(cx - track.start_centroid[0], cy - track.start_centroid[1])
            margin = 12
            is_fully_in_frame = (x >= margin and (x + bw) <= (w - margin) and y >= margin and (y + bh) <= (h - margin))
            near_exit_edge = ((x <= margin) or ((x + bw) >= (w - margin)) or (y <= margin) or ((y + bh) >= (h - margin)))

            pad = 6
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            crop = frame[y1:y2, x1:x2].copy()

            if is_fully_in_frame and not track.classified:
                if crop.shape[0] > 10 and crop.shape[1] > 10 and len(track.snapshots) < snapshot_target:
                    should_capture = False
                    if len(track.snapshots) == 0:
                        should_capture = True
                    else:
                        step_dist = math.hypot(cx - track.last_snap_centroid[0], cy - track.last_snap_centroid[1])
                        step_dt = current_time - track.last_snap_time
                        # Capture next frame only after brick has moved meaningfully or enough time+motion elapsed
                        if step_dist >= min_step_px or (step_dt >= min_step_time and step_dist >= 4.0):
                            should_capture = True

                    if should_capture:
                        track.snapshots.append(crop)
                        track.last_snap_centroid = (cx, cy)
                        track.last_snap_time = current_time

                # Queue for async classification once we have spaced snapshots and verified motion
                if len(track.snapshots) >= snapshot_target and total_displacement >= min_total_motion_px:
                    self._enqueue_classification(track)
            elif not track.classified and len(track.snapshots) >= 2 and near_exit_edge and total_displacement >= min_total_motion_px:
                # Brick is exiting frame with >= 2 spaced snapshots -> classify before it leaves
                self._enqueue_classification(track)

            # 3. Draw bounding boxes & tracking info
            with self.lock:
                res = track.result

            color = (0, 255, 0) if not track.classified else (255, 180, 0)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

            if len(track.history) > 1:
                pts = np.array(track.history[-15:], np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], False, (0, 200, 255), 1)

            if res:
                label = f"#{res['part_id']} -> Bin {res['bin_id']} ({res['confidence_pct']}%)"
                cv2.putText(annotated, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            elif track.classified:
                label = f"ID:{track.track_id} Classifying ({len(track.snapshots)} snaps)..."
                cv2.putText(annotated, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 0), 1)
            else:
                label = f"ID:{track.track_id} Motion ({len(track.snapshots)}/{snapshot_target} snaps)"
                cv2.putText(annotated, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Remove stale tracks that disappeared
        stale_ids = [tid for tid, t in self.tracked_objects.items() if current_time - t.last_seen > 1.2]
        for tid in stale_ids:
            del self.tracked_objects[tid]

        # Draw HUD banner
        source = self.config.settings.get("camera_source", 0)
        warmup_tag = " | BG Warmup..." if self.bg_warmup_frames < 15 else " | Motion Trigger"
        cv2.putText(annotated, f"FPS: {self.fps} | Sorted: {self.parts_sorted_count}{warmup_tag}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(annotated, f"CAMERA: /dev/video{source}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1)

        return annotated

    def _enqueue_classification(self, track: TrackedBrick):
        if track.classified or not track.snapshots:
            return
        track.classified = True
        try:
            self._classify_queue.put_nowait((track, list(track.snapshots)))
        except queue.Full:
            logger.warning(f"Classification queue full; dropping track {track.track_id}.")

    # ----------------------------------------------------
    # Thread 3: Async Classifier & Arduino Gate Worker
    # ----------------------------------------------------
    def _classification_worker(self):
        while self.running:
            try:
                item = self._classify_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            track, snapshots = item
            try:
                self._classify_and_route(track, snapshots)
            except Exception as e:
                logger.error(f"Classification worker error for track {track.track_id}: {e}", exc_info=True)
            finally:
                self._classify_queue.task_done()

    def _classify_and_route(self, track: TrackedBrick, snapshots: List[np.ndarray]):
        """
        Classifies the collected snapshots, averages confidences, routes to bin,
        and dispatches USB command to Arduino with per-bin hardware travel delay.
        """
        if not snapshots:
            return

        active_parts = []
        for b in self.config.bins:
            active_parts.extend(b.get("assigned_parts", []))

        top_k = self.config.settings.get("top_k", 5)
        results, latency_ms = self.classifier.classify_multi_crop_averaged(
            snapshots,
            top_k=top_k,
            assigned_parts=active_parts
        )

        if not results:
            return

        winner = results[0]
        part_id = winner["part_id"]
        confidence = winner["confidence"]

        bin_id = self.config.get_bin_for_part(part_id, confidence)
        self.config.increment_bin_count(bin_id)

        bin_dict = self.config.get_bin_dict(bin_id)
        delay_ms = bin_dict.get("delay_ms", 0) if bin_dict else 0
        pulse_ms = bin_dict.get("pulse_ms", 350) if bin_dict else 350
        spot = bin_dict.get("spot", 0) if bin_dict else 0
        side = bin_dict.get("side", "top") if bin_dict else "top"
        row = bin_dict.get("row", 0) if bin_dict else 0

        serial_ok, serial_msg = self.arduino.send_bin(
            bin_id,
            delay_ms=delay_ms,
            pulse_ms=pulse_ms,
            spot=spot,
            row=row
        )

        captured_crops_b64 = []
        for sn in snapshots[:5]:
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
            "snapshots_taken": len(snapshots),
            "inference_time_ms": round(latency_ms, 1),
            "serial_ok": serial_ok,
            "serial_msg": serial_msg
        }

        with self.lock:
            self.parts_sorted_count += 1
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
            if self.latest_jpeg is not None:
                return self.latest_jpeg
        blank = self._get_no_signal_frame()
        _, buffer = cv2.imencode('.jpg', blank)
        return buffer.tobytes()
