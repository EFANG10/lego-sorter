#!/usr/bin/env python3
"""
LEGO Brick Vision Classifier
Supports ONNX Runtime models, PyTorch checkpoints (ConvNeXt-Nano / MobileNetV3),
and an intelligent mock classifier for testing and hardware calibration.
Performs multi-snapshot crop averaging across the top-5 predictions.
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

logger = logging.getLogger("lego_sorter.classifier")

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class BrickClassifier:
    def __init__(self, catalog_path: Optional[Path] = None, model_path: str = "", labels_path: str = ""):
        self.catalog: Dict[str, Dict[str, Any]] = {}
        if catalog_path and Path(catalog_path).exists():
            try:
                with open(catalog_path, "r", encoding="utf-8") as f:
                    cat_list = json.load(f)
                    for item in cat_list:
                        self.catalog[item["id"].lower()] = item
            except Exception as e:
                logger.error(f"Failed to load catalog into classifier: {e}")

        self._lock = threading.Lock()
        self.model_type = "mock"
        self.model_path = model_path
        self.labels_path = labels_path
        self.classes: List[str] = []
        self.onnx_session = None
        self.torch_model = None
        self.device = torch.device("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu") if HAS_TORCH else None
        self.input_size = (224, 224)
        self.last_inference_time_ms = 0.0

        if model_path:
            self.load_model(model_path, labels_path)
        else:
            self._init_mock()

    def _init_mock(self):
        self.model_type = "mock"
        if not self.classes and self.catalog:
            self.classes = sorted(list(self.catalog.keys()))
        elif not self.classes:
            self.classes = ["3001", "3003", "3004", "3005", "3020", "3022", "3023", "3039", "3040", "3068b", "2780", "3626"]
        logger.info(f"Initialized mock classifier with {len(self.classes)} catalog parts.")

    def load_model(self, model_path: str, labels_path: str = "") -> Tuple[bool, str]:
        path = Path(model_path).resolve()
        if not path.exists():
            self._init_mock()
            return False, f"Model path does not exist: {model_path}. Using mock mode."

        # If directory passed, look for .onnx or .pth inside
        if path.is_dir():
            onnx_files = list(path.glob("*.onnx"))
            pth_files = list(path.glob("*.pth")) + list(path.glob("*.pt"))
            if onnx_files:
                path = onnx_files[0]
            elif pth_files:
                path = pth_files[0]
            else:
                self._init_mock()
                return False, f"No .onnx or .pth model found in folder: {model_path}"

        # Load labels if available
        loaded_classes: List[str] = []
        labels_file = None
        if labels_path and Path(labels_path).exists():
            labels_file = Path(labels_path)
        else:
            # Check adjacent classes.json or classes.txt
            parent = path.parent
            for cand in ["classes.json", "labels.json", "classes.txt", "labels.txt"]:
                if (parent / cand).exists():
                    labels_file = parent / cand
                    break

        if labels_file:
            try:
                if labels_file.suffix == ".json":
                    with open(labels_file, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            loaded_classes = [str(x).lower().strip() for x in data]
                        elif isinstance(data, dict):
                            loaded_classes = [str(v).lower().strip() for k, v in sorted(data.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))]
                else:
                    with open(labels_file, "r") as f:
                        loaded_classes = [line.strip().lower() for line in f if line.strip()]
                logger.info(f"Loaded {len(loaded_classes)} classes from {labels_file}")
            except Exception as e:
                logger.warning(f"Could not parse classes file {labels_file}: {e}")

        # Load ONNX
        if path.suffix.lower() == ".onnx":
            if not HAS_ONNX:
                return False, "onnxruntime is not installed."
            try:
                options = ort.SessionOptions()
                options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
                session = ort.InferenceSession(str(path), options)
                with self._lock:
                    if loaded_classes:
                        self.classes = loaded_classes
                    self.onnx_session = session
                    self.torch_model = None
                    self.model_type = "onnx"
                    self.model_path = str(path)
                logger.info(f"Successfully loaded ONNX model: {path}")
                return True, f"Loaded ONNX model: {path.name}"
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                with self._lock:
                    self._init_mock()
                return False, f"Failed to load ONNX model: {e}"

        # Load PyTorch (.pth or .pt)
        if path.suffix.lower() in [".pth", ".pt"]:
            if not HAS_TORCH:
                return False, "PyTorch is not installed in the environment."
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loaded = torch.load(str(path), map_location=device, weights_only=False)
                torch_model = None

                if isinstance(loaded, torch.nn.Module):
                    torch_model = loaded
                elif isinstance(loaded, dict):
                    ckpt_classes = loaded.get("class_names") or loaded.get("classes")
                    if not loaded_classes and isinstance(ckpt_classes, list):
                        loaded_classes = [str(x).lower().strip() for x in ckpt_classes]
                        logger.info(f"Extracted {len(loaded_classes)} class names from checkpoint.")

                    state_dict = (
                        loaded.get("state_dict")
                        or loaded.get("model_state_dict")
                        or loaded.get("model")
                    )
                    if state_dict is not None and isinstance(state_dict, dict):
                        if not HAS_TIMM:
                            self._init_mock()
                            return False, "timm is not installed to instantiate checkpoint architecture."
                        model_name = loaded.get("model_name", "convnext_nano")
                        num_classes = int(loaded.get("num_classes") or len(loaded_classes) or 1995)
                        clean_sd = {
                            (k[7:] if k.startswith("module.") else k): v
                            for k, v in state_dict.items()
                        }
                        torch_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                        torch_model.load_state_dict(clean_sd, strict=True)
                        logger.info(f"Instantiated {model_name} ({num_classes} classes) from checkpoint.")
                    else:
                        self._init_mock()
                        return False, "Checkpoint dict missing 'state_dict'."
                else:
                    self._init_mock()
                    return False, f"Unsupported PyTorch checkpoint object: {type(loaded).__name__}"

                torch_model.to(device)
                torch_model.eval()
                with self._lock:
                    if loaded_classes:
                        self.classes = loaded_classes
                    self.torch_model = torch_model
                    self.onnx_session = None
                    self.device = device
                    self.model_type = "torch"
                    self.model_path = str(path)
                logger.info(f"Successfully loaded PyTorch model: {path.name} on {device}")
                return True, f"Loaded PyTorch model: {path.name} ({len(self.classes)} classes)"
            except Exception as e:
                logger.error(f"Failed to load PyTorch model: {e}", exc_info=True)
                with self._lock:
                    self._init_mock()
                return False, f"Failed to load PyTorch model: {e}"

        self._init_mock()
        return False, f"Unsupported model format: {path.suffix}"

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Resize, convert BGR to RGB, normalize with standard ImageNet statistics.
        Returns float32 array shaped (1, 3, 224, 224).
        """
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
        img_data = resized.astype(np.float32) / 255.0
        # ImageNet mean & std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        # Transpose HWC -> CHW
        img_data = np.transpose(img_data, (2, 0, 1))
        # Add batch dim -> (1, 3, 224, 224)
        return np.expand_dims(img_data, axis=0)

    def classify_crop(self, crop: np.ndarray, assigned_parts: Optional[List[str]] = None) -> np.ndarray:
        """
        Runs model on single crop and returns 1D array of class probabilities (softmax).
        """
        if self.model_type == "onnx" and self.onnx_session is not None:
            inp = self._preprocess_crop(crop)
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: inp})
            logits = outputs[0][0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            return probs

        elif self.model_type == "torch" and self.torch_model is not None and HAS_TORCH:
            inp = self._preprocess_crop(crop)
            with torch.no_grad():
                tensor = torch.from_numpy(inp).to(self.device)
                out = self.torch_model(tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                return probs

        else:
            # Intelligent Mock Classifier:
            # Uses visual features of crop (aspect ratio, dominant color) + active assigned parts
            # to produce realistic probabilities and predictions for testing
            num_classes = len(self.classes)
            probs = np.full(num_classes, 0.001 / num_classes, dtype=np.float32)
            
            # Extract basic geometry from crop
            h, w = crop.shape[:2]
            aspect = w / max(1, h)
            
            # Prefer active assigned parts if provided, else popular bricks
            candidate_pool = [p for p in (assigned_parts or []) if p in self.classes]
            if not candidate_pool:
                candidate_pool = [c for c in ["3001", "3003", "3004", "3020", "3022", "3039", "3068b", "2780"] if c in self.classes]
            if not candidate_pool:
                candidate_pool = self.classes[:min(10, num_classes)]

            # Deterministic selection based on crop dimensions & checksum
            seed_idx = (int(aspect * 100) + int(np.mean(crop))) % len(candidate_pool)
            winner_part = candidate_pool[seed_idx]
            winner_idx = self.classes.index(winner_part)

            # High primary confidence 82% - 94%
            main_conf = 0.82 + 0.12 * ((hash(f"{aspect}") % 100) / 100.0)
            probs[winner_idx] = main_conf

            # Distribute remaining confidence among runner-ups
            remaining = 1.0 - main_conf
            for i in range(1, min(5, len(candidate_pool))):
                runner_part = candidate_pool[(seed_idx + i) % len(candidate_pool)]
                r_idx = self.classes.index(runner_part)
                portion = remaining * (0.6 / (i + 1))
                probs[r_idx] = portion
                remaining -= portion

            # Normalize to 1.0
            probs = probs / np.sum(probs)
            return probs

    def classify_multi_crop_averaged(
        self,
        crops: List[np.ndarray],
        top_k: int = 5,
        assigned_parts: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Takes several screenshots of a brick as it goes by, sends them to the classifier,
        classifies each crop separately, and averages confidences across all snapshots.
        Returns:
            (top_predictions_list, total_inference_time_ms)
        """
        if not crops:
            return [], 0.0

        t0 = time.perf_counter()
        valid_crops = [c for c in crops if c is not None and c.size > 0 and c.shape[0] >= 5 and c.shape[1] >= 5]
        if not valid_crops:
            return [], 0.0

        with self._lock:
            if self.model_type == "torch" and self.torch_model is not None and HAS_TORCH:
                batch = np.concatenate([self._preprocess_crop(c) for c in valid_crops], axis=0)
                with torch.no_grad():
                    tensor = torch.from_numpy(batch).to(self.device)
                    out = self.torch_model(tensor)
                    probs_batch = torch.softmax(out, dim=1).cpu().numpy()
                avg_probs = np.mean(probs_batch, axis=0)
            else:
                probs_list = [self.classify_crop(c, assigned_parts=assigned_parts) for c in valid_crops]
                avg_probs = np.mean(probs_list, axis=0)

        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        self.last_inference_time_ms = inference_time_ms

        # Get top-k indices
        top_indices = np.argsort(avg_probs)[::-1][:top_k]

        results = []
        for idx in top_indices:
            conf = float(avg_probs[idx])
            part_id = self.classes[idx] if idx < len(self.classes) else f"unknown_{idx}"
            
            # Lookup metadata from catalog
            meta = self.catalog.get(part_id.lower(), {})
            name = meta.get("name", f"LEGO Part {part_id}")
            category = meta.get("category", "Uncategorized")
            thumb = meta.get("thumbnail", f"/static/thumbnails/{part_id}.png")

            results.append({
                "part_id": part_id,
                "name": name,
                "category": category,
                "confidence": round(conf, 4),
                "confidence_percent": round(conf * 100.0, 1),
                "thumbnail": thumb
            })

        return results, inference_time_ms
