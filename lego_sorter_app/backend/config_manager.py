#!/usr/bin/env python3
"""
Configuration and Bin Layout Manager for LEGO Sorter
Handles persistent settings, dynamic bin addition/removal, individual hardware delays,
conveyor two-sided chute mapping (Top & Bottom sides, Spots & Rows), presets, and clearing.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger("lego_sorter.config")

class ConfigManager:
    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.config_dir = self.base_dir / "config"
        self.presets_dir = self.config_dir / "presets"
        self.settings_file = self.config_dir / "settings.json"
        self.bins_file = self.config_dir / "current_bins.json"
        self.catalog_file = self.base_dir / "data" / "parts_catalog.json"

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

        self.settings = self._load_default_settings()
        self.bins = self._load_default_bins()
        self.catalog = self._load_catalog()

        # Load persisted settings and bins
        self.load_settings()
        self.load_current_bins()

    def _load_default_settings(self) -> Dict[str, Any]:
        return {
            "camera_source": 0, # Physical USB Camera device index 0 (/dev/video0)
            "camera_resolution": [640, 480],
            "camera_fps": 30,
            "threshold_mode": "adaptive", # "adaptive" or "binary"
            "threshold_val": 120,
            "min_contour_area": 800,
            "max_contour_area": 120000,
            "snapshot_count": 5,
            "trigger_line_x": 0.5, # normalized 0.0 to 1.0 along conveyor axis
            "arduino_port": "MOCK", # e.g. "/dev/ttyUSB0", "/dev/ttyACM0", or "MOCK"
            "arduino_baud": 115200,
            "gate_pulse_ms": 350,
            "model_type": "mock", # "onnx", "torch", or "mock"
            "model_path": "",
            "labels_path": "",
            "top_k": 5,
            "confidence_threshold": 0.35, # below this sends to bin -1
            "num_conveyor_spots": 3, # Number of conveyor gate spots along the belt
            "chute_rows_per_spot": 2, # default vertical chute rows
            "active_preset": ""
        }

    def _load_default_bins(self) -> List[Dict[str, Any]]:
        # Default physical bins 0 to 5 mapped to 3 conveyor spots with Top & Bottom sides
        colors = ["#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22", "#1abc9c", "#e84393", "#00cec9"]
        default_bins = [
            {
                "id": 0,
                "name": "Bin 0: Standard Bricks",
                "color": colors[0],
                "spot": 0,
                "side": "top",
                "row": 0,
                "delay_ms": 400,
                "pulse_ms": 350,
                "assigned_parts": ["3001", "3003", "3004", "3005"],
                "count": 0
            },
            {
                "id": 1,
                "name": "Bin 1: Standard Plates",
                "color": colors[1],
                "spot": 0,
                "side": "bottom",
                "row": 0,
                "delay_ms": 400,
                "pulse_ms": 350,
                "assigned_parts": ["3020", "3022", "3023", "3710"],
                "count": 0
            },
            {
                "id": 2,
                "name": "Bin 2: Slopes & Curved",
                "color": colors[2],
                "spot": 1,
                "side": "top",
                "row": 0,
                "delay_ms": 750,
                "pulse_ms": 350,
                "assigned_parts": ["3039", "3040", "3298", "15068"],
                "count": 0
            },
            {
                "id": 3,
                "name": "Bin 3: Tiles & Smooth",
                "color": colors[3],
                "spot": 1,
                "side": "bottom",
                "row": 0,
                "delay_ms": 750,
                "pulse_ms": 350,
                "assigned_parts": ["3068b", "3069b", "3070b", "2412b"],
                "count": 0
            },
            {
                "id": 4,
                "name": "Bin 4: Technic Components",
                "color": colors[4],
                "spot": 2,
                "side": "top",
                "row": 0,
                "delay_ms": 1100,
                "pulse_ms": 350,
                "assigned_parts": ["2780", "3705", "3706", "3673", "6558"],
                "count": 0
            },
            {
                "id": 5,
                "name": "Bin 5: Minifig & Specialty",
                "color": colors[5],
                "spot": 2,
                "side": "bottom",
                "row": 0,
                "delay_ms": 1100,
                "pulse_ms": 350,
                "assigned_parts": ["3626", "3626b", "3626c", "973"],
                "count": 0
            },
            {
                "id": -1,
                "name": "Bin -1: End-of-Belt / Reject",
                "color": "#e74c3c",
                "spot": -1,
                "side": "end",
                "row": 0,
                "delay_ms": 0,
                "pulse_ms": 0,
                "assigned_parts": [], # Auto-catches everything unassigned or low confidence
                "count": 0
            }
        ]
        return default_bins

    def _load_catalog(self) -> List[Dict[str, Any]]:
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load catalog: {e}")
        return []

    def load_settings(self) -> Dict[str, Any]:
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.settings.update(saved)
            except Exception as e:
                logger.error(f"Failed to read settings: {e}")
        return self.settings

    def save_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        self.settings.update(new_settings)
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            logger.info("Saved settings to disk.")
        except Exception as e:
            logger.error(f"Failed to write settings: {e}")
        return self.settings

    def load_current_bins(self) -> List[Dict[str, Any]]:
        if self.bins_file.exists():
            try:
                with open(self.bins_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list) and len(loaded) > 0:
                        for idx, b in enumerate(loaded):
                            if "spot" not in b:
                                b["spot"] = idx // 2 if b["id"] != -1 else -1
                            if "side" not in b:
                                b["side"] = "top" if (idx % 2 == 0) else "bottom"
                                if b["id"] == -1: b["side"] = "end"
                            if "row" not in b:
                                b["row"] = 0
                            if "delay_ms" not in b:
                                b["delay_ms"] = (b["spot"] + 1) * 350 if b["id"] != -1 else 0
                            if "pulse_ms" not in b:
                                b["pulse_ms"] = 350 if b["id"] != -1 else 0
                        self.bins = loaded
            except Exception as e:
                logger.error(f"Failed to read bins file: {e}")

        # Ensure Bin -1 is always present at the end
        if not any(b["id"] == -1 for b in self.bins):
            self.bins.append({
                "id": -1,
                "name": "Bin -1: End-of-Belt / Reject",
                "color": "#e74c3c",
                "spot": -1,
                "side": "end",
                "row": 0,
                "delay_ms": 0,
                "pulse_ms": 0,
                "assigned_parts": [],
                "count": 0
            })

        # Update num_conveyor_spots based on existing bins
        max_spot = max([b.get("spot", 0) for b in self.bins if b["id"] >= 0] + [2])
        if self.settings.get("num_conveyor_spots", 3) < max_spot + 1:
            self.settings["num_conveyor_spots"] = max_spot + 1
            self.save_settings(self.settings)

        return self.bins

    def save_current_bins(self) -> None:
        try:
            with open(self.bins_file, "w", encoding="utf-8") as f:
                json.dump(self.bins, f, indent=2)
            logger.info(f"Persisted {len(self.bins)} bins to {self.bins_file}")
        except Exception as e:
            logger.error(f"Failed to save current bins: {e}")

    # ----------------------------------------------------
    # Conveyor Spots Management
    # ----------------------------------------------------
    def add_conveyor_spot(self) -> int:
        cur_spots = self.settings.get("num_conveyor_spots", 3)
        cur_spots += 1
        self.settings["num_conveyor_spots"] = cur_spots
        self.save_settings(self.settings)
        logger.info(f"Conveyor spots increased to {cur_spots}.")
        return cur_spots

    def remove_conveyor_spot(self) -> int:
        cur_spots = self.settings.get("num_conveyor_spots", 3)
        if cur_spots <= 1:
            return 1
        target_removed_spot = cur_spots - 1
        cur_spots -= 1
        self.settings["num_conveyor_spots"] = cur_spots
        self.save_settings(self.settings)

        # Move any bins in the removed spot to the last available spot
        for b in self.bins:
            if b["id"] >= 0 and b.get("spot", 0) >= cur_spots:
                b["spot"] = cur_spots - 1
        self.save_current_bins()
        logger.info(f"Conveyor spots decreased to {cur_spots}.")
        return cur_spots

    # ----------------------------------------------------
    # Bin Management (Add, Delete, Clear, Remove All)
    # ----------------------------------------------------
    def add_bin(
        self,
        name: Optional[str] = None,
        spot: Optional[int] = None,
        side: Optional[str] = None,
        row: Optional[int] = None,
        delay_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        palette = ["#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22", "#1abc9c", "#e84393", "#00cec9", "#fd79a8", "#6c5ce7"]
        
        existing_ids = [b["id"] for b in self.bins if b["id"] >= 0]
        new_id = (max(existing_ids) + 1) if existing_ids else 0

        max_spots = self.settings.get("num_conveyor_spots", 3)

        if spot is None or side is None:
            # Find first spot and side with lowest occupancy
            found_spot = 0
            found_side = "top"
            found_row = 0

            # Scan spots 0..max_spots-1
            spot_counts = {}
            for s in range(max_spots):
                for sd in ["top", "bottom"]:
                    bins_here = [b for b in self.bins if b["id"] >= 0 and b.get("spot", 0) == s and b.get("side", "top") == sd]
                    spot_counts[(s, sd)] = len(bins_here)

            best_loc = min(spot_counts.items(), key=lambda x: x[1])[0]
            found_spot, found_side = best_loc
            found_row = spot_counts[best_loc]

            spot = spot if spot is not None else found_spot
            side = side if side is not None else found_side
            row = row if row is not None else found_row

        if row is None:
            bins_here = [b for b in self.bins if b["id"] >= 0 and b.get("spot", 0) == spot and b.get("side", "top") == side]
            row = len(bins_here)

        if delay_ms is None:
            delay_ms = (spot + 1) * 350

        color = palette[new_id % len(palette)]
        bin_name = name or f"Bin {new_id}: Custom Bin"

        new_bin = {
            "id": new_id,
            "name": bin_name,
            "color": color,
            "spot": spot,
            "side": side,
            "row": row,
            "delay_ms": delay_ms,
            "pulse_ms": self.settings.get("gate_pulse_ms", 350),
            "assigned_parts": [],
            "count": 0
        }

        # Insert before Bin -1
        reject_bin = None
        for i, b in enumerate(self.bins):
            if b["id"] == -1:
                reject_bin = self.bins.pop(i)
                break

        self.bins.append(new_bin)
        if reject_bin:
            self.bins.append(reject_bin)

        self.save_current_bins()
        logger.info(f"Added new Bin {new_id} at Spot {spot} ({side.upper()}), Row {row} with delay {delay_ms}ms.")
        return new_bin

    def delete_bin(self, bin_id: int) -> bool:
        if bin_id == -1:
            return False

        initial_len = len(self.bins)
        self.bins = [b for b in self.bins if b["id"] != bin_id]
        if len(self.bins) < initial_len:
            self.save_current_bins()
            logger.info(f"Deleted Bin {bin_id}.")
            return True
        return False

    def clear_bin(self, bin_id: int) -> bool:
        """Clears all assigned parts and resets count for a single bin."""
        for b in self.bins:
            if b["id"] == bin_id:
                b["assigned_parts"] = []
                b["count"] = 0
                self.save_current_bins()
                logger.info(f"Cleared all parts from Bin {bin_id}.")
                return True
        return False

    def clear_all_bins(self) -> bool:
        """Clears all assigned parts and resets counts for ALL physical bins."""
        for b in self.bins:
            b["assigned_parts"] = []
            b["count"] = 0
        self.save_current_bins()
        logger.info("Cleared all assigned parts across ALL bins.")
        return True

    def remove_all_bins(self) -> bool:
        """Removes all physical bins (id >= 0), keeping only protected Bin -1."""
        reject_bin = next((b for b in self.bins if b["id"] == -1), None)
        if not reject_bin:
            reject_bin = {
                "id": -1,
                "name": "Bin -1: End-of-Belt / Reject",
                "color": "#e74c3c",
                "spot": -1,
                "side": "end",
                "row": 0,
                "delay_ms": 0,
                "pulse_ms": 0,
                "assigned_parts": [],
                "count": 0
            }
        reject_bin["count"] = 0
        self.bins = [reject_bin]
        self.save_current_bins()
        logger.info("Removed all physical bins.")
        return True

    def update_bin(self, bin_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for b in self.bins:
            if b["id"] == bin_id:
                for k in ["name", "color", "spot", "side", "row", "delay_ms", "pulse_ms", "assigned_parts"]:
                    if k in updates:
                        b[k] = updates[k]
                self.save_current_bins()
                return b
        return None

    def get_bin_dict(self, bin_id: int) -> Optional[Dict[str, Any]]:
        for b in self.bins:
            if b["id"] == bin_id:
                return b
        return None

    def get_bin_for_part(self, part_id: str, confidence: float) -> int:
        if confidence < self.settings.get("confidence_threshold", 0.35):
            return -1

        pid = part_id.lower().strip()
        for b in self.bins:
            if b["id"] == -1:
                continue
            assigned = [p.lower().strip() for p in b.get("assigned_parts", [])]
            if pid in assigned:
                return b["id"]

        return -1

    def increment_bin_count(self, bin_id: int) -> int:
        for b in self.bins:
            if b["id"] == bin_id:
                b["count"] = b.get("count", 0) + 1
                return b["count"]
        return 0

    def reset_bin_counts(self) -> None:
        for b in self.bins:
            b["count"] = 0
        self.save_current_bins()

    def get_assigned_parts_set(self) -> Set[str]:
        assigned = set()
        for b in self.bins:
            for p in b.get("assigned_parts", []):
                assigned.add(p.lower().strip())
        return assigned

    @staticmethod
    def _normalize_search_text(text: str) -> str:
        """Collapses whitespace and normalizes dimension separators (e.g. '2 x 4' -> '2x4')."""
        collapsed = " ".join(text.lower().split())
        return re.sub(r"\s*x\s*", "x", collapsed)

    def search_catalog_items(self, items: List[Dict[str, Any]], search: str) -> List[Dict[str, Any]]:
        """
        Filters and ranks catalog items against a search query, handling
        dimension spacing ('2x4' matching 'Brick  2 x  4') and multi-word queries.
        """
        q_norm = self._normalize_search_text(search)
        if not q_norm:
            return items
        q_tokens = q_norm.split()
        q_compact = q_norm.replace(" ", "")

        matched = []
        for x in items:
            pid = x.get("id", "").lower()
            name_norm = self._normalize_search_text(x.get("name", ""))
            haystack = f"{pid} {name_norm}"
            haystack_compact = haystack.replace(" ", "")

            if all(tok in haystack for tok in q_tokens) or q_compact in haystack_compact:
                # Rank exact ID match first, then exact/shorter name matches (e.g. "Brick 2 x 4" before variants)
                is_exact_id = (pid == q_compact)
                starts_with_tok = name_norm.startswith(q_tokens[0])
                matched.append((0 if is_exact_id else 1, 0 if starts_with_tok else 1, len(name_norm), pid, x))

        matched.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return [t[4] for t in matched]

    def get_unassigned_parts(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        limit: int = 120
    ) -> Dict[str, Any]:
        assigned_set = self.get_assigned_parts_set()
        unassigned = [item for item in self.catalog if item["id"].lower() not in assigned_set]

        if category and category.lower() != "all":
            unassigned = [x for x in unassigned if x.get("category", "").lower() == category.lower()]

        if search:
            unassigned = self.search_catalog_items(unassigned, search)

        total = len(unassigned)
        start = (page - 1) * limit
        end = start + limit
        paginated = unassigned[start:end]

        return {
            "total": total,
            "page": page,
            "pages": max(1, (total + limit - 1) // limit),
            "items": paginated
        }

    # ----------------------------------------------------
    # Presets Management (Save, Load, Delete, List)
    # ----------------------------------------------------
    def list_presets(self) -> List[str]:
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        presets = []
        for p in self.presets_dir.glob("*.json"):
            presets.append(p.stem)
        return sorted(presets)

    def save_preset(self, name: str) -> bool:
        clean_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
        if not clean_name:
            return False
        preset_file = self.presets_dir / f"{clean_name}.json"
        try:
            with open(preset_file, "w", encoding="utf-8") as f:
                json.dump({
                    "name": clean_name,
                    "num_conveyor_spots": self.settings.get("num_conveyor_spots", 3),
                    "bins": self.bins
                }, f, indent=2)
            self.settings["active_preset"] = clean_name
            self.save_settings(self.settings)
            logger.info(f"Saved preset '{clean_name}'.")
            return True
        except Exception as e:
            logger.error(f"Error saving preset {clean_name}: {e}")
            return False

    def load_preset(self, name: str) -> bool:
        preset_file = self.presets_dir / f"{name}.json"
        if not preset_file.exists():
            return False
        try:
            with open(preset_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_bins = data.get("bins", [])
            if loaded_bins:
                self.bins = loaded_bins
                self.save_current_bins()
            if "num_conveyor_spots" in data:
                self.settings["num_conveyor_spots"] = data["num_conveyor_spots"]
            self.settings["active_preset"] = name
            self.save_settings(self.settings)
            logger.info(f"Loaded preset '{name}'.")
            return True
        except Exception as e:
            logger.error(f"Error loading preset {name}: {e}")
            return False

    def delete_preset(self, name: str) -> bool:
        preset_file = self.presets_dir / f"{name}.json"
        if preset_file.exists():
            try:
                preset_file.unlink()
                if self.settings.get("active_preset") == name:
                    self.settings["active_preset"] = ""
                    self.save_settings(self.settings)
                logger.info(f"Deleted preset '{name}'.")
                return True
            except Exception as e:
                logger.error(f"Error deleting preset {name}: {e}")
        return False
