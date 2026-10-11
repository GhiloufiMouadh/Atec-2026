"""CRATER Sentinel: local lunar rover reconnaissance simulator (smooth version).

Run from VS Code or a terminal with:
    python robot_simulation.py

Adds smooth camera interpolation, faster realistic rover motion, and a
more legible forward navigation view. Standard library only.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    tk = None
    messagebox = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "simulation_data"
LOG_PATH = DATA_DIRECTORY / "detection_log.jsonl"
REPORT_PATH = DATA_DIRECTORY / "mission_report.json"

GRID_COLUMNS = 30
GRID_ROWS = 22
CELL_SIZE = 20
SCAN_RADIUS_CELLS = 4.2
METRES_PER_CELL = 2.5

# --- speed / smoothness knobs -------------------------------------------------
CAMERA_FPS = 60                 # how many times per second the camera redraws
ROVER_STEP_MS = 380             # time between two grid cells (was 950)
ROVER_STEP_INTERVAL_MS = 40     # how often we update the interpolated position
SMOOTHING = 0.18                # lerp factor for camera follow (0..1)
# ------------------------------------------------------------------------------

HAZARD_STYLE = {
    "rock_formation": {"label": "Rock formation", "colour": "#ffb454", "priority": "MEDIUM"},
    "fissure":        {"label": "Fissure",       "colour": "#ff697f", "priority": "HIGH"},
    "hole":           {"label": "Lunar hole",    "colour": "#b78cff", "priority": "HIGH"},
    "dark_spot":      {"label": "Dark spot",     "colour": "#a8bbc8", "priority": "LOW"},
}


@dataclass(frozen=True)
class SurfaceFeature:
    feature_id: str
    feature_type: str
    grid_x: int
    grid_y: int
    size_m: float


@dataclass
class DetectionRecord:
    sequence: int
    detected_at_utc: str
    feature_id: str
    feature_type: str
    priority: str
    grid_coordinate: dict[str, int]
    rover_grid_coordinate: dict[str, int]
    estimated_position_m: dict[str, float]
    confidence: float
    uncertainty_m: float
    rover_distance_m: float


def build_surface() -> list[SurfaceFeature]:
    return [
        SurfaceFeature("RK-01", "rock_formation", 6, 17, 3.2),
        SurfaceFeature("FS-01", "fissure", 10, 15, 6.8),
        SurfaceFeature("RK-02", "rock_formation", 16, 12, 4.1),
        SurfaceFeature("FS-02", "fissure", 20, 9, 8.4),
        SurfaceFeature("RK-03", "rock_formation", 25, 7, 3.6),
        SurfaceFeature("FS-03", "fissure", 27, 17, 7.1),
        SurfaceFeature("RK-04", "rock_formation", 14, 4, 4.8),
        SurfaceFeature("HL-01", "hole", 23, 11, 5.4),
        SurfaceFeature("DS-01", "dark_spot", 9, 10, 4.2),
    ]


def build_route() -> list[tuple[int, int]]:
    return [
        (3, 20), (5, 18), (8, 16), (11, 14), (14, 13), (17, 11),
        (20, 9), (23, 8), (26, 7), (28, 9), (28, 13), (27, 17),
        (24, 18), (20, 17), (17, 15), (14, 12), (12, 8), (14, 5),
    ]


def build_random_route(steps: int = 100) -> list[tuple[int, int]]:
    position = (GRID_COLUMNS // 2, GRID_ROWS // 2)
    route = [position]
    previous = None
    for _ in range(steps - 1):
        x, y = position
        candidates = [
            (x + dx, y + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1))
            if 0 <= x + dx < GRID_COLUMNS and 0 <= y + dy < GRID_ROWS
        ]
        forward_options = [c for c in candidates if c != previous]
        if forward_options:
            candidates = forward_options
        next_position = random.choice(candidates)
        previous, position = position, next_position
        route.append(position)
    return route


def distance_in_cells(first, second) -> float:
    return math.dist(first, second)


def estimate_detection(feature, rover_position, sequence) -> DetectionRecord | None:
    distance_cells = distance_in_cells(rover_position, (feature.grid_x, feature.grid_y))
    if distance_cells > SCAN_RADIUS_CELLS:
        return None
    distance_m = distance_cells * METRES_PER_CELL
    confidence = round(max(0.67, 0.98 - distance_cells * 0.062), 2)
    uncertainty = round(0.45 + distance_m * 0.045 + (1 - confidence) * 2.5, 2)
    return DetectionRecord(
        sequence=sequence,
        detected_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        feature_id=feature.feature_id,
        feature_type=feature.feature_type,
        priority=HAZARD_STYLE[feature.feature_type]["priority"],
        grid_coordinate={"x": feature.grid_x, "y": feature.grid_y},
        rover_grid_coordinate={"x": rover_position[0], "y": rover_position[1]},
        estimated_position_m={
            "x": round((feature.grid_x - rover_position[0]) * METRES_PER_CELL, 2),
            "y": round((rover_position[1] - feature.grid_y) * METRES_PER_CELL, 2),
        },
        confidence=confidence,
        uncertainty_m=uncertainty,
        rover_distance_m=round(distance_m, 2),
    )


def write_record(record: DetectionRecord) -> None:
    DATA_DIRECTORY.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(asdict(record)) + "\n")


def write_report(records, rover_position, rover_path=()) -> None:
    records_list = list(records)
    path_list = list(rover_path)
    DATA_DIRECTORY.mkdir(exist_ok=True)
    report = {
        "mission": "CRATER Sentinel - Lunar Surface Reconnaissance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rover_grid_position": {"x": rover_position[0], "y": rover_position[1]},
        "rover_path_grid_coordinates": [
            {"step": step, "x": x, "y": y}
            for step, (x, y) in enumerate(path_list, start=1)
        ],
        "rover_path_length_steps": len(path_list),
        "verified_detection_count": len(records_list),
        "fissure_count":        sum(r.feature_type == "fissure"        for r in records_list),
        "rock_formation_count": sum(r.feature_type == "rock_formation" for r in records_list),
        "hole_count":           sum(r.feature_type == "hole"           for r in records_list),
        "dark_spot_count":      sum(r.feature_type == "dark_spot"      for r in records_list),
        "records": [asdict(r) for r in records_list],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = DATA_DIRECTORY / "detection_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "sequence", "detected_at_utc", "feature_id", "feature_type", "priority",
            "feature_grid_x", "feature_grid_y", "estimated_x_m", "estimated_y_m",
            "confidence", "uncertainty_m", "rover_distance_m",
            "rover_grid_x_at_detection", "rover_grid_y_at_detection",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records_list:
            writer.writerow({
                "sequence": record.sequence,
                "detected_at_utc": record.detected_at_utc,
                "feature_id": record.feature_id,
                "feature_type": record.feature_type,
                "priority": record.priority,
                "feature_grid_x": record.grid_coordinate["x"],
                "feature_grid_y": record.grid_coordinate["y"],
                "estimated_x_m": record.estimated_position_m["x"],
                "estimated_y_m": record.estimated_position_m["y"],
                "confidence": record.confidence,
                "uncertainty_m": record.uncertainty_m,
                "rover_distance_m": record.rover_distance_m,
                "rover_grid_x_at_detection": record.rover_grid_coordinate["x"],
                "rover_grid_y_at_detection": record.rover_grid_coordinate["y"],
            })


def run_headless_mission() -> list[DetectionRecord]:
    detected_feature_ids: set[str] = set()
    records: list[DetectionRecord] = []
    for rover_position in build_route():
        for feature in build_surface():
            if feature.feature_id in detected_feature_ids:
                continue
            record = estimate_detection(feature, rover_position, len(records) + 1)
            if record is not None:
                records.append(record)
                detected_feature_ids.add(feature.feature_id)
    return records


# ---------------------------------------------------------------------------
# Smooth camera helpers
# ---------------------------------------------------------------------------
def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation with clamping."""
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def blend_colour(c1: str, c2: str, t: float) -> str:
    """Blend two hex colours (#rrggbb)."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class LunarRoverSimulation:
    """Tkinter Mission Control with smooth camera and realistic motion."""

    def __init__(self) -> None:
        if tk is None:
            raise RuntimeError("Tkinter is unavailable. Run 'python robot_simulation.py --headless' instead.")
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            raise RuntimeError(
                "The selected Python interpreter cannot open Tkinter. In VS Code, choose a standard "
                "Python installation with Tcl/Tk support, then run this file again. "
                "You can still verify the mission logic with: python robot_simulation.py --headless"
            ) from error

        self.root.title("CRATER Sentinel | Lunar Rover Reconnaissance")
        self.root.configure(background="#07111f")
        self.root.resizable(False, False)

        self.features = build_surface()
        self.route = build_random_route(steps=100)
        self.route_index = 0
        self.rover_path: list[tuple[int, int]] = [self.route[0]]
        self.detected_feature_ids: set[str] = set()
        self.records: list[DetectionRecord] = []
        self.running = False

        # --- smooth motion state -----------------------------------------
        self.segment_start_time = time.time()
        self.segment_duration = ROVER_STEP_MS / 1000.0
        self.smooth_pos = (float(self.route[0][0]), float(self.route[0][1]))
        self.camera_pos = self.smooth_pos  # lerp-follow
        self.roll = 0.0                     # bouncing
        self.pitch = 0.0
        self.jitter = 0.0                   # camera shake
        # -----------------------------------------------------------------

        self.tick_after_id: str | None = None
        self.sensor_values: dict[str, tk.Label] = {}
        self._build_interface()
        self._draw_scene()
        self._schedule_camera_loop()

    # ---- position helpers -------------------------------------------------
    @property
    def rover_position(self) -> tuple[int, int]:
        return self.route[self.route_index]

    def _update_smooth_motion(self) -> None:
        """Advance the interpolated rover position between two grid cells."""
        now = time.time()
        if not self.running:
            self.smooth_pos = (float(self.rover_position[0]), float(self.rover_position[1]))
        else:
            elapsed = now - self.segment_start_time
            t = elapsed / self.segment_duration
            if t >= 1.0:
                # Advance to next waypoint
                if self.route_index < len(self.route) - 1:
                    self.route_index += 1
                    self.rover_path.append(self.rover_position)
                    self.segment_start_time = now
                    self.segment_duration = ROVER_STEP_MS / 1000.0
                    t = 0.0
                else:
                    self.running = False
                    self._on_mission_complete()
                    return

            a = self.route[max(0, self.route_index - 1)]
            b = self.route[self.route_index]
            self.smooth_pos = (
                lerp(a[0], b[0], t),
                lerp(a[1], b[1], t),
            )

            # Realistic suspension: vertical bounce + slight roll/pitch
            bounce = math.sin(t * math.pi) * 0.18
            self.roll = math.sin(t * math.pi * 2 + self.route_index * 0.7) * 2.2
            self.pitch = math.cos(t * math.pi * 2 + self.route_index * 0.55) * 1.4
            self.jitter = bounce

        # Camera follows smoothly with a delay (realistic head-lag)
        self.camera_pos = (
            lerp(self.camera_pos[0], self.smooth_pos[0], SMOOTHING),
            lerp(self.camera_pos[1], self.smooth_pos[1], SMOOTHING),
        )

    # ---- camera loop (60 FPS) ---------------------------------------------
    def _schedule_camera_loop(self) -> None:
        interval = int(1000 / CAMERA_FPS)
        self.root.after(interval, self._camera_loop)

    def _camera_loop(self) -> None:
        self._update_smooth_motion()
        self._draw_camera_feed()
        self._schedule_camera_loop()

    # ---- UI construction (unchanged layout) --------------------------------
    def _build_interface(self) -> None:
        header = tk.Frame(self.root, bg="#0a1a2a", padx=18, pady=12)
        header.pack(fill="x")
        title = tk.Frame(header, bg="#0a1a2a")
        title.pack(side="left")
        tk.Label(title, text="PROJECT C.R.A.T.E.R  /  EDGE RECONNAISSANCE", fg="#63d9ff", bg="#0a1a2a", font=("Arial", 9, "bold")).pack(anchor="w")
        tk.Label(title, text="CRATER Sentinel", fg="#edf5ff", bg="#0a1a2a", font=("Arial", 21, "bold")).pack(anchor="w", pady=(1, 0))
        tk.Label(title, text="L-SCOUT 01 · LOCAL PERCEPTION · SMOOTH CAMERA", fg="#9cb0c5", bg="#0a1a2a", font=("Arial", 8, "bold")).pack(anchor="w", pady=(2, 0))
        mission_badge = tk.Frame(header, bg="#112e38", padx=12, pady=8)
        mission_badge.pack(side="right", anchor="ne")
        tk.Label(mission_badge, text="●  SIMULATION READY", fg="#85e4b1", bg="#112e38", font=("Arial", 9, "bold")).pack()
        tk.Label(mission_badge, text="LUNAR SURFACE / SECTOR A-17", fg="#9cb0c5", bg="#112e38", font=("Consolas", 8)).pack(pady=(2, 0))

        content = tk.Frame(self.root, bg="#07111f", padx=15, pady=15)
        content.pack(fill="both", expand=True)
        mission_area = tk.Frame(content, bg="#07111f")
        mission_area.pack(side="left", fill="both", expand=True)

        camera_panel = tk.Frame(mission_area, bg="#0b1c2e", highlightthickness=1, highlightbackground="#36516c")
        camera_panel.pack(fill="x")
        self._panel_title(camera_panel, "NAVCAM-01", "FORWARD NAVIGATION CAMERA · 60 FPS SMOOTH")
        self.camera_canvas = tk.Canvas(camera_panel, width=GRID_COLUMNS * CELL_SIZE, height=200, bg="#08111c", highlightthickness=0)
        self.camera_canvas.pack(padx=1, pady=(0, 1))

        map_panel = tk.Frame(mission_area, bg="#0b1c2e", highlightthickness=1, highlightbackground="#36516c")
        map_panel.pack(fill="both", expand=True, pady=(12, 0))
        self._panel_title(map_panel, "LIDAR / TERRAIN MAP", "360° RANGE SCAN · GRID RESOLUTION 2.5 m")
        self.canvas = tk.Canvas(map_panel, width=GRID_COLUMNS * CELL_SIZE, height=GRID_ROWS * CELL_SIZE, bg="#1c252d", highlightthickness=0)
        self.canvas.pack(padx=1, pady=(0, 1))

        side = tk.Frame(content, bg="#0b1c2e", width=350, padx=15, pady=14, highlightthickness=1, highlightbackground="#36516c")
        side.pack(side="right", fill="both", padx=(14, 0))
        side.pack_propagate(False)

        self.status_label = tk.Label(side, text="READY FOR LOCAL SCAN", fg="#85e4b1", bg="#0b1c2e", font=("Arial", 10, "bold"))
        self.status_label.pack(anchor="w")
        self.position_label = tk.Label(side, text="Rover grid: --", fg="#cbd8e7", bg="#0b1c2e", font=("Consolas", 10))
        self.position_label.pack(anchor="w", pady=(6, 0))
        self.metrics_label = tk.Label(side, text="Verified detections: 0\nFissures: 0 · Rocks: 0", fg="#9cb0c5", bg="#0b1c2e", justify="left", font=("Arial", 10))
        self.metrics_label.pack(anchor="w", pady=(6, 10))

        tk.Label(side, text="SENSORS & ROVER HEALTH", fg="#63d9ff", bg="#0b1c2e", font=("Arial", 9, "bold")).pack(anchor="w")
        sensor_grid = tk.Frame(side, bg="#0b1c2e")
        sensor_grid.pack(fill="x", pady=(6, 11))
        for column in range(2):
            sensor_grid.grid_columnconfigure(column, weight=1)
        self._make_sensor_card(sensor_grid, "NAV CAMERA", 0, 0, "60.0 FPS · ACTIVE", "#63d9ff")
        self._make_sensor_card(sensor_grid, "HAZ CAMERA", 0, 1, "WIDE FOV · ACTIVE", "#63d9ff")
        self._make_sensor_card(sensor_grid, "LIDAR", 1, 0, "SCANNING", "#85e4b1")
        self._make_sensor_card(sensor_grid, "IMU", 1, 1, "HDG 000°", "#b78cff")
        self._make_sensor_card(sensor_grid, "THERMAL", 2, 0, "-183°C · CLEAR", "#ffb454")
        self._make_sensor_card(sensor_grid, "S-BAND LINK", 2, 1, "98% · NOMINAL", "#85e4b1")
        self._make_sensor_card(sensor_grid, "POWER", 3, 0, "88% · NOMINAL", "#85e4b1")
        self._make_sensor_card(sensor_grid, "DEPTH CAMERA", 3, 1, "0.25 m RES", "#b78cff")

        tk.Label(side, text="MISSION EVENT LOG", fg="#63d9ff", bg="#0b1c2e", font=("Arial", 9, "bold")).pack(anchor="w")
        self.log_box = tk.Text(side, height=9, width=36, bg="#07111f", fg="#d5e4f4", insertbackground="#d5e4f4", relief="flat", wrap="word", font=("Consolas", 9), state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(7, 11))

        buttons = tk.Frame(side, bg="#0b1c2e")
        buttons.pack(fill="x")
        self.start_button = tk.Button(buttons, text="Start mission", command=self.toggle_running, bg="#63d9ff", fg="#07111f", relief="flat", font=("Arial", 10, "bold"), padx=12, pady=8)
        self.start_button.pack(side="left", fill="x", expand=True)
        tk.Button(buttons, text="Reset", command=self.reset, bg="#17324c", fg="#edf5ff", relief="flat", font=("Arial", 10, "bold"), padx=12, pady=8).pack(side="left", padx=(8, 0))
        tk.Button(side, text="Export mission report", command=self.export_report, bg="#17324c", fg="#edf5ff", relief="flat", font=("Arial", 10, "bold"), pady=8).pack(fill="x", pady=(8, 0))

        legend = tk.Frame(side, bg="#0b1c2e", pady=10)
        legend.pack(fill="x")
        for feature_type in ("rock_formation", "fissure", "hole", "dark_spot"):
            style = HAZARD_STYLE[feature_type]
            tk.Label(legend, text=f"●  {style['label']}  /  {style['priority']}", fg=style["colour"], bg="#0b1c2e", font=("Arial", 8, "bold")).pack(anchor="w")

    @staticmethod
    def _panel_title(parent: tk.Frame, label: str, detail: str) -> None:
        title = tk.Frame(parent, bg="#0b1c2e", padx=11, pady=7)
        title.pack(fill="x")
        tk.Label(title, text=label, fg="#63d9ff", bg="#0b1c2e", font=("Arial", 9, "bold")).pack(side="left")
        tk.Label(title, text=detail, fg="#71869a", bg="#0b1c2e", font=("Consolas", 8)).pack(side="right")

    def _make_sensor_card(self, parent, name, row, column, initial_value, colour) -> None:
        card = tk.Frame(parent, bg="#10283a", padx=8, pady=6)
        card.grid(row=row, column=column, sticky="nsew", padx=2, pady=2)
        tk.Label(card, text=name, fg="#9cb0c5", bg="#10283a", font=("Arial", 7, "bold")).pack(anchor="w")
        value = tk.Label(card, text=initial_value, fg=colour, bg="#10283a", font=("Consolas", 8, "bold"))
        value.pack(anchor="w", pady=(2, 0))
        self.sensor_values[name] = value

    def _heading_degrees(self) -> int:
        if self.route_index == 0:
            return 0
        a = self.route[self.route_index - 1]
        b = self.route[self.route_index]
        return round((math.degrees(math.atan2(b[0] - a[0], a[1] - b[1])) + 360) % 360)

    def _nearby_features(self) -> list[tuple[SurfaceFeature, float]]:
        # Use the interpolated rover position for a smoothly updating view.
        pos = (int(round(self.smooth_pos[0])), int(round(self.smooth_pos[1])))
        return sorted(
            (
                (feature, distance_in_cells(pos, (feature.grid_x, feature.grid_y)))
                for feature in self.features
                if distance_in_cells(pos, (feature.grid_x, feature.grid_y)) <= SCAN_RADIUS_CELLS
            ),
            key=lambda item: item[1],
        )

    # ---- Camera rendering (improved clarity) ------------------------------
    def _draw_camera_feed(self) -> None:
        """Higher-clarity synthetic forward camera with smoothing and depth."""
        canvas = self.camera_canvas
        canvas.delete("all")
        width = int(canvas["width"])
        height = int(canvas["height"])
        horizon = 70

        # --- Sky with smooth vertical gradient (many bands for fluency) ---
        top_colour, bottom_colour = "#03060f", "#16283a"
        bands = 24
        for i in range(bands):
            t = i / (bands - 1)
            colour = blend_colour(top_colour, bottom_colour, t)
            y0 = i * horizon / bands
            y1 = (i + 1) * horizon / bands + 1
            canvas.create_rectangle(0, y0, width, y1, fill=colour, outline="")

        # --- Stars with soft twinkle --------------------------------------
        t_now = time.time()
        for star in range(46):
            x = (star * 71 + 17) % width
            y = 6 + (star * 29) % (horizon - 12)
            base = 1 if star % 9 else 2
            twinkle = 0.55 + 0.45 * math.sin(t_now * 2.0 + star)
            star_size = max(1, int(base * (0.6 + twinkle * 0.6)))
            canvas.create_oval(x, y, x + star_size, y + star_size,
                               fill="#e9f5ff" if twinkle > 0.85 else "#bcd6e8", outline="")

        # --- Earth crescent ----------------------------------------------
        earth_x, earth_y, earth_radius = width - 108, 30, 20
        canvas.create_oval(earth_x - earth_radius, earth_y - earth_radius,
                           earth_x + earth_radius, earth_y + earth_radius,
                           fill="#5f9cbe", outline="#c6ecff")
        canvas.create_oval(earth_x - earth_radius + 9, earth_y - earth_radius - 2,
                           earth_x + earth_radius + 9, earth_y + earth_radius - 2,
                           fill="#050b18", outline="")

        # --- Distant mountain ridges (parallax with rover motion) --------
        parallax = (self.camera_pos[0] - self.route[0][0]) * 1.2
        ridge_far = [
            (0, horizon + 6), (70, 42), (137, horizon - 6), (230, 50),
            (325, horizon + 2), (422, 36), (520, horizon - 3), (width, 42),
        ]
        ridge_near = [
            (0, horizon + 16), (106, 57), (188, horizon + 6), (274, 63),
            (386, horizon - 5), (486, 58), (width, horizon + 2),
        ]
        for i, (x, y) in enumerate(ridge_far):
            ridge_far[i] = (x - parallax * 0.25, y)
        for i, (x, y) in enumerate(ridge_near):
            ridge_near[i] = (x - parallax * 0.55, y)
        canvas.create_polygon(*[c for pt in ridge_far for c in pt],
                              width, horizon + 20, 0, horizon + 20,
                              fill="#1b2a36", outline="")
        canvas.create_polygon(*[c for pt in ridge_near for c in pt],
                              width, horizon + 26, 0, horizon + 26,
                              fill="#2d3d47", outline="")

        # --- Ground with smooth shaded bands ------------------------------
        ground_top, ground_bottom = "#6f7573", "#2b3538"
        ground_bands = 28
        for i in range(ground_bands):
            t = i / (ground_bands - 1)
            colour = blend_colour(ground_top, ground_bottom, t)
            y0 = horizon + i * (height - horizon) / ground_bands
            y1 = horizon + (i + 1) * (height - horizon) / ground_bands + 1
            canvas.create_rectangle(0, y0, width, y1, fill=colour, outline="")

        # --- Craters with shadow and rim highlight ------------------------
        for crater_x, crater_y, crater_radius in (
            (60, 118, 12), (170, 152, 18), (350, 108, 10),
            (470, 144, 20), (552, 98, 11), (280, 170, 15),
        ):
            canvas.create_oval(
                crater_x - crater_radius, crater_y - crater_radius * .42,
                crater_x + crater_radius, crater_y + crater_radius * .42,
                fill="#283336", outline="#7c847f",
            )
            canvas.create_arc(
                crater_x - crater_radius, crater_y - crater_radius * .42,
                crater_x + crater_radius, crater_y + crater_radius * .42,
                start=200, extent=160, style="arc", outline="#aab2ac", width=1,
            )
            canvas.create_arc(
                crater_x - crater_radius, crater_y - crater_radius * .42,
                crater_x + crater_radius, crater_y + crater_radius * .42,
                start=20, extent=160, style="arc", outline="#4a5456", width=1,
            )

        # --- Dust speckles (perspective-scaled) ---------------------------
        for dust in range(70):
            x = (dust * 83 + 13) % width
            depth = (dust * 47) % (height - horizon - 15)
            y = horizon + 7 + depth
            depth_t = depth / max(1, (height - horizon - 15))
            dust_size = max(1, int(1 + depth_t * 3))
            canvas.create_oval(x, y, x + dust_size, y + max(1, dust_size // 2),
                               fill=blend_colour("#565c59", "#8a918c", depth_t), outline="")

        # --- Perspective grid (smoother, three tiers) ---------------------
        for row in range(7):
            y = horizon + 18 + row * row * 5
            if y >= height: break
            canvas.create_line(0, y, width, y, fill="#7c8886")
        cx = width / 2 - parallax * 3
        for x in range(-300, width + 320, 70):
            canvas.create_line(cx, horizon, x - parallax * 3, height, fill="#6a7674")

        # --- Atmospheric haze along the horizon ---------------------------
        for i in range(6):
            t = i / 5
            y = horizon - 3 + i
            canvas.create_line(0, y, width, y,
                               fill=blend_colour("#0a1826", "#3b556a", 1 - t))

        # --- HUD frame -----------------------------------------------------
        canvas.create_line(width / 2 - 12, horizon, width / 2 + 12, horizon, fill="#63d9ff", width=2)
        canvas.create_line(width / 2, horizon - 12, width / 2, horizon + 12, fill="#63d9ff", width=2)
        for x, y, hx, vy in ((7, 7, 22, 0), (7, 7, 0, 18),
                              (width - 7, 7, -22, 0), (width - 7, 7, 0, 18),
                              (7, height - 7, 22, 0), (7, height - 7, 0, -18),
                              (width - 7, height - 7, -22, 0), (width - 7, height - 7, 0, -18)):
            canvas.create_line(x, y, x + hx, y + vy, fill="#63d9ff", width=2)

        canvas.create_rectangle(7, 7, 225, 32, fill="#07111f", outline="#28566f")
        canvas.create_text(13, 13, text="NAVCAM-01  |  EDGE CV ENABLED",
                           anchor="nw", fill="#d5eaf8", font=("Consolas", 8, "bold"))
        canvas.create_text(width - 12, 13,
                           text=f"FRAME {self.route_index + 1:04d}   REC ●",
                           anchor="ne", fill="#ff9bad", font=("Consolas", 8, "bold"))
        canvas.create_text(12, height - 14,
                           text="FOV 110°  ·  EXP 3.4 ms  ·  LOCAL ONLY",
                           anchor="sw", fill="#d8e9f1", font=("Consolas", 8, "bold"))
        canvas.create_text(width - 12, height - 14,
                           text=f"HDG {self._heading_degrees():03d}°  ·  R {self.roll:+.1f}°",
                           anchor="se", fill="#85e4b1", font=("Consolas", 8, "bold"))

        # --- Feature overlays (interpolated position, gentle bobbing) -----
        visible = self._nearby_features()
        bob = self.jitter * 18  # subtle vertical bob
        for feature, distance in visible:
            distance_ratio = distance / SCAN_RADIUS_CELLS
            offset_x = (feature.grid_x - self.camera_pos[0]) * 42
            object_x = max(34, min(width - 34, width / 2 + offset_x))
            object_y = horizon + 42 + (1 - distance_ratio) * 52 + bob
            scale = 10 + (1 - distance_ratio) * 14
            colour = HAZARD_STYLE[feature.feature_type]["colour"]
            if feature.feature_type == "rock_formation":
                canvas.create_oval(object_x - scale, object_y - scale * .55,
                                   object_x + scale, object_y + scale * .55,
                                   fill="#6f6252", outline="#c0a884", width=1)
                canvas.create_oval(object_x - scale * .6, object_y - scale * .8,
                                   object_x + scale * .3, object_y - scale * .1,
                                   fill="#8f8271", outline="")
            elif feature.feature_type == "fissure":
                canvas.create_line(object_x - scale, object_y - 4,
                                   object_x - 2, object_y + 3,
                                   object_x + 4, object_y - 4,
                                   object_x + scale, object_y + 5,
                                   fill="#0b1015", width=5, smooth=True)
            elif feature.feature_type == "hole":
                canvas.create_oval(object_x - scale, object_y - scale * .62,
                                   object_x + scale, object_y + scale * .62,
                                   fill="#040709", outline="#6a5590", width=1)
                canvas.create_oval(object_x - scale * .3, object_y - scale * .18,
                                   object_x + scale * .3, object_y + scale * .18,
                                   fill="#000000", outline="")
            else:
                canvas.create_oval(object_x - scale * 1.2, object_y - scale * .45,
                                   object_x + scale * 1.2, object_y + scale * .45,
                                   fill="#1d242a", outline="#9bacb7", width=1)
            canvas.create_rectangle(object_x - scale - 5, object_y - scale - 18,
                                    object_x + scale + 5, object_y + scale + 7,
                                    outline=colour, width=2)
            canvas.create_text(object_x - scale - 4, object_y - scale - 21,
                               text=f"{feature.feature_id}  {distance * METRES_PER_CELL:.1f}m",
                               anchor="sw", fill=colour, font=("Consolas", 8, "bold"))

    def _refresh_sensor_values(self) -> None:
        visible = self._nearby_features()
        nearest = visible[0] if visible else None
        heading = self._heading_degrees()
        power = max(72, 88 - self.route_index // 4)
        self.sensor_values["NAV CAMERA"].configure(text=f"60.0 FPS · {len(visible)} TARGET(S)", fg="#63d9ff")
        self.sensor_values["HAZ CAMERA"].configure(text=f"WIDE FOV · {len(visible)} FLAG(S)", fg="#63d9ff")
        if nearest:
            feature, distance = nearest
            self.sensor_values["LIDAR"].configure(text=f"{distance * METRES_PER_CELL:.1f} m · {feature.feature_id}",
                                                  fg=HAZARD_STYLE[feature.feature_type]["colour"])
        else:
            self.sensor_values["LIDAR"].configure(text="CLEAR TO 10.5 m", fg="#85e4b1")
        self.sensor_values["IMU"].configure(text=f"HDG {heading:03d}° · R {self.roll:+.1f}°", fg="#b78cff")
        nearby_dark_spot = any(f.feature_type == "dark_spot" for f, _ in visible)
        thermal_text = "ANOMALY FLAG" if nearby_dark_spot else "-183°C · CLEAR"
        self.sensor_values["THERMAL"].configure(text=thermal_text, fg="#ffb454" if nearby_dark_spot else "#85e4b1")
        link_quality = max(78, 98 - self.route_index)
        self.sensor_values["S-BAND LINK"].configure(text=f"{link_quality}% · NOMINAL", fg="#85e4b1")
        self.sensor_values["POWER"].configure(text=f"{power}% · NOMINAL", fg="#85e4b1")
        depth_quality = "0.25 m RES" if visible else "IDLE · 0.25 m"
        self.sensor_values["DEPTH CAMERA"].configure(text=depth_quality, fg="#b78cff")

    def _draw_scene(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_route()
        self._draw_features()
        self._draw_rover()
        self._draw_camera_feed()
        self._refresh_sidebar()
        self._refresh_sensor_values()

    def _draw_grid(self) -> None:
        width = GRID_COLUMNS * CELL_SIZE
        height = GRID_ROWS * CELL_SIZE
        terrain_bands = ("#101d27", "#13232e", "#172934", "#1a2c35", "#182933", "#14242f")
        band_height = height / len(terrain_bands)
        for index, colour in enumerate(terrain_bands):
            self.canvas.create_rectangle(0, index * band_height, width, (index + 1) * band_height, fill=colour, outline="")
        for crater_x, crater_y, radius in ((74, 74, 31), (221, 95, 22), (480, 69, 38), (535, 232, 26), (150, 348, 34), (387, 398, 21), (555, 422, 28)):
            self.canvas.create_oval(crater_x - radius, crater_y - radius * .62, crater_x + radius, crater_y + radius * .62, fill="#0d1821", outline="#31424c")
            self.canvas.create_arc(crater_x - radius, crater_y - radius * .62, crater_x + radius, crater_y + radius * .62, start=208, extent=150, style="arc", outline="#53636a", width=1)
        for column in range(GRID_COLUMNS + 1):
            x = column * CELL_SIZE
            colour = "#3a5666" if column % 5 == 0 else "#263e4c"
            self.canvas.create_line(x, 0, x, height, fill=colour)
        for row in range(GRID_ROWS + 1):
            y = row * CELL_SIZE
            colour = "#3a5666" if row % 5 == 0 else "#263e4c"
            self.canvas.create_line(0, y, width, y, fill=colour)
        for index in range(0, GRID_COLUMNS, 5):
            self.canvas.create_text(index * CELL_SIZE + 4, 9, text=f"{index:02d}", anchor="nw", fill="#85a7ba", font=("Consolas", 8, "bold"))
        for index in range(5, GRID_ROWS, 5):
            self.canvas.create_text(8, index * CELL_SIZE + 3, text=f"{index:02d}", anchor="nw", fill="#85a7ba", font=("Consolas", 8, "bold"))
        self.canvas.create_text(width - 12, 12, text="N ↑", anchor="ne", fill="#c6e6f2", font=("Consolas", 9, "bold"))
        self.canvas.create_text(width - 12, height - 10, text="LOCAL TERRAIN / 2.5m CELLS", anchor="se", fill="#7b98aa", font=("Consolas", 7))

    def _draw_route(self) -> None:
        coordinates = []
        for x, y in self.route:
            coordinates.extend((x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2))
        self.canvas.create_line(*coordinates, fill="#143f58", width=5, smooth=True)
        self.canvas.create_line(*coordinates, fill="#63d9ff", dash=(4, 5), width=2, smooth=True)

    def _draw_features(self) -> None:
        for feature in self.features:
            x = feature.grid_x * CELL_SIZE + CELL_SIZE / 2
            y = feature.grid_y * CELL_SIZE + CELL_SIZE / 2
            discovered = feature.feature_id in self.detected_feature_ids
            colour = HAZARD_STYLE[feature.feature_type]["colour"] if discovered else "#51565a"
            if feature.feature_type == "rock_formation":
                radius = 6 + feature.size_m
                self.canvas.create_oval(x - radius, y - radius * .65, x + radius, y + radius * .65,
                                        fill=colour, outline="#c89b69" if discovered else "#73777a")
            elif feature.feature_type == "fissure":
                self.canvas.create_line(x - 11, y - 6, x - 3, y + 1, x + 4, y - 4, x + 13, y + 6,
                                        fill=colour, width=5, smooth=True)
            elif feature.feature_type == "hole":
                radius = 5 + feature.size_m
                self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                        fill="#10161d", outline=colour, width=2)
                self.canvas.create_oval(x - radius * .55, y - radius * .55, x + radius * .55, y + radius * .55,
                                        fill="#020509", outline="")
            else:
                radius = 5 + feature.size_m
                self.canvas.create_oval(x - radius * 1.25, y - radius * .7, x + radius * 1.25, y + radius * .7,
                                        fill="#1a2026", outline=colour, width=2, dash=(2, 2))
            if discovered:
                self.canvas.create_text(x, y - 18, text=feature.feature_id, fill=colour, font=("Consolas", 8, "bold"))

    def _draw_rover(self) -> None:
        # Use the interpolated smooth position instead of the grid cell.
        cx = self.smooth_pos[0] * CELL_SIZE + CELL_SIZE / 2
        cy = self.smooth_pos[1] * CELL_SIZE + CELL_SIZE / 2
        radius = SCAN_RADIUS_CELLS * CELL_SIZE
        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                outline="#285a70", dash=(3, 5), width=1)
        self.canvas.create_oval(cx - radius * .54, cy - radius * .54,
                                cx + radius * .54, cy + radius * .54,
                                outline="#21495d", dash=(2, 7), width=1)
        scan_start = (self.route_index * 37) % 360
        self.canvas.create_arc(cx - radius, cy - radius, cx + radius, cy + radius,
                               start=scan_start, extent=65, style="arc", outline="#63d9ff", width=3)
        # Rover body with slight bounce
        bounce_y = self.jitter * 6
        self.canvas.create_oval(cx - 14, cy - 14 + bounce_y, cx + 14, cy + 14 + bounce_y,
                                fill="#153950", outline="#63d9ff", width=2)
        self.canvas.create_polygon(cx, cy - 12 + bounce_y,
                                   cx + 10, cy + 9 + bounce_y,
                                   cx, cy + 13 + bounce_y,
                                   cx - 10, cy + 9 + bounce_y,
                                   fill="#63d9ff", outline="#d7f5ff")
        self.canvas.create_line(cx, cy - 13 + bounce_y, cx, cy - 23 + bounce_y, fill="#d7f5ff", width=2)
        self.canvas.create_text(cx, cy + 29, text="L-SCOUT 01", fill="#d7f5ff", font=("Consolas", 8, "bold"))

    def _refresh_sidebar(self) -> None:
        x, y = self.rover_position
        self.position_label.configure(text=f"Rover grid: ({x:02d}, {y:02d}) · Scan: {SCAN_RADIUS_CELLS * METRES_PER_CELL:.1f} m")
        fissures = sum(r.feature_type == "fissure" for r in self.records)
        rocks = sum(r.feature_type == "rock_formation" for r in self.records)
        holes = sum(r.feature_type == "hole" for r in self.records)
        dark_spots = sum(r.feature_type == "dark_spot" for r in self.records)
        self.metrics_label.configure(
            text=(
                f"Verified detections: {len(self.records)}\n"
                f"Fissures: {fissures} · Rocks: {rocks}\n"
                f"Holes: {holes} · Dark spots: {dark_spots}"
            )
        )

    def _append_log(self, record: DetectionRecord) -> None:
        display_type = HAZARD_STYLE[record.feature_type]["label"]
        line = (
            f"[{record.sequence:02d}] {display_type.upper()} VERIFIED\n"
            f"  Grid ({record.grid_coordinate['x']}, {record.grid_coordinate['y']}) · "
            f"confidence {record.confidence:.0%}\n"
            f"  distance {record.rover_distance_m:.1f} m · ±{record.uncertainty_m:.1f} m\n\n"
        )
        self.log_box.configure(state="normal")
        self.log_box.insert("1.0", line)
        self.log_box.configure(state="disabled")

    def _scan_surface(self) -> None:
        pos = (int(round(self.smooth_pos[0])), int(round(self.smooth_pos[1])))
        for feature in self.features:
            if feature.feature_id in self.detected_feature_ids:
                continue
            record = estimate_detection(feature, pos, len(self.records) + 1)
            if record is None:
                continue
            self.detected_feature_ids.add(feature.feature_id)
            self.records.append(record)
            write_record(record)
            self._append_log(record)
            self.status_label.configure(
                text=f"VERIFIED: {HAZARD_STYLE[record.feature_type]['label'].upper()}",
                fg=HAZARD_STYLE[record.feature_type]["colour"],
            )
        write_report(self.records, self.rover_position, self.rover_path)

    # ---- Main mission loop (map refresh) ----------------------------------
    def _tick(self) -> None:
        if not self.running:
            return
        self._scan_surface()
        self._draw_scene()
        self.tick_after_id = self.root.after(ROVER_STEP_INTERVAL_MS, self._tick)

    def _on_mission_complete(self) -> None:
        write_report(self.records, self.rover_position, self.rover_path)
        self.start_button.configure(text="Mission complete", state="disabled")
        self.status_label.configure(text="MISSION COMPLETE · REPORT SAVED", fg="#85e4b1")
        self._draw_scene()

    def toggle_running(self) -> None:
        if self.running:
            self.running = False
            if self.tick_after_id:
                self.root.after_cancel(self.tick_after_id)
            self.start_button.configure(text="Resume mission")
            self.status_label.configure(text="MISSION PAUSED", fg="#ffb454")
        else:
            self.running = True
            self.segment_start_time = time.time()
            self.start_button.configure(text="Pause mission")
            self.status_label.configure(text="LOCAL SURFACE SCAN ACTIVE", fg="#85e4b1")
            self._tick()

    def reset(self) -> None:
        self.running = False
        if self.tick_after_id:
            self.root.after_cancel(self.tick_after_id)
        self.route = build_random_route(steps=100)
        self.route_index = 0
        self.rover_path = [self.route[0]]
        self.smooth_pos = (float(self.route[0][0]), float(self.route[0][1]))
        self.camera_pos = self.smooth_pos
        self.segment_start_time = time.time()
        self.detected_feature_ids.clear()
        self.records.clear()
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.start_button.configure(text="Start mission", state="normal")
        self.status_label.configure(text="READY FOR LOCAL SCAN", fg="#85e4b1")
        self._draw_scene()

    def export_report(self) -> None:
        write_report(self.records, self.rover_position, self.rover_path)
        if messagebox:
            messagebox.showinfo("Mission report exported", f"Saved to:\n{REPORT_PATH}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    if "--headless" in sys.argv:
        records = run_headless_mission()
        types = {r.feature_type for r in records}
        assert len(records) == len(build_surface()), "The complete route should detect every synthetic feature."
        assert {"fissure", "rock_formation", "hole", "dark_spot"}.issubset(types), "The mission must record every required feature type."
        print(f"Headless rover simulation passed: {len(records)} verified features recorded in memory.")
        return 0
    LunarRoverSimulation().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())