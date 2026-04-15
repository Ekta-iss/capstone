import numpy as np
import time

# =========================================================
# CLASS MAP (UPDATE IF NEEDED)
# =========================================================
CLASS_MAP = {
    0: "spreader",
    1: "container",
    2: "agv"
}

# =========================================================
# TEMPORAL MEMORY (VERY IMPORTANT)
# =========================================================
_prev_state = {
    "spreader_center": None,
    "time": None,
    "phase": "ALIGNMENT"
}

# =========================================================
# HELPERS
# =========================================================
def get_center(box):
    x1, y1, x2, y2 = box[:4]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def dist(p1, p2):
    return float(np.linalg.norm(p1 - p2))

def normalize_angle(angle):
    # FIX: always stable operator-friendly angle
    return (angle + 180) % 360 - 180

# =========================================================
# MAIN FUNCTION
# =========================================================
def run_fusion_model(detections, radar_data=None):
    """
    detections: [[x1,y1,x2,y2,cls], ...]
    radar_data (optional):
        {
            "distance_m": float,
            "velocity": float
        }
    """

    global _prev_state

    spreader = None
    container = None
    agv = None

    # =========================
    # CLASSIFICATION
    # =========================
    for d in detections:
        cls = CLASS_MAP.get(d[4], None)

        if cls == "spreader":
            spreader = d
        elif cls == "container":
            container = d
        elif cls == "agv":
            agv = d

    # =========================
    # DEFAULT OUTPUT
    # =========================
    output = {
        "distance_m": 0.0,
        "speed": 0.0,
        "angle_deg": 0.0,
        "direction": "STABLE",
        "phase": _prev_state["phase"],
        "confidence": 0.3
    }

    if spreader is None:
        return output

    # =========================
    # SPREADER POSITION
    # =========================
    sp_center = get_center(spreader)

    now = time.time()

    # =========================
    # SPEED (CV MOTION)
    # =========================
    if _prev_state["spreader_center"] is not None and _prev_state["time"] is not None:
        dt = now - _prev_state["time"]

        if dt > 0:
            pixel_speed = dist(sp_center, _prev_state["spreader_center"]) / dt
            output["speed"] = float(pixel_speed * 0.01)

    _prev_state["spreader_center"] = sp_center
    _prev_state["time"] = now

    # =========================================================
    # TARGET SELECTION (PRIORITY LOGIC)
    # =========================================================
    target = None
    phase = _prev_state["phase"]

    if container is not None:
        target = get_center(container)
        phase = "LOADING"

    elif agv is not None:
        target = get_center(agv)
        phase = "UNLOADING"

    else:
        phase = "ALIGNMENT"

    # =========================================================
    # DISTANCE (RADAR-FIRST FUSION)
    # =========================================================
    if radar_data is not None and "distance_m" in radar_data:
        # 🔥 RADAR has priority (REAL WORLD CORRECT)
        output["distance_m"] = float(radar_data["distance_m"])
    elif target is not None:
        # fallback CV
        output["distance_m"] = dist(sp_center, target) * 0.01

    # =========================================================
    # ANGLE (STABLE + NORMALIZED)
    # =========================================================
    if target is not None:
        dx = target[0] - sp_center[0]
        dy = target[1] - sp_center[1]

        angle = np.degrees(np.arctan2(dy, dx))
        output["angle_deg"] = normalize_angle(angle)

        # direction
        if dx > 5:
            output["direction"] = "RIGHT"
        elif dx < -5:
            output["direction"] = "LEFT"
        else:
            output["direction"] = "STABLE"

    # =========================================================
    # PHASE STABILITY (IMPORTANT FIX)
    # =========================================================
    # prevents flickering LOADING bug
    if phase == "LOADING" and _prev_state["phase"] == "UNLOADING":
        phase = "ALIGNMENT"

    _prev_state["phase"] = phase
    output["phase"] = phase

    # =========================================================
    # CONFIDENCE (REALISTIC)
    # =========================================================
    obj_count = len(detections)

    conf = np.clip(obj_count / 3, 0, 1)

    if spreader is not None and (container is not None or agv is not None):
        conf += 0.2

    output["confidence"] = float(np.clip(conf, 0, 1))

    return output