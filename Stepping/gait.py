#!/usr/bin/env python3
"""
gait.py — Bezier gait coordinator + IK + built-in debug window

Listens on  /robot/cmd          (string commands from sender.py)
Publishes on /gait/left/angles  (4 doubles: hip_roll hip_pitch knee_pitch ankle_pitch, radians)
             /gait/right/angles (4 doubles: same for right leg)
             /gait/left/foot    (3 doubles: x y z  in metres, hip-relative — kept for visualizer)
             /gait/right/foot   (3 doubles: same for right)
             /gait/viz          (full state bottle for visualizer)

"""

import time, sys, math, threading, os
import tkinter as tk
import numpy as np
import yarp

# ─── IK solver (optional — graceful fallback if ikpy not installed) ───────────
try:
    from ik_solver import LegIK
    _IK_AVAILABLE = True
except ImportError as _e:
    print(f"[gait] WARNING: IK unavailable ({_e}). Joint angle ports will not be published.")
    _IK_AVAILABLE = False

# ─── Try to load PyYAML — fall back to built-in parser if missing ─────────────
try:
    import yaml as _yaml
    def _load_yaml(path):
        with open(path) as f:
            return _yaml.safe_load(f)
except ImportError:
    # Minimal fallback: parses only the flat structure we need
    def _load_yaml(path):
        raise RuntimeError(
            "PyYAML not installed. Run:  pip install pyyaml\n"
            "Or the gait will use the built-in default curve."
        )

# ─── Robot geometry — extracted from humanoid_beta.urdf ──────────────────────
# To update: measure joint-to-joint distances from your URDF and edit here.
HIP_WIDTH    = 0.0495  # half the hip-to-hip lateral distance (m)
THIGH_LEN    = 0.1189  # hip joint → knee joint (m)
SHANK_LEN    = 0.1027  # knee joint → ankle joint (m)
ANKLE_HEIGHT = 0.0781  # ankle joint → floor contact point (m)
STEP_LEN     = 0.08    # forward distance per step (m) — tuned for robot size
STEP_HEIGHT  = 0.035   # peak foot lift height (m)
STAND_HEIGHT = -(THIGH_LEN + SHANK_LEN + ANKLE_HEIGHT) * 0.85  # comfortable standing z (~-0.255m)

# ─── Timing ───────────────────────────────────────────────────────────────────
RATE_HZ      = 50
DT           = 1.0 / RATE_HZ
SWING_TIME   = 0.40
STANCE_DWELL = 0.05
TURN_SPEED      = math.radians(30)
TURN_STEP_ANGLE = math.radians(20)
SLOW_FACTOR  = 0.55

# ─── Ports ────────────────────────────────────────────────────────────────────
PORT_CMD     = "/robot/cmd"
PORT_CMD_SRC = "/debug/sender"
PORT_LEFT    = "/gait/left/foot"
PORT_RIGHT   = "/gait/right/foot"
PORT_LEFT_ANGLES  = "/gait/left/angles"
PORT_RIGHT_ANGLES = "/gait/right/angles"
PORT_VIZ          = "/gait/viz"

# Joint order packed into YARP bottles — matches humanoid_beta.urdf joint names
# Right leg: 6 joints (Revolute_6..1)
# Left leg:  5 joints (Revolute_7..11)
RIGHT_JOINT_ORDER = ["Revolute_6", "Revolute_5", "Revolute_4",
                     "Revolute_3", "Revolute_2", "Revolute_1"]
LEFT_JOINT_ORDER  = ["Revolute_7", "Revolute_8", "Revolute_9",
                     "Revolute_10", "Revolute_11"]

# ─── Debug window colours ─────────────────────────────────────────────────────
CMD_COLORS = {
    "stop":     ("#444444", "white", "STOP"),
    "forward":  ("#27ae60", "white", "FORWARD  ▲"),
    "backward": ("#2980b9", "white", "BACKWARD ▼"),
    "left":     ("#f1c40f", "black", "LEFT  ◀"),
    "right":    ("#e67e22", "white", "RIGHT ▶"),
}
CMD_DEFAULT = ("#c0392b", "white", "???")

STATE_COLORS = {
    "IDLE":     "#555555",
    "STEPPING": "#27ae60",
    "STOPPING": "#e67e22",
    "TURNING":  "#8e44ad",
}

# ─── Bezier curve loader ───────────────────────────────────────────────────────

# Built-in fallback curve (same as original make_swing_curve shape)
_DEFAULT_CURVE_PROFILE = [
    [0.00, 0.00, 0.00],
    [0.15, 0.00, 0.20],
    [0.35, 0.00, 0.90],
    [0.50, 0.00, 1.00],
    [0.70, 0.00, 0.60],
    [1.00, 0.00, 0.00],
]

def load_swing_profile(yaml_path: str = None) -> list:
    """
    Load the active curve profile from swing_curves.yaml.
    Returns a list of 6 [x_frac, y_frac, z_frac] control points.
    Falls back to the default profile on any error.
    """
    if yaml_path is None:
        # os.path.dirname(__file__) returns "" when run as "python3 gait.py"
        # abspath fixes that so the yaml is always found next to gait.py
        here      = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(here, "swing_curves.yaml")
        print(f"[gait] looking for swing_curves.yaml at: {yaml_path}")

    try:
        data         = _load_yaml(yaml_path)
        active_name  = data.get("active_curve", "normal")
        curves       = data.get("curves", {})
        if active_name not in curves:
            print(f"[gait] WARNING: curve '{active_name}' not found in {yaml_path}. "
                  f"Using built-in default.")
            return _DEFAULT_CURVE_PROFILE
        profile = curves[active_name]["points"]
        print(f"[gait] Loaded swing curve '{active_name}' from {yaml_path}")
        return profile
    except FileNotFoundError:
        print(f"[gait] WARNING: {yaml_path} not found. Using built-in default curve.")
        return _DEFAULT_CURVE_PROFILE
    except Exception as e:
        print(f"[gait] WARNING: Could not load swing curves ({e}). Using built-in default.")
        return _DEFAULT_CURVE_PROFILE


def make_swing_curve(start: np.ndarray, end: np.ndarray,
                     height: float,
                     profile: list = None) -> np.ndarray:
    """
    Build a Bezier control-point array from start to end, shaped by profile.

    profile: list of 6 [x_frac, y_frac, z_frac] where:
      x_frac  — how far along the horizontal (XY) path  (0 = start, 1 = end)
      y_frac  — lateral offset as a fraction of step length (usually 0)
      z_frac  — height as a fraction of the STEP_HEIGHT constant

    The bug that was here: adding xf*vec (a 3D vector) mixed the z-difference
    between start and end into the control point positions AND double-counted z.
    Fix: separate horizontal travel (xy only) from vertical (z from profile).
    """
    if profile is None:
        profile = _DEFAULT_CURVE_PROFILE

    s = start.copy()
    e = end.copy()

    # Horizontal displacement only (ignore z — that's driven by the profile)
    horiz = np.array([e[0] - s[0], e[1] - s[1], 0.0])

    # Perpendicular to travel in XY plane (for y_frac lateral offset)
    horiz_len = np.linalg.norm(horiz[:2])
    if horiz_len > 1e-6:
        perp = np.array([-horiz[1], horiz[0], 0.0]) / horiz_len
    else:
        perp = np.array([0.0, 1.0, 0.0])

    pts = []
    for (xf, yf, zf) in profile:
        pt = np.array([
            s[0] + xf * horiz[0] + yf * perp[0],   # forward + lateral
            s[1] + xf * horiz[1] + yf * perp[1],
            s[2] + (e[2] - s[2]) * xf + zf * height # z: lerp start→end + arc
        ])
        pts.append(pt)

    pts[0]  = s.copy()   # anchor to exact start
    pts[-1] = e.copy()   # anchor to exact end
    return np.array(pts, dtype=float)


# ─── Bezier evaluator ─────────────────────────────────────────────────────────

def bezier(pts: np.ndarray, t: float) -> np.ndarray:
    p = pts.copy()
    n = len(p)
    for r in range(1, n):
        p[:n-r] = (1-t)*p[:n-r] + t*p[1:n-r+1]
    return p[0]

# ─── Gait state machine ───────────────────────────────────────────────────────

class Side:
    LEFT  = 0
    RIGHT = 1
    OTHER = {0: 1, 1: 0}

class GaitState:
    IDLE     = "IDLE"
    STEPPING = "STEPPING"
    STOPPING = "STOPPING"
    TURNING  = "TURNING"

class GaitCoordinator:
    def __init__(self, swing_profile: list = None):
        self.swing_profile  = swing_profile if swing_profile is not None else _DEFAULT_CURVE_PROFILE

        self.heading        = 0.0
        self.target_heading = 0.0
        self.body_pos       = np.array([0.0, 0.0])
        self.foot_world     = [
            np.array([-HIP_WIDTH, 0.0, STAND_HEIGHT]),
            np.array([ HIP_WIDTH, 0.0, STAND_HEIGHT]),
        ]
        self.state        = GaitState.IDLE
        self.swing_side   = Side.LEFT
        self.swing_t      = 0.0
        self.dwell_t      = 0.0
        self.swing_curve  = None
        self.swing_end    = None
        self.speed_scale  = 1.0

        # Turn queue — each tap of a/d adds 1 to the queue
        self.turn_queue   = []     # list of "left" or "right" strings
        self.turn_dir         = 0
        self.turn_resume      = GaitState.IDLE
        self.turn_pivot_side  = Side.LEFT
        self.turn_heading_end = 0.0

        self.cmd_lock  = threading.Lock()
        self.cmd       = "stop"    # current hold command (forward/backward/stop)

    def set_command(self, c: str):
        """Called by the listener thread on every incoming YARP message."""
        with self.cmd_lock:
            if c in ("left", "right"):
                # Each tap adds one entry to the queue — supports rapid tapping
                self.turn_queue.append(c)
                print(f"[gait] turn queued: {c}  (queue depth={len(self.turn_queue)})")
            else:
                # forward / backward / stop — these are hold commands
                self.cmd = c

    def _get_state(self):
        with self.cmd_lock:
            return self.cmd, list(self.turn_queue)

    def _pop_turn(self):
        with self.cmd_lock:
            if self.turn_queue:
                return self.turn_queue.pop(0)
            return None

    def _hip_world(self, side: int) -> np.ndarray:
        sign   = -1 if side == Side.LEFT else 1
        perp   = np.array([-math.sin(self.heading), math.cos(self.heading)])
        hip_xy = self.body_pos + sign * HIP_WIDTH * perp
        return np.array([hip_xy[0], hip_xy[1], 0.0])

    def _next_foot_pos(self, side: int) -> np.ndarray:
        fwd  = np.array([math.cos(self.heading), math.sin(self.heading)])
        perp = np.array([-math.sin(self.heading), math.cos(self.heading)])
        sign = -1 if side == Side.LEFT else 1
        xy   = self.body_pos + fwd * STEP_LEN * self.speed_scale + sign * HIP_WIDTH * perp
        return np.array([xy[0], xy[1], STAND_HEIGHT])

    def _turn_foot_pos(self, swing_side: int, new_heading: float) -> np.ndarray:
        pivot = self.foot_world[Side.OTHER[swing_side]].copy()
        sign  = -1 if swing_side == Side.LEFT else 1
        perp  = np.array([-math.sin(new_heading), math.cos(new_heading)])
        xy    = pivot[:2] + sign * 2 * HIP_WIDTH * perp
        return np.array([xy[0], xy[1], STAND_HEIGHT])

    def _advance_body(self):
        fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
        self.body_pos = self.body_pos + fwd * STEP_LEN * self.speed_scale * 0.5

    def _start_swing(self, side: int):
        self.swing_side  = side
        self.swing_t     = 0.0
        start            = self.foot_world[side].copy()
        end              = self._next_foot_pos(side)
        self.swing_end   = end
        self.swing_curve = make_swing_curve(start, end, STEP_HEIGHT,
                                             self.swing_profile)

    def _begin_turn(self, direction: str, resume_state: str):
        """Kick off one pivot step in direction, then return to resume_state."""
        self.turn_dir    = +1 if direction == "left" else -1
        self.turn_resume = resume_state

        self.turn_pivot_side = Side.LEFT  if direction == "left" else Side.RIGHT
        swing_side           = Side.RIGHT if direction == "left" else Side.LEFT

        self.turn_heading_end = self.heading + self.turn_dir * TURN_STEP_ANGLE

        start = self.foot_world[swing_side].copy()
        end   = self._turn_foot_pos(swing_side, self.turn_heading_end)

        self.swing_side  = swing_side
        self.swing_t     = 0.0
        self.swing_end   = end
        self.swing_curve = make_swing_curve(start, end, STEP_HEIGHT,
                                             self.swing_profile)
        self.state       = GaitState.TURNING

    def tick(self) -> dict:
        cmd, turn_queue = self._get_state()

        # ── Check if we should start a turn (only from non-TURNING states) ───
        if turn_queue and self.state != GaitState.TURNING:
            direction = self._pop_turn()
            # resume walking after turn if w is held, else go idle
            resume = GaitState.STEPPING if cmd == "forward" else GaitState.IDLE
            self._begin_turn(direction, resume_state=resume)

        # ── State machine ────────────────────────────────────────────────────
        if self.state == GaitState.IDLE:
            if cmd == "forward":
                self._start_swing(Side.LEFT)
                self.state = GaitState.STEPPING

        elif self.state == GaitState.STEPPING:
            if cmd == "stop":
                self.state = GaitState.STOPPING
            self.swing_t += DT / SWING_TIME
            if self.swing_t >= 1.0:
                self.foot_world[self.swing_side] = self.swing_end.copy()
                self._advance_body()
                self._start_swing(Side.OTHER[self.swing_side])

        elif self.state == GaitState.STOPPING:
            self.swing_t += DT / SWING_TIME
            if self.swing_t >= 1.0:
                self.foot_world[self.swing_side] = self.swing_end.copy()
                self._advance_body()
                lf, rf = self.foot_world[Side.LEFT], self.foot_world[Side.RIGHT]
                if abs(lf[0] - rf[0]) < STEP_LEN * 0.6:
                    self.state = GaitState.IDLE
                else:
                    self._start_swing(Side.OTHER[self.swing_side])
            if cmd == "forward":
                self.state = GaitState.STEPPING

        elif self.state == GaitState.TURNING:
            self.swing_t += DT / SWING_TIME
            # Smoothly rotate heading toward turn target during swing
            hdiff = (self.turn_heading_end - self.heading + math.pi) % (2*math.pi) - math.pi
            self.heading += hdiff * min(1.0, 4.0 * DT)

            if self.swing_t >= 1.0:
                # Foot lands — snap to exact target heading
                self.foot_world[self.swing_side] = self.swing_end.copy()
                self.heading = self.turn_heading_end
                self.target_heading = self.heading

                # Update body centre
                lf = self.foot_world[Side.LEFT]
                rf = self.foot_world[Side.RIGHT]
                self.body_pos = np.array([(lf[0]+rf[0])*0.5, (lf[1]+rf[1])*0.5])

                # Is there another turn queued?
                next_turn = self._pop_turn()
                if next_turn is not None:
                    # Execute next queued turn immediately
                    resume = GaitState.STEPPING if cmd == "forward" else GaitState.IDLE
                    self._begin_turn(next_turn, resume_state=resume)
                else:
                    # No more turns — go back to whatever we should be doing
                    self.state = self.turn_resume
                    if self.state == GaitState.STEPPING:
                        self._start_swing(Side.OTHER[self.swing_side])
                    elif self.state == GaitState.IDLE and cmd == "forward":
                        # w is now held — start walking
                        self._start_swing(Side.LEFT)
                        self.state = GaitState.STEPPING

        # ── Smooth heading for non-turning states ─────────────────────────────
        if self.state != GaitState.TURNING:
            hdiff = (self.target_heading - self.heading + math.pi) % (2*math.pi) - math.pi
            self.heading += hdiff * min(1.0, 6.0 * DT)

        # ── Bezier foot interpolation ─────────────────────────────────────────
        if self.state != GaitState.IDLE and self.swing_curve is not None:
            t = max(0.0, min(1.0, self.swing_t))
            self.foot_world[self.swing_side] = bezier(self.swing_curve, t)

        feet_hip_rel = [
            self.foot_world[s] - self._hip_world(s)
            for s in (Side.LEFT, Side.RIGHT)
        ]

        with self.cmd_lock:
            q_depth = len(self.turn_queue)

        return {
            "state":        self.state,
            "heading":      self.heading,
            "body_pos":     self.body_pos.copy(),
            "foot_world":   [f.copy() for f in self.foot_world],
            "feet_hip_rel": feet_hip_rel,
            "swing_side":   self.swing_side,
            "swing_t":      self.swing_t,
            "swing_curve":  self.swing_curve,
            "speed_scale":  self.speed_scale,
            "turn_queue":   q_depth,
        }

# ─── Debug window ─────────────────────────────────────────────────────────────

class DebugWindow:
    POLL_MS = 40

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("gait.py — debug")
        root.geometry("360x290")
        root.resizable(False, False)
        root.configure(bg="#1a1a2e")

        # last command
        self.cmd_frame = tk.Frame(root, bg="#444444", height=90)
        self.cmd_frame.pack(fill=tk.X, padx=16, pady=(14, 2))
        self.cmd_frame.pack_propagate(False)
        self.cmd_label = tk.Label(
            self.cmd_frame, text="STOP",
            font=("Helvetica", 30, "bold"), fg="white", bg="#444444")
        self.cmd_label.pack(expand=True)
        tk.Label(root, text="last command", font=("Helvetica", 9),
                 fg="#556677", bg="#1a1a2e").pack()

        # gait state
        self.state_frame = tk.Frame(root, bg="#555555", height=56)
        self.state_frame.pack(fill=tk.X, padx=16, pady=(10, 2))
        self.state_frame.pack_propagate(False)
        self.state_label = tk.Label(
            self.state_frame, text="IDLE",
            font=("Helvetica", 20, "bold"), fg="white", bg="#555555")
        self.state_label.pack(expand=True)
        tk.Label(root, text="gait state", font=("Helvetica", 9),
                 fg="#556677", bg="#1a1a2e").pack()

        # turn queue depth
        self.queue_var = tk.StringVar(value="turns queued: 0")
        tk.Label(root, textvariable=self.queue_var,
                 font=("Courier", 9), fg="#aaaaff",
                 bg="#1a1a2e").pack(pady=(4, 0))

        # IK status indicator
        ik_row = tk.Frame(root, bg="#1a1a2e")
        ik_row.pack(pady=(2, 0))
        self._ik_dot = tk.Canvas(ik_row, width=12, height=12,
                                  bg="#1a1a2e", highlightthickness=0)
        self._ik_dot.pack(side=tk.LEFT, padx=(0, 4))
        self._ik_dot_id = self._ik_dot.create_oval(2, 2, 10, 10,
                                                     fill="#444444", outline="")
        self._ik_label = tk.Label(ik_row, text="IK: not loaded",
                                   font=("Helvetica", 9), fg="#556677",
                                   bg="#1a1a2e")
        self._ik_label.pack(side=tk.LEFT)
        self._q_ik = None

        # connection indicator
        dot_row = tk.Frame(root, bg="#1a1a2e")
        dot_row.pack(pady=(10, 0))
        self.dot = tk.Canvas(dot_row, width=12, height=12,
                             bg="#1a1a2e", highlightthickness=0)
        self.dot.pack(side=tk.LEFT, padx=(0, 4))
        self._dot_id = self.dot.create_oval(2, 2, 10, 10,
                                             fill="#444444", outline="")
        self.conn_label = tk.Label(dot_row, text="sender not connected",
                                   font=("Helvetica", 9), fg="#556677",
                                   bg="#1a1a2e")
        self.conn_label.pack(side=tk.LEFT)

        self.info_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.info_var,
                 font=("Courier", 9), fg="#445566",
                 bg="#1a1a2e").pack(pady=(4, 0))

        self._lock    = threading.Lock()
        self._q_cmd   = None
        self._q_state = None
        self._q_conn  = None
        self._q_info  = None
        self._q_queue = None

        root.after(self.POLL_MS, self._poll)

    def push(self, cmd=None, state=None, connected=None, info=None, turn_queue=None, ik_status=None):
        with self._lock:
            if cmd        is not None: self._q_cmd   = cmd
            if state      is not None: self._q_state = state
            if connected  is not None: self._q_conn  = connected
            if info       is not None: self._q_info  = info
            if turn_queue is not None: self._q_queue = turn_queue
            if ik_status  is not None: self._q_ik    = ik_status

    def _poll(self):
        with self._lock:
            cmd   = self._q_cmd;   self._q_cmd   = None
            state = self._q_state; self._q_state = None
            conn  = self._q_conn;  self._q_conn  = None
            info  = self._q_info;  self._q_info  = None
            queue = self._q_queue; self._q_queue = None
            ik    = self._q_ik;    self._q_ik    = None

        if cmd is not None:
            bg, fg, text = CMD_COLORS.get(cmd, CMD_DEFAULT)
            self.cmd_frame.configure(bg=bg)
            self.cmd_label.configure(bg=bg, fg=fg, text=text)

        if state is not None:
            bg = STATE_COLORS.get(state, "#555555")
            self.state_frame.configure(bg=bg)
            self.state_label.configure(bg=bg, text=state)

        if conn is not None:
            if conn:
                self.dot.itemconfigure(self._dot_id, fill="#2ecc71")
                self.conn_label.configure(text="sender connected")
            else:
                self.dot.itemconfigure(self._dot_id, fill="#e74c3c")
                self.conn_label.configure(text="sender disconnected")

        if info is not None:
            self.info_var.set(info)

        if queue is not None:
            self.queue_var.set(f"turns queued: {queue}")

        if ik is not None:
            active, msg = ik
            color = "#2ecc71" if active else "#e74c3c"
            self._ik_dot.itemconfigure(self._ik_dot_id, fill=color)
            self._ik_label.configure(text=msg, fg="#aaffaa" if active else "#ff8888")

        self.root.after(self.POLL_MS, self._poll)

# ─── YARP command listener thread ─────────────────────────────────────────────

class CmdListener(threading.Thread):
    def __init__(self, gait: GaitCoordinator, window: DebugWindow):
        super().__init__(daemon=True)
        self.gait       = gait
        self.window     = window
        self.port       = yarp.Port()
        self._stop_evt  = threading.Event()
        self.connected  = False
        self.last_retry = 0.0

    def run(self):
        self.port.open(PORT_CMD)
        print(f"[gait] cmd port open: {PORT_CMD}")
        print(f"[gait] waiting for sender on {PORT_CMD_SRC} ...")

        while not self._stop_evt.is_set():
            now = time.monotonic()

            actually_connected = yarp.Network.isConnected(PORT_CMD_SRC, PORT_CMD)
            if actually_connected != self.connected:
                self.connected = actually_connected
                if actually_connected:
                    print(f"[gait] OK  sender connected: {PORT_CMD_SRC} -> {PORT_CMD}")
                else:
                    print(f"[gait] !!  sender disconnected")
                self.window.push(connected=actually_connected)

            if not actually_connected and (now - self.last_retry) > 1.0:
                yarp.Network.connect(PORT_CMD_SRC, PORT_CMD)
                self.last_retry = now

            b = yarp.Bottle()
            if self.port.read(b, False):
                msg = b.get(0).asString() if b.size() > 0 else ""
                if msg:
                    print(f"[gait] << '{msg}'  (state={self.gait.state})")
                    self.gait.set_command(msg)
                    self.window.push(cmd=msg)
            else:
                time.sleep(0.01)

    def stop(self):
        self._stop_evt.set()
        self.port.interrupt()
        self.port.close()

# ─── Gait loop (background thread) ───────────────────────────────────────────

def gait_loop(gait: GaitCoordinator, window: DebugWindow,
              port_left, port_right, port_viz,
              port_left_angles, port_right_angles, ik):
    viz_connected  = False
    last_viz_check = 0.0
    last_state     = None

    while True:
        t0 = time.monotonic()

        snap = gait.tick()

        # Push state changes to debug window
        if snap["state"] != last_state:
            last_state = snap["state"]
            window.push(state=snap["state"])

        # Always push turn queue depth so it stays fresh
        window.push(turn_queue=snap["turn_queue"])

        # Viz connection check every 1 s
        now = time.monotonic()
        if now - last_viz_check > 1.0:
            viz_now = yarp.Network.isConnected(PORT_VIZ, "/viz/in")
            if viz_now != viz_connected:
                viz_connected = viz_now
                if viz_now:
                    print(f"[gait] OK  viz connected: {PORT_VIZ} -> /viz/in")
                    window.push(info="viz connected")
                else:
                    print(f"[gait] !!  viz not connected")
                    window.push(info="viz not connected")
            last_viz_check = now

        # Publish foot positions (xyz, hip-relative)
        for port, rel in zip((port_left, port_right), snap["feet_hip_rel"]):
            b = yarp.Bottle()
            b.clear()
            for v in rel:
                b.addFloat64(float(v))
            port.write(b)

        # Publish joint angles via IK
        if ik is not None:
            window.push(ik_status=(True, "IK active ✓  humanoid_beta.urdf"))
            foot_l = snap["foot_world"][0]
            foot_r = snap["foot_world"][1]
            bp     = snap["body_pos"]
            hdg    = snap["heading"]

            # Convert foot from world frame → hip-relative frame.
            #
            # The IK solver chains start from GE_27_1 / GE_27_2 (hip attachment
            # links), so targets must be expressed relative to each hip joint,
            # not relative to the body centre.
            #
            # In gait.py the hip positions are:
            #   right hip world XY = body_pos + rotate([ HIP_WIDTH, 0], heading)  (right = -Y in gait convention... but URDF has right at -X)
            #   left  hip world XY = body_pos + rotate([-HIP_WIDTH, 0], heading)
            # Z of hip joint ≈ 0 (hip is at body height, feet are below)
            #
            # Steps:
            #   1. Compute hip world position for each leg
            #   2. Subtract hip position from foot world position
            #   3. Rotate by -heading so x = forward for the robot

            cos_h = math.cos(-hdg)
            sin_h = math.sin(-hdg)

            def hip_world(side):
                # side: 0=left, 1=right in gait.py convention
                # Left hip is at +HIP_WIDTH lateral, right at -HIP_WIDTH
                sign = -1 if side == 0 else 1   # 0=left(-), 1=right(+) in gait perp
                perp_x = -math.sin(hdg)
                perp_y =  math.cos(hdg)
                hx = bp[0] + sign * HIP_WIDTH * perp_x
                hy = bp[1] + sign * HIP_WIDTH * perp_y
                return np.array([hx, hy, 0.0])

            def world_to_hip_rel(foot, hip):
                dx = foot[0] - hip[0]
                dy = foot[1] - hip[1]
                dz = foot[2] - hip[2]
                return np.array([
                    cos_h * dx - sin_h * dy,   # forward in robot frame
                    sin_h * dx + cos_h * dy,   # lateral in robot frame
                    dz                          # z unchanged (up is up)
                ])

            hip_r = hip_world(1)   # right
            hip_l = hip_world(0)   # left
            right_rel = world_to_hip_rel(foot_r, hip_r)
            left_rel  = world_to_hip_rel(foot_l, hip_l)

            right_angles = ik.solve_right(right_rel)
            left_angles  = ik.solve_left(left_rel)

            # Pack into YARP bottle in fixed order
            # Right: 6 joints, Left: 5 joints (humanoid_beta.urdf asymmetry)
            for port_a, angles, order in [
                (port_right_angles, right_angles, RIGHT_JOINT_ORDER),
                (port_left_angles,  left_angles,  LEFT_JOINT_ORDER),
            ]:
                ab = yarp.Bottle()
                ab.clear()
                for joint in order:
                    ab.addFloat64(angles.get(joint, 0.0))
                port_a.write(ab)

        # Publish viz bottle
        vb = yarp.Bottle()
        vb.addString(snap["state"])
        vb.addFloat64(float(snap["heading"]))
        vb.addFloat64(float(snap["body_pos"][0]))
        vb.addFloat64(float(snap["body_pos"][1]))
        for x in snap["foot_world"][0]: vb.addFloat64(float(x))
        for x in snap["foot_world"][1]: vb.addFloat64(float(x))
        vb.addInt32(int(snap["swing_side"]))
        vb.addFloat64(float(snap["swing_t"]))
        vb.addFloat64(float(snap["speed_scale"]))
        if snap["swing_curve"] is not None:
            for pt in snap["swing_curve"]:
                for x in pt:
                    vb.addFloat64(float(x))
        port_viz.write(vb)

        elapsed = time.monotonic() - t0
        wait    = DT - elapsed
        if wait > 0:
            time.sleep(wait)

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    yarp.Network.init()
    if not yarp.Network.checkNetwork(3.0):
        print("[gait] ERROR: yarpserver not reachable.")
        sys.exit(1)

    # Load swing curve profile from file
    swing_profile = load_swing_profile()

    gait = GaitCoordinator(swing_profile=swing_profile)

    # tkinter must own the main thread
    root   = tk.Tk()
    window = DebugWindow(root)

    listener = CmdListener(gait, window)
    listener.start()

    port_left  = yarp.Port(); port_left.open(PORT_LEFT);   port_left.enableBackgroundWrite(True)
    port_right = yarp.Port(); port_right.open(PORT_RIGHT); port_right.enableBackgroundWrite(True)
    port_viz   = yarp.Port(); port_viz.open(PORT_VIZ);     port_viz.enableBackgroundWrite(True)

    port_left_angles  = yarp.Port(); port_left_angles.open(PORT_LEFT_ANGLES);   port_left_angles.enableBackgroundWrite(True)
    port_right_angles = yarp.Port(); port_right_angles.open(PORT_RIGHT_ANGLES); port_right_angles.enableBackgroundWrite(True)

    # Load IK solver (graceful fallback if ikpy not installed)
    ik = None
    if _IK_AVAILABLE:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            ik = LegIK(urdf_dir=here)   # looks for leg_right.urdf + leg_left.urdf
        except Exception as e:
            print(f"[gait] WARNING: IK solver failed to load ({e}). Angles will not be published.")

    print(f"[gait] running at {RATE_HZ}Hz  |  IK: {'ON' if ik else 'OFF'}")
    if not ik:
        window.push(ik_status=(False, "IK: not loaded (pip install ikpy)"))
    print(f"[gait] ports: cmd={PORT_CMD}  left={PORT_LEFT}  right={PORT_RIGHT}  viz={PORT_VIZ}")
    print(f"[gait] angle ports: {PORT_LEFT_ANGLES}  {PORT_RIGHT_ANGLES}")
    if ik:
        print(f"[gait] angle order: {LEFT_JOINT_ORDER}")

    threading.Thread(
        target=gait_loop,
        args=(gait, window, port_left, port_right, port_viz,
              port_left_angles, port_right_angles, ik),
        daemon=True
    ).start()

    try:
        root.protocol("WM_DELETE_WINDOW", root.quit)
        root.mainloop()
    finally:
        listener.stop()
        port_left.close()
        port_right.close()
        port_viz.close()
        port_left_angles.close()
        port_right_angles.close()
        yarp.Network.fini()
        print("\n[gait] bye.")

if __name__ == "__main__":
    main()