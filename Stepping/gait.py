#!/usr/bin/env python3
"""
gait.py

Listens on  /robot/cmd
Publishes   /gait/left/foot    x y z hip-relative (metres)
            /gait/right/foot
            /gait/left/angles  hip_roll hip_pitch knee_pitch ankle_pitch ankle_roll ankle_yaw (radians)
            /gait/right/angles
            /gait/viz          full state for visualizer

Swing curves loaded from swing_curves.yaml (same folder).
IK solved via ikpy + biped.urdf (same folder).

    pip install ikpy pyyaml
    python3 gait.py
"""

import time, sys, math, threading, os
import numpy as np
import yarp

try:
    from ik_solver import LegIK
    _IK_AVAILABLE = True
except ImportError as _e:
    print(f"[gait] IK unavailable: {_e}")
    _IK_AVAILABLE = False

try:
    import yaml as _yaml
    def _load_yaml(path):
        with open(path) as f:
            return _yaml.safe_load(f)
except ImportError:
    def _load_yaml(path):
        raise RuntimeError("PyYAML not installed — run: pip install pyyaml")

HIP_WIDTH    = 0.08
THIGH_LEN    = 0.15
SHANK_LEN    = 0.15
STEP_LEN     = 0.06
STEP_HEIGHT  = 0.05
STAND_HEIGHT = -(THIGH_LEN + SHANK_LEN) * 0.85

RATE_HZ         = 50
DT              = 1.0 / RATE_HZ
SWING_TIME      = 0.40
TURN_STEP_ANGLE = math.radians(20)

PORT_CMD          = "/robot/cmd"
PORT_CMD_SRC      = "/debug/sender"
PORT_LEFT         = "/gait/left/foot"
PORT_RIGHT        = "/gait/right/foot"
PORT_LEFT_ANGLES  = "/gait/left/angles"
PORT_RIGHT_ANGLES = "/gait/right/angles"
PORT_VIZ          = "/gait/viz"

LEFT_JOINT_ORDER  = ["left_hip_roll",  "left_hip_pitch",  "left_knee_pitch",
                     "left_ankle_pitch",  "left_ankle_roll",  "left_ankle_yaw"]
RIGHT_JOINT_ORDER = ["right_hip_roll", "right_hip_pitch", "right_knee_pitch",
                     "right_ankle_pitch", "right_ankle_roll", "right_ankle_yaw"]

_DEFAULT_CURVE_PROFILE = [
    [0.00, 0.00, 0.00],
    [0.15, 0.00, 0.20],
    [0.35, 0.00, 0.90],
    [0.50, 0.00, 1.00],
    [0.70, 0.00, 0.60],
    [1.00, 0.00, 0.00],
]


def load_swing_profile(yaml_path: str = None) -> list:
    if yaml_path is None:
        here      = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(here, "swing_curves.yaml")
        print(f"[gait] swing curves: {yaml_path}")
    try:
        data        = _load_yaml(yaml_path)
        active_name = data.get("active_curve", "normal")
        curves      = data.get("curves", {})
        if active_name not in curves:
            print(f"[gait] curve '{active_name}' not found, using default")
            return _DEFAULT_CURVE_PROFILE
        print(f"[gait] loaded curve '{active_name}'")
        return curves[active_name]["points"]
    except FileNotFoundError:
        print(f"[gait] swing_curves.yaml not found, using default")
        return _DEFAULT_CURVE_PROFILE
    except Exception as e:
        print(f"[gait] curve load error ({e}), using default")
        return _DEFAULT_CURVE_PROFILE


def make_swing_curve(start, end, height, profile=None):
    if profile is None:
        profile = _DEFAULT_CURVE_PROFILE
    s     = start.copy()
    e     = end.copy()
    horiz = np.array([e[0]-s[0], e[1]-s[1], 0.0])
    hl    = np.linalg.norm(horiz[:2])
    perp  = np.array([-horiz[1], horiz[0], 0.0]) / hl if hl > 1e-6 else np.array([0., 1., 0.])
    pts   = []
    for xf, yf, zf in profile:
        pts.append(np.array([
            s[0] + xf*horiz[0] + yf*perp[0],
            s[1] + xf*horiz[1] + yf*perp[1],
            s[2] + (e[2]-s[2])*xf + zf*height,
        ]))
    pts[0]  = s.copy()
    pts[-1] = e.copy()
    return np.array(pts, dtype=float)


def bezier(pts, t):
    p = pts.copy()
    n = len(p)
    for r in range(1, n):
        p[:n-r] = (1-t)*p[:n-r] + t*p[1:n-r+1]
    return p[0]


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
    def __init__(self, swing_profile=None):
        self.swing_profile    = swing_profile if swing_profile is not None else _DEFAULT_CURVE_PROFILE
        self.heading          = 0.0
        self.target_heading   = 0.0
        self.body_pos         = np.array([0.0, 0.0])
        self.foot_world       = [
            np.array([-HIP_WIDTH, 0.0, STAND_HEIGHT]),
            np.array([ HIP_WIDTH, 0.0, STAND_HEIGHT]),
        ]
        self.state            = GaitState.IDLE
        self.swing_side       = Side.LEFT
        self.swing_t          = 0.0
        self.swing_curve      = None
        self.swing_end        = None
        self.speed_scale      = 1.0
        self.turn_queue       = []
        self.turn_dir         = 0
        self.turn_resume      = GaitState.IDLE
        self.turn_pivot_side  = Side.LEFT
        self.turn_heading_end = 0.0
        self.cmd_lock         = threading.Lock()
        self.cmd              = "stop"

    def set_command(self, c):
        with self.cmd_lock:
            if c in ("left", "right"):
                self.turn_queue.append(c)
                print(f"[gait] turn queued: {c} (depth={len(self.turn_queue)})")
            else:
                self.cmd = c

    def _get_state(self):
        with self.cmd_lock:
            return self.cmd, list(self.turn_queue)

    def _pop_turn(self):
        with self.cmd_lock:
            return self.turn_queue.pop(0) if self.turn_queue else None

    def _hip_world(self, side):
        sign   = -1 if side == Side.LEFT else 1
        perp   = np.array([-math.sin(self.heading), math.cos(self.heading)])
        hip_xy = self.body_pos + sign * HIP_WIDTH * perp
        return np.array([hip_xy[0], hip_xy[1], 0.0])

    def _next_foot_pos(self, side):
        fwd  = np.array([math.cos(self.heading), math.sin(self.heading)])
        perp = np.array([-math.sin(self.heading), math.cos(self.heading)])
        sign = -1 if side == Side.LEFT else 1
        xy   = self.body_pos + fwd * STEP_LEN * self.speed_scale + sign * HIP_WIDTH * perp
        return np.array([xy[0], xy[1], STAND_HEIGHT])

    def _turn_foot_pos(self, swing_side, new_heading):
        pivot = self.foot_world[Side.OTHER[swing_side]].copy()
        sign  = -1 if swing_side == Side.LEFT else 1
        perp  = np.array([-math.sin(new_heading), math.cos(new_heading)])
        xy    = pivot[:2] + sign * 2 * HIP_WIDTH * perp
        return np.array([xy[0], xy[1], STAND_HEIGHT])

    def _advance_body(self):
        fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
        self.body_pos = self.body_pos + fwd * STEP_LEN * self.speed_scale * 0.5

    def _start_swing(self, side):
        self.swing_side  = side
        self.swing_t     = 0.0
        start            = self.foot_world[side].copy()
        end              = self._next_foot_pos(side)
        self.swing_end   = end
        self.swing_curve = make_swing_curve(start, end, STEP_HEIGHT, self.swing_profile)

    def _begin_turn(self, direction, resume_state):
        self.turn_dir         = +1 if direction == "left" else -1
        self.turn_resume      = resume_state
        self.turn_pivot_side  = Side.LEFT  if direction == "left" else Side.RIGHT
        swing_side            = Side.RIGHT if direction == "left" else Side.LEFT
        self.turn_heading_end = self.heading + self.turn_dir * TURN_STEP_ANGLE
        start                 = self.foot_world[swing_side].copy()
        end                   = self._turn_foot_pos(swing_side, self.turn_heading_end)
        self.swing_side       = swing_side
        self.swing_t          = 0.0
        self.swing_end        = end
        self.swing_curve      = make_swing_curve(start, end, STEP_HEIGHT, self.swing_profile)
        self.state            = GaitState.TURNING

    def tick(self):
        cmd, turn_queue = self._get_state()

        if turn_queue and self.state != GaitState.TURNING:
            direction = self._pop_turn()
            resume    = GaitState.STEPPING if cmd == "forward" else GaitState.IDLE
            self._begin_turn(direction, resume_state=resume)

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
            hdiff = (self.turn_heading_end - self.heading + math.pi) % (2*math.pi) - math.pi
            self.heading += hdiff * min(1.0, 4.0 * DT)

            if self.swing_t >= 1.0:
                self.foot_world[self.swing_side] = self.swing_end.copy()
                self.heading        = self.turn_heading_end
                self.target_heading = self.heading
                lf = self.foot_world[Side.LEFT]
                rf = self.foot_world[Side.RIGHT]
                self.body_pos = np.array([(lf[0]+rf[0])*0.5, (lf[1]+rf[1])*0.5])

                next_turn = self._pop_turn()
                if next_turn is not None:
                    resume = GaitState.STEPPING if cmd == "forward" else GaitState.IDLE
                    self._begin_turn(next_turn, resume_state=resume)
                else:
                    self.state = self.turn_resume
                    if self.state == GaitState.STEPPING:
                        self._start_swing(Side.OTHER[self.swing_side])
                    elif self.state == GaitState.IDLE and cmd == "forward":
                        self._start_swing(Side.LEFT)
                        self.state = GaitState.STEPPING

        if self.state != GaitState.TURNING:
            hdiff = (self.target_heading - self.heading + math.pi) % (2*math.pi) - math.pi
            self.heading += hdiff * min(1.0, 6.0 * DT)

        if self.state != GaitState.IDLE and self.swing_curve is not None:
            self.foot_world[self.swing_side] = bezier(self.swing_curve, max(0., min(1., self.swing_t)))

        feet_hip_rel = [self.foot_world[s] - self._hip_world(s) for s in (Side.LEFT, Side.RIGHT)]

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
        }


class CmdListener(threading.Thread):
    def __init__(self, gait):
        super().__init__(daemon=True)
        self.gait       = gait
        self.port       = yarp.Port()
        self._stop_evt  = threading.Event()
        self.connected  = False
        self.last_retry = 0.0

    def run(self):
        self.port.open(PORT_CMD)
        print(f"[gait] cmd port: {PORT_CMD}")
        while not self._stop_evt.is_set():
            now = time.monotonic()
            actually_connected = yarp.Network.isConnected(PORT_CMD_SRC, PORT_CMD)
            if actually_connected != self.connected:
                self.connected = actually_connected
                print(f"[gait] sender {'connected' if actually_connected else 'disconnected'}")
            if not actually_connected and (now - self.last_retry) > 1.0:
                yarp.Network.connect(PORT_CMD_SRC, PORT_CMD)
                self.last_retry = now
            b = yarp.Bottle()
            if self.port.read(b, False):
                msg = b.get(0).asString() if b.size() > 0 else ""
                if msg:
                    print(f"[gait] << '{msg}'  state={self.gait.state}")
                    self.gait.set_command(msg)
            else:
                time.sleep(0.01)

    def stop(self):
        self._stop_evt.set()
        self.port.interrupt()
        self.port.close()


def gait_loop(gait, port_left, port_right, port_viz, port_left_angles, port_right_angles, ik):
    viz_connected  = False
    last_viz_check = 0.0

    while True:
        t0   = time.monotonic()
        snap = gait.tick()

        now = time.monotonic()
        if now - last_viz_check > 1.0:
            viz_now = yarp.Network.isConnected(PORT_VIZ, "/viz/in")
            if viz_now != viz_connected:
                viz_connected = viz_now
                print(f"[gait] viz {'connected' if viz_now else 'disconnected'}")
            last_viz_check = now

        for port, rel in zip((port_left, port_right), snap["feet_hip_rel"]):
            b = yarp.Bottle()
            for v in rel:
                b.addFloat64(float(v))
            port.write(b)

        if ik is not None:
            foot_l = snap["foot_world"][0]
            foot_r = snap["foot_world"][1]
            bp     = snap["body_pos"]
            hdg    = snap["heading"]
            cos_h  = math.cos(-hdg)
            sin_h  = math.sin(-hdg)

            def hip_world(side):
                sign = -1 if side == Side.LEFT else 1
                px   = -math.sin(hdg)
                py   =  math.cos(hdg)
                return np.array([bp[0] + sign*HIP_WIDTH*px, bp[1] + sign*HIP_WIDTH*py, 0.0])

            def to_hip_rel(foot, hip):
                dx, dy, dz = foot[0]-hip[0], foot[1]-hip[1], foot[2]-hip[2]
                return np.array([cos_h*dx - sin_h*dy, sin_h*dx + cos_h*dy, dz])

            left_angles  = ik.solve_left( to_hip_rel(foot_l, hip_world(Side.LEFT)))
            right_angles = ik.solve_right(to_hip_rel(foot_r, hip_world(Side.RIGHT)))

            for port_a, angles, order in [
                (port_left_angles,  left_angles,  LEFT_JOINT_ORDER),
                (port_right_angles, right_angles, RIGHT_JOINT_ORDER),
            ]:
                ab = yarp.Bottle()
                for joint in order:
                    ab.addFloat64(angles.get(joint, 0.0))
                port_a.write(ab)

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

        wait = DT - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)


def main():
    yarp.Network.init()
    if not yarp.Network.checkNetwork(3.0):
        print("[gait] ERROR: yarpserver not reachable")
        sys.exit(1)

    gait = GaitCoordinator(swing_profile=load_swing_profile())

    listener = CmdListener(gait)
    listener.start()

    def make_port(name):
        p = yarp.Port()
        p.open(name)
        p.enableBackgroundWrite(True)
        return p

    port_left         = make_port(PORT_LEFT)
    port_right        = make_port(PORT_RIGHT)
    port_viz          = make_port(PORT_VIZ)
    port_left_angles  = make_port(PORT_LEFT_ANGLES)
    port_right_angles = make_port(PORT_RIGHT_ANGLES)

    ik = None
    if _IK_AVAILABLE:
        try:
            ik = LegIK()
        except Exception as e:
            print(f"[gait] IK load failed: {e}")

    print(f"[gait] {RATE_HZ}Hz  IK={'on' if ik else 'off'}")

    threading.Thread(
        target=gait_loop,
        args=(gait, port_left, port_right, port_viz, port_left_angles, port_right_angles, ik),
        daemon=True
    ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        for p in (port_left, port_right, port_viz, port_left_angles, port_right_angles):
            p.close()
        yarp.Network.fini()
        print("[gait] bye")


if __name__ == "__main__":
    main()
