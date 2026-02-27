#!/usr/bin/env python3
"""
visualizer.py — 3D matplotlib stick figure for gait debug

Receives full state from /viz/in (published by gait.py).
Camera is fixed in azimuth (side view), tracks robot position.

Viz bottle layout (from gait.py):
  [0]       state string
  [1]       heading (rad)
  [2],[3]   body_pos x,y
  [4-6]     left  foot world x,y,z
  [7-9]     right foot world x,y,z
  [10]      swing_side (int)
  [11]      swing_t
  [12]      speed_scale
  [13-30]   swing curve control points (6×3), may be absent if IDLE

Run standalone (no YARP) with --demo for a preview:
    python3 visualizer.py --demo
Or normally:
    python3 visualizer.py
"""

import sys, threading, time, math, argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yarp

# ─── Robot geometry (must match gait.py) ──────────────────────────────────────
HIP_WIDTH    = 0.08
THIGH_LEN    = 0.15
SHANK_LEN    = 0.15
STAND_HEIGHT = -(THIGH_LEN + SHANK_LEN) * 0.85
PELVIS_H     =  0.04   # pelvis block above hip joints
TORSO_H      =  0.20   # torso above pelvis

# ─── Colours ──────────────────────────────────────────────────────────────────
C_BODY      = "#e8e0d0"
C_LEFT      = "#4fc3f7"    # light blue  — left leg
C_RIGHT     = "#ef9a9a"    # salmon      — right leg
C_SWING     = "#ffd54f"    # amber       — swinging foot trail
C_BEZIER    = "#80cbc4"    # teal        — bezier curve
C_CP        = "#b2dfdb"    # bezier control points
C_GROUND    = "#2d3436"
C_STATE     = "#dfe6e9"
C_HEADING   = "#fdcb6e"
C_FOOT_L    = C_LEFT
C_FOOT_R    = C_RIGHT

# ─── Ports ────────────────────────────────────────────────────────────────────
PORT_IN   = "/viz/in"
PORT_SRC  = "/gait/viz"

# ─── Bezier sampler ───────────────────────────────────────────────────────────

def bezier_pts(ctrl: np.ndarray, n: int = 40) -> np.ndarray:
    """Sample n points along a Bezier curve defined by ctrl (k,3)."""
    result = []
    for t in np.linspace(0, 1, n):
        p = ctrl.copy()
        for r in range(1, len(p)):
            p[:len(p)-r] = (1-t)*p[:len(p)-r] + t*p[1:len(p)-r+1]
        result.append(p[0])
    return np.array(result)

# ─── Analytical 2-link IK (sagittal plane) ────────────────────────────────────

def solve_ik(foot_rel: np.ndarray, l1=THIGH_LEN, l2=SHANK_LEN):
    fx, fy, fz = foot_rel
    # Work in sagittal plane: forward = x, up = z
    dist2 = fx*fx + fz*fz
    dist  = math.sqrt(dist2)
    dist  = min(dist, l1+l2-1e-4)
    dist  = max(dist, abs(l1-l2)+1e-4)

    cos_knee = (l1**2 + l2**2 - dist2) / (2*l1*l2)
    cos_knee = max(-1.0, min(1.0, cos_knee))
    knee_angle = math.pi - math.acos(cos_knee)

    alpha = math.atan2(fx, -fz)
    cos_a = (l1**2 + dist2 - l2**2) / (2*l1*dist)
    cos_a = max(-1.0, min(1.0, cos_a))
    beta  = math.acos(cos_a)
    hip_angle = alpha + beta   # + bends knee forward (correct for biped)

    kx = l1 * math.sin(hip_angle)
    kz = -l1 * math.cos(hip_angle)
    return (0.0, 0.0, kx, kz, fx, fz)

# ─── State container ──────────────────────────────────────────────────────────

class RobotState:
    def __init__(self):
        self.lock        = threading.Lock()
        self.gait_state  = "IDLE"
        self.heading     = 0.0
        self.body_pos    = np.array([0.0, 0.0])
        self.foot_world  = [
            np.array([-HIP_WIDTH, 0.0, STAND_HEIGHT]),
            np.array([ HIP_WIDTH, 0.0, STAND_HEIGHT]),
        ]
        self.swing_side  = 0
        self.swing_t     = 0.0
        self.speed_scale = 1.0
        self.ctrl_pts    = None   # (6,3) or None

    def update_from_bottle(self, b: yarp.Bottle):
        if b.size() < 13:
            return
        with self.lock:
            self.gait_state  = b.get(0).asString()
            self.heading     = b.get(1).asFloat64()
            self.body_pos    = np.array([b.get(2).asFloat64(),
                                         b.get(3).asFloat64()])
            self.foot_world  = [
                np.array([b.get(4).asFloat64(),
                           b.get(5).asFloat64(),
                           b.get(6).asFloat64()]),
                np.array([b.get(7).asFloat64(),
                           b.get(8).asFloat64(),
                           b.get(9).asFloat64()]),
            ]
            self.swing_side  = b.get(10).asInt32()
            self.swing_t     = b.get(11).asFloat64()
            self.speed_scale = b.get(12).asFloat64()
            # Control points: 6*3 = 18 values starting at index 13
            if b.size() >= 13 + 18:
                pts = []
                for i in range(6):
                    base = 13 + i*3
                    pts.append([b.get(base).asFloat64(),
                                 b.get(base+1).asFloat64(),
                                 b.get(base+2).asFloat64()])
                self.ctrl_pts = np.array(pts)
            else:
                self.ctrl_pts = None

    def snapshot(self):
        with self.lock:
            return {
                "gait_state":  self.gait_state,
                "heading":     self.heading,
                "body_pos":    self.body_pos.copy(),
                "foot_world":  [f.copy() for f in self.foot_world],
                "swing_side":  self.swing_side,
                "swing_t":     self.swing_t,
                "speed_scale": self.speed_scale,
                "ctrl_pts":    self.ctrl_pts.copy() if self.ctrl_pts is not None else None,
            }

# ─── YARP listener thread ──────────────────────────────────────────────────────

class VizListener(threading.Thread):
    def __init__(self, state: RobotState):
        super().__init__(daemon=True)
        self.state = state
        self.port  = yarp.Port()

    def run(self):
        # Network.init already called on main thread — do NOT call again here
        self.port.open(PORT_IN)
        print(f"[viz] port open: {PORT_IN}")

        # block until connected — same pattern as sniff.py which we know works
        print(f"[viz] connecting {PORT_SRC} -> {PORT_IN} ...")
        while not yarp.Network.connect(PORT_SRC, PORT_IN):
            print(f"[viz] retrying...")
            time.sleep(1.0)
        print(f"[viz] CONNECTED — reading")

        frames = 0
        while True:
            b = yarp.Bottle()
            self.port.read(b)        # BLOCKING — no poll, no sleep, just waits
            self.state.update_from_bottle(b)
            frames += 1
            if frames == 1:
                print(f"[viz] first frame! size={b.size()}")
            elif frames % 500 == 0:
                print(f"[viz] {frames} frames received")

    def stop(self):
        self.port.interrupt()
        self.port.close()

# ─── Demo animator (no YARP) ──────────────────────────────────────────────────

def demo_tick(state: RobotState, frame: int):
    """Animate a fake walking cycle for --demo mode."""
    t   = frame * 0.04
    cyc = t % 2.0
    progress = (cyc % 1.0)

    state.gait_state  = "STEPPING"
    state.heading     = math.sin(t * 0.3) * 0.4
    state.body_pos    = np.array([t * 0.03, 0.0])
    state.swing_side  = int(cyc >= 1.0)
    state.swing_t     = progress
    state.speed_scale = 0.8

    step = 0.06
    fwd  = np.array([math.cos(state.heading), math.sin(state.heading)])

    def foot(side, phase_offset):
        sign = -1 if side == 0 else 1
        perp = np.array([-math.sin(state.heading), math.cos(state.heading)])
        xy   = state.body_pos + sign * HIP_WIDTH * perp
        z    = STAND_HEIGHT
        if state.swing_side == side:
            xy = xy + fwd * step * progress
            z  = STAND_HEIGHT + 0.05 * math.sin(math.pi * progress)
        return np.array([xy[0], xy[1], z])

    state.foot_world = [foot(0, 0), foot(1, 0.5)]

    # Fake control points for swing leg
    s   = state.foot_world[state.swing_side] - np.array([0, 0, 0.05*progress])
    end = s + np.array([step, 0, 0])
    from gait_dummy import make_swing_curve_demo
    # inline it:
    mid = 0.5*(s+end)
    h   = np.array([0,0,0.05])
    state.ctrl_pts = np.array([
        s,
        s + (end-s)*0.15 + h*0.2,
        s + (end-s)*0.35 + h*0.9,
        mid + h,
        s + (end-s)*0.70 + h*0.6,
        end,
    ])

# ─── Build geometry for one frame ─────────────────────────────────────────────

def build_skeleton(snap: dict):
    """
    Returns a dict of named line segments to draw, each a list of (x,y,z) tuples.
    All in world coordinates.
    """
    heading   = snap["heading"]
    body_pos  = snap["body_pos"]
    foot_w    = snap["foot_world"]

    # Hip joint world positions (z=0 plane = hip height reference)
    def hip_world(side):
        sign = -1 if side == 0 else 1
        perp = np.array([-math.sin(heading), math.cos(heading)])
        xy   = body_pos + sign * HIP_WIDTH * perp
        return np.array([xy[0], xy[1], 0.0])

    hips  = [hip_world(0), hip_world(1)]
    pelvis_center = 0.5*(hips[0]+hips[1])

    # Torso / pelvis block
    torso_top = pelvis_center + np.array([0, 0, PELVIS_H + TORSO_H])
    head_pos  = torso_top + np.array([0, 0, 0.04])

    # Shoulders (slight width)
    sh_w = 0.06
    perp = np.array([-math.sin(heading), math.cos(heading), 0.0])
    sh_l = torso_top - perp * sh_w
    sh_r = torso_top + perp * sh_w

    segs = {}

    # Torso
    segs["spine"]     = [pelvis_center, torso_top]
    segs["shoulders"] = [sh_l, sh_r]
    segs["head"]      = [torso_top, head_pos]
    segs["pelvis"]    = [hips[0], hips[1]]

    # Arms (static hang)
    arm_end_l = sh_l + np.array([0, 0, -0.12])
    arm_end_r = sh_r + np.array([0, 0, -0.12])
    segs["arm_l"] = [sh_l, arm_end_l]
    segs["arm_r"] = [sh_r, arm_end_r]

    # Legs via IK
    colors = {0: C_LEFT, 1: C_RIGHT}
    for side in (0, 1):
        hip  = hips[side]
        foot = foot_w[side]
        rel  = foot - hip

        # IK returns local coords relative to hip
        hx, hz, kx, kz, fx, fz = solve_ik(rel)

        # Reconstruct 3D knee position
        fwd  = np.array([math.cos(heading), math.sin(heading)])
        # Sagittal plane: local x = forward, local z = up
        knee_world = hip + np.array([
            fwd[0]*kx,
            fwd[1]*kx,
            kz
        ])
        segs[f"thigh_{side}"]  = [hip, knee_world]
        segs[f"shank_{side}"]  = [knee_world, foot]
        # Foot platform (small line perpendicular to forward)
        foot_tip  = foot + np.array([fwd[0], fwd[1], 0.0]) * 0.025
        foot_heel = foot - np.array([fwd[0], fwd[1], 0.0]) * 0.025
        segs[f"foot_{side}"]   = [foot_heel, foot_tip]

    # Heading arrow
    arrow_tip = pelvis_center + np.array([
        math.cos(heading)*0.12,
        math.sin(heading)*0.12,
        0.0
    ])
    segs["heading_arrow"] = [pelvis_center, arrow_tip]

    return segs, hips, pelvis_center

# ─── Visualizer ───────────────────────────────────────────────────────────────

class Visualizer:

    # Camera is fixed looking from the side (azimuth locked)
    CAM_ELEV  = 18
    CAM_AZIM  = -60   # degrees — side+front view

    # View window half-size around robot
    VIEW_R    = 0.55

    def __init__(self, state: RobotState, demo: bool = False):
        self.state  = state
        self.demo   = demo
        self.frame  = 0

        self.fig = plt.figure(figsize=(10, 7), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("Gait Visualizer")

        # Main 3D axis
        self.ax = self.fig.add_axes([0.02, 0.12, 0.72, 0.86],
                                     projection="3d")
        self.ax.set_facecolor("#1a1a2e")
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        for spine in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            spine.pane.set_edgecolor("#2d3436")

        # Bezier inset (2D side view of current swing curve)
        self.ax_bez = self.fig.add_axes([0.76, 0.45, 0.22, 0.40],
                                         facecolor="#12121f")

        # Info text panel
        self.ax_info = self.fig.add_axes([0.76, 0.05, 0.22, 0.35],
                                          facecolor="#12121f")
        self.ax_info.axis("off")

        self._init_artists()

        self.anim = FuncAnimation(
            self.fig, self._update,
            interval=40,   # ~25fps — smooth enough, not heavy
            blit=False,
            cache_frame_data=False,
        )

    def _init_artists(self):
        ax = self.ax

        # Ground grid
        self._ground_drawn = False

        # Line collections for skeleton
        line_kw = dict(linewidth=2.5, solid_capstyle="round")
        self.lines = {
            #"spine":          ax.plot([], [], [], color=C_BODY,  lw=3)[0],
            #"shoulders":      ax.plot([], [], [], color=C_BODY,  lw=3)[0],
            #"head":           ax.plot([], [], [], color=C_BODY,  lw=5)[0],
            "pelvis":         ax.plot([], [], [], color=C_BODY,  lw=3)[0],
            #"arm_l":          ax.plot([], [], [], color=C_LEFT,  lw=2)[0],
            #"arm_r":          ax.plot([], [], [], color=C_RIGHT, lw=2)[0],
            "thigh_0":        ax.plot([], [], [], color=C_LEFT,  lw=4)[0],
            "shank_0":        ax.plot([], [], [], color=C_LEFT,  lw=3)[0],
            "foot_0":         ax.plot([], [], [], color=C_LEFT,  lw=4)[0],
            "thigh_1":        ax.plot([], [], [], color=C_RIGHT, lw=4)[0],
            "shank_1":        ax.plot([], [], [], color=C_RIGHT, lw=3)[0],
            "foot_1":         ax.plot([], [], [], color=C_RIGHT, lw=4)[0],
            "heading_arrow":  ax.plot([], [], [], color=C_HEADING, lw=2,
                                      linestyle="--")[0],
        }
        # Bezier arc (swing foot path)
        self.bez_line,  = ax.plot([], [], [], color=C_BEZIER, lw=1.5,
                                   linestyle=":", alpha=0.8)
        self.bez_ctrl,  = ax.plot([], [], [], color=C_CP, lw=0.8,
                                   linestyle="--", alpha=0.5)
        self.bez_dots   = ax.scatter([0],[0],[0], color=C_CP, s=12, alpha=0)

        # Foot markers — initialise with dummy point, hide via alpha
        self.foot_marks = [
            ax.scatter([0], [0], [0], color=C_LEFT,  s=40, zorder=5, alpha=0),
            ax.scatter([0], [0], [0], color=C_RIGHT, s=40, zorder=5, alpha=0),
        ]

        # Inset Bezier panel artists
        self.bez_inset_curve, = self.ax_bez.plot([], [], color=C_BEZIER, lw=2)
        self.bez_inset_ctrl,  = self.ax_bez.plot([], [], color=C_CP, lw=1,
                                                   linestyle="--", marker="o",
                                                   markersize=5)
        self.bez_inset_dot,   = self.ax_bez.plot([], [], "o", color=C_SWING,
                                                   markersize=8)
        self.ax_bez.set_title("Swing Bezier", color="#aaaaaa",
                               fontsize=8, pad=3)
        self.ax_bez.tick_params(colors="#555555", labelsize=7)
        for sp in self.ax_bez.spines.values():
            sp.set_edgecolor("#333344")

        # Info text
        self.info_text = self.ax_info.text(
            0.05, 0.95, "", transform=self.ax_info.transAxes,
            color="#dfe6e9", fontsize=9, va="top", fontfamily="monospace"
        )

    def _draw_ground(self, cx, cy):
        """Draw a ground grid centred on robot position."""
        ax = self.ax
        z  = STAND_HEIGHT - 0.005
        r  = self.VIEW_R * 1.2
        spacing = 0.10
        xs = np.arange(cx - r, cx + r + spacing, spacing)
        ys = np.arange(cy - r, cy + r + spacing, spacing)
        kw = dict(color="#2d3436", lw=0.5, alpha=0.6)
        for x in xs:
            ax.plot([x, x], [cy-r, cy+r], [z, z], **kw)
        for y in ys:
            ax.plot([cx-r, cx+r], [y, y], [z, z], **kw)

    def _update(self, frame):
        self.frame = frame

        # Demo tick
        if self.demo:
            _demo_tick(self.state, frame)

        snap = self.state.snapshot()
        segs, hips, pelvis = build_skeleton(snap)
        bp   = snap["body_pos"]
        ctrl = snap["ctrl_pts"]

        # ── Update skeleton lines ────────────────────────────────────────────
        for name, line in self.lines.items():
            if name not in segs:
                continue
            pts = segs[name]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            line.set_data_3d(xs, ys, zs)

        # ── Foot markers ─────────────────────────────────────────────────────
        for side in (0, 1):
            f = snap["foot_world"][side]
            self.foot_marks[side]._offsets3d = ([f[0]], [f[1]], [f[2]])
            self.foot_marks[side].set_alpha(1)

        # ── Bezier arc in 3D ─────────────────────────────────────────────────
        if ctrl is not None:
            curve = bezier_pts(ctrl, 60)
            self.bez_line.set_data_3d(curve[:,0], curve[:,1], curve[:,2])
            self.bez_ctrl.set_data_3d(ctrl[:,0],  ctrl[:,1],  ctrl[:,2])
            self.bez_dots._offsets3d = (ctrl[:,0], ctrl[:,1], ctrl[:,2])
            self.bez_dots.set_alpha(0.6)

            # Current foot position on curve
            t   = max(0.0, min(1.0, snap["swing_t"]))
            pos = bezier_pts(ctrl, 100)[int(t*99)]
            # (shown via foot_marks already)
        else:
            self.bez_line.set_data_3d([], [], [])
            self.bez_ctrl.set_data_3d([], [], [])
            self.bez_dots._offsets3d = ([0],[0],[0])
            self.bez_dots.set_alpha(0)

        # ── Bezier inset (x = forward progress, z = height) ─────────────────
        if ctrl is not None:
            curve = bezier_pts(ctrl, 80)
            # Project onto sagittal: use world-x as proxy for forward
            self.bez_inset_curve.set_data(curve[:,0], curve[:,2])
            self.bez_inset_ctrl.set_data(ctrl[:,0],   ctrl[:,2])
            t   = max(0.0, min(1.0, snap["swing_t"]))
            pos = bezier_pts(ctrl, 100)[int(t*99)]
            self.bez_inset_dot.set_data([pos[0]], [pos[2]])
            self.ax_bez.relim()
            self.ax_bez.autoscale_view()
        else:
            self.bez_inset_curve.set_data([], [])
            self.bez_inset_ctrl.set_data([], [])
            self.bez_inset_dot.set_data([], [])

        # ── Info panel ───────────────────────────────────────────────────────
        hdeg = math.degrees(snap["heading"])
        side_name = "LEFT" if snap["swing_side"] == 0 else "RIGHT"
        info = (
            f"state:   {snap['gait_state']}\n"
            f"heading: {hdeg:+.1f}°\n"
            f"speed:   {snap['speed_scale']:.2f}\n"
            f"swing:   {side_name}\n"
            f"swing_t: {snap['swing_t']:.2f}\n"
            f"body: ({bp[0]:.2f},{bp[1]:.2f})"
        )
        self.info_text.set_text(info)

        # ── Camera — fixed azimuth, tracks robot ─────────────────────────────
        ax = self.ax
        cx, cy = float(bp[0]), float(bp[1])
        r = self.VIEW_R

        # Redraw ground each frame (cheap, follows robot)
        # Clear old ground by removing children that aren't our named artists
        # Easiest: just set axis limits — no dynamic ground redraws needed for perf
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(STAND_HEIGHT - 0.02, TORSO_H + PELVIS_H + 0.08)
        ax.set_box_aspect([1, 1, 0.9])

        ax.view_init(elev=self.CAM_ELEV, azim=self.CAM_AZIM)
        ax.set_xlabel("forward", color="#555566", fontsize=7, labelpad=1)
        ax.set_ylabel("lateral", color="#555566", fontsize=7, labelpad=1)
        ax.set_zlabel("height",  color="#555566", fontsize=7, labelpad=1)
        ax.tick_params(colors="#333344", labelsize=6)

        # Draw ground plane
        self._draw_ground_plane(cx, cy)

        return []

    def _draw_ground_plane(self, cx, cy):
        """Thin ground plane quad — recreated each frame to follow robot."""
        ax = self.ax
        z  = STAND_HEIGHT - 0.005
        r  = self.VIEW_R
        if hasattr(self, "_ground_surf") and self._ground_surf is not None:
            try:
                self._ground_surf.remove()
            except Exception:
                pass
            self._ground_surf = None
        X = np.array([[cx-r, cx+r], [cx-r, cx+r]])
        Y = np.array([[cy-r, cy-r], [cy+r, cy+r]])
        Z = np.full_like(X, z)
        self._ground_surf = ax.plot_surface(
            X, Y, Z, color="#1e272e", alpha=0.4, linewidth=0
        )

    def show(self):
        plt.show()

# ─── Demo mode: fake walk cycle without YARP ─────────────────────────────────

def _demo_tick(state: RobotState, frame: int):
    t   = frame * 0.04
    cyc = (t * 0.8) % 2.0

    state.gait_state  = "STEPPING"
    state.heading     = math.sin(t * 0.25) * 0.5
    state.body_pos    = np.array([t * 0.025, math.sin(t * 0.25) * 0.05])
    state.swing_side  = int(cyc >= 1.0)
    state.swing_t     = cyc % 1.0
    state.speed_scale = 1.0

    step = 0.06
    heading = state.heading
    fwd  = np.array([math.cos(heading), math.sin(heading)])
    perp = np.array([-math.sin(heading), math.cos(heading)])

    def foot(side):
        sign = -1 if side == 0 else 1
        xy   = state.body_pos + sign * HIP_WIDTH * perp
        z    = STAND_HEIGHT
        if state.swing_side == side:
            p = state.swing_t
            xy = xy + fwd * step * p
            z  = STAND_HEIGHT + 0.05 * math.sin(math.pi * p)
        return np.array([xy[0], xy[1], z])

    state.foot_world = [foot(0), foot(1)]

    # Swing Bezier
    s   = state.foot_world[state.swing_side].copy()
    e   = s.copy()
    e[0] += fwd[0] * step * (1 - state.swing_t)
    e[1] += fwd[1] * step * (1 - state.swing_t)
    mid  = 0.5*(s + e)
    h    = np.array([0, 0, 0.05])
    state.ctrl_pts = np.array([
        s,
        s + (e-s)*0.15 + h*0.2,
        s + (e-s)*0.35 + h*0.9,
        mid + h,
        s + (e-s)*0.70 + h*0.6,
        e,
    ])

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Run without YARP (fake walk cycle)")
    args = parser.parse_args()

    state    = RobotState()
    listener = None

    if not args.demo:
        yarp.Network.init()
        assert yarp.Network.checkNetwork(3.0), "yarpserver not reachable"
        listener = VizListener(state)
        listener.start()
    else:
        print("[viz] Demo mode — no YARP needed")

    viz = Visualizer(state, demo=args.demo)

    try:
        viz.show()
    finally:
        if listener:
            listener.stop()

if __name__ == "__main__":
    main()