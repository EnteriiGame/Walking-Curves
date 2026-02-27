#!/usr/bin/env python3
"""
ik_solver.py — IK solver for humanoid_beta robot

Uses two minimal per-leg URDF files (leg_right.urdf, leg_left.urdf) that
contain ONLY the joints of each leg. This means:
  - ikpy builds a clean chain with no pelvis or unrelated links
  - active_links_mask is not needed (all revolute joints are active by default)
  - No warnings about fixed links
  - Targets are hip-relative: [x, y, z] relative to GE_27_1 / GE_27_2

Standing target ≈ [0, 0, -0.197]  (straight below hip, full leg extension)

Right leg: GE_27_1 → Part_1_4_1     6 revolute joints: Revolute_6..1
Left  leg: GE_27_2 → StopaLewa_v6_1  5 revolute joints: Revolute_7..11

YARP bottles:
  /gait/right/angles  6 floats (Revolute_6,5,4,3,2,1)  radians
  /gait/left/angles   5 floats (Revolute_7,8,9,10,11)  radians

Install:  pip install ikpy
Generate leg URDFs: already done — leg_right.urdf and leg_left.urdf
Test:     python3 ik_solver.py
"""

import os, math, warnings
import numpy as np

try:
    from ikpy.chain import Chain
except ImportError:
    raise ImportError("ikpy not installed.  Run:  pip install ikpy")

# Joint names in output order (must match joint names in the leg URDF files)
JOINTS_RIGHT = ["Revolute_6", "Revolute_5", "Revolute_4",
                "Revolute_3", "Revolute_2", "Revolute_1"]
JOINTS_LEFT  = ["Revolute_7", "Revolute_8", "Revolute_9",
                "Revolute_10", "Revolute_11"]

# Standing foot position relative to hip attachment link
STAND_TARGET = [0.0, 0.0, -0.197]


class _LegChain:
    """
    One ikpy chain for one leg, loaded from a minimal per-leg URDF.
    The chain starts from the hip attachment link so targets are hip-relative.
    Warm-start reuses the previous solution for speed and stability.
    """

    def __init__(self, urdf_path, base_link, joint_names):
        self.joint_names = joint_names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.chain = Chain.from_urdf_file(
                urdf_path,
                base_elements=[base_link],
                last_link_vector=[0, 0, 0],
            )

        n = len(self.chain.links)
        rev = [l.name for l in self.chain.links if l.name in joint_names]
        print(f"[ik]   {urdf_path}: {n} links, {len(rev)} revolute: {rev}")

        # Read joint limits for output clamping
        self.limits = {}
        for lnk in self.chain.links:
            if lnk.name in joint_names and hasattr(lnk, "bounds"):
                lo, hi = lnk.bounds
                if lo is not None and hi is not None:
                    self.limits[lnk.name] = (lo, hi)

        # Warm-start vector sized to actual chain length
        self._last = [0.0] * n

    def solve(self, target_xyz):
        """
        target_xyz: [x, y, z] relative to base_link (hip attachment).
        Returns {joint_name: angle_radians}.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            angles = self.chain.inverse_kinematics(
                target_position=list(target_xyz),
                initial_position=self._last,
                orientation_mode=None,
            )
        self._last = list(angles)

        result = {}
        for i, lnk in enumerate(self.chain.links):
            if lnk.name in self.joint_names:
                a = float(angles[i])
                if lnk.name in self.limits:
                    a = max(self.limits[lnk.name][0],
                            min(self.limits[lnk.name][1], a))
                result[lnk.name] = a
        return result

    def reset(self):
        self._last = [0.0] * len(self.chain.links)


class LegIK:
    """
    IK solver for humanoid_beta robot.

    Expects leg_right.urdf and leg_left.urdf in the same folder.
    These are generated automatically by running:
        python3 ik_solver.py --gen-urdfs humanoid_beta.urdf

    Target convention (both legs):
        [x, y, z] relative to the hip attachment link
        x = forward (robot heading direction)
        y = lateral
        z = up  (negative = below hip)
        Standing: approx [0, 0, -0.197]
    """

    def __init__(self, urdf_dir=None):
        if urdf_dir is None:
            urdf_dir = os.path.dirname(os.path.abspath(__file__))

        r_path = os.path.join(urdf_dir, "leg_right.urdf")
        l_path = os.path.join(urdf_dir, "leg_left.urdf")

        for p in (r_path, l_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"[ik] Missing: {p}\n"
                    "Generate with:  python3 ik_solver.py --gen-urdfs humanoid_beta.urdf"
                )

        print(f"[ik] Loading right leg: {r_path}")
        self._right = _LegChain(r_path, "GE_27_1", JOINTS_RIGHT)
        print(f"[ik] Loading left  leg: {l_path}")
        self._left  = _LegChain(l_path, "GE_27_2", JOINTS_LEFT)

        print("[ik] Warming up...")
        self.solve_right(np.array(STAND_TARGET))
        self.solve_left( np.array(STAND_TARGET))
        print("[ik] Ready.")

    def solve_right(self, foot_xyz_hip_rel):
        """foot_xyz relative to GE_27_1 (right hip). Returns {joint: radians}."""
        return self._right.solve(np.asarray(foot_xyz_hip_rel, dtype=float))

    def solve_left(self, foot_xyz_hip_rel):
        """foot_xyz relative to GE_27_2 (left hip). Returns {joint: radians}."""
        return self._left.solve(np.asarray(foot_xyz_hip_rel, dtype=float))

    def solve_both(self, left_xyz, right_xyz):
        """Solve both legs, return flat dict of all joints."""
        r = {}
        r.update(self.solve_right(np.asarray(right_xyz, dtype=float)))
        r.update(self.solve_left( np.asarray(left_xyz,  dtype=float)))
        return r

    def joint_names_right(self): return list(JOINTS_RIGHT)
    def joint_names_left(self):  return list(JOINTS_LEFT)

    def reset(self):
        self._right.reset()
        self._left.reset()


# ── URDF generator ────────────────────────────────────────────────────────────

def generate_leg_urdfs(src_urdf, out_dir):
    """
    Extract per-leg URDF files from the full robot URDF.
    Call this once whenever humanoid_beta.urdf changes.
    """
    import xml.etree.ElementTree as ET
    from collections import deque

    tree = ET.parse(src_urdf)
    root_el = tree.getroot()
    joints_raw = {j.get('name'): j for j in root_el.findall('joint')}

    def get_chain(start, end):
        children = {}
        for jn, j in joints_raw.items():
            p = j.find('parent').get('link')
            if p not in children: children[p] = []
            children[p].append((jn, j.find('child').get('link')))
        q = deque([(start, [])])
        while q:
            link, path = q.popleft()
            if link == end: return path
            for jn, child in children.get(link, []):
                q.append((child, path + [(jn, child)]))
        return []

    def write_urdf(chain, base_link, tip_link, filename):
        link_names = {base_link, tip_link}
        for jn, cl in chain:
            j = joints_raw[jn]
            link_names.add(j.find('parent').get('link'))
            link_names.add(cl)
        lines = ['<?xml version="1.0"?>', '<robot name="leg">']
        for ln in sorted(link_names):
            lines.append(f'  <link name="{ln}"/>')
        for jn, cl in chain:
            j = joints_raw[jn]
            jtype = j.get('type')
            parent = j.find('parent').get('link')
            origin = j.find('origin')
            xyz = origin.get('xyz','0 0 0') if origin is not None else '0 0 0'
            rpy = origin.get('rpy','0 0 0') if origin is not None else '0 0 0'
            axis_el = j.find('axis')
            axis = axis_el.get('xyz','0 0 1') if axis_el is not None else '0 0 1'
            limit_el = j.find('limit')
            lines += [f'  <joint name="{jn}" type="{jtype}">',
                      f'    <parent link="{parent}"/>',
                      f'    <child link="{cl}"/>',
                      f'    <origin xyz="{xyz}" rpy="{rpy}"/>']
            if jtype == 'revolute':
                lines.append(f'    <axis xyz="{axis}"/>')
                # Use ±pi regardless of URDF limits — CAD-exported limits like
                # 261°–458° are raw encoder offsets that put 0.0 outside bounds,
                # crashing the warm-start. Firmware enforces real limits.
                lines.append('    <limit effort="10" velocity="5" lower="-3.14159" upper="3.14159"/>')
            lines.append('  </joint>')
        lines.append('</robot>')
        path = os.path.join(out_dir, filename)
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        rev = sum(1 for jn,_ in chain if joints_raw[jn].get('type')=='revolute')
        print(f"[ik] Generated {path}  ({len(chain)} joints, {rev} revolute)")

    write_urdf(get_chain('GE_27_1','Part_1_4_1'),    'GE_27_1','Part_1_4_1',    'leg_right.urdf')
    write_urdf(get_chain('GE_27_2','StopaLewa_v6_1'),'GE_27_2','StopaLewa_v6_1','leg_left.urdf')


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # python3 ik_solver.py --gen-urdfs humanoid_beta.urdf
    if "--gen-urdfs" in sys.argv:
        idx = sys.argv.index("--gen-urdfs")
        src = sys.argv[idx+1]
        out = os.path.dirname(os.path.abspath(src))
        generate_leg_urdfs(src, out)
        sys.exit(0)

    print()
    ik = LegIK()

    print("\n── Right leg (relative to GE_27_1) ──────────────────────────────────")
    hdr = f"{'pose':<16}" + "".join(f"  {j.replace('Revolute_','R'):<8}" for j in JOINTS_RIGHT)
    print(hdr); print("─"*len(hdr))
    for name, pos in [("stand",     [0.00, 0.00,-0.197]),
                      ("step fwd",  [0.04, 0.00,-0.185]),
                      ("mid swing", [0.02, 0.00,-0.160]),
                      ("deep bend", [0.00, 0.00,-0.150])]:
        a = ik.solve_right(np.array(pos))
        vals = "".join(f"  {math.degrees(a.get(j,0)):>+6.1f}°" for j in JOINTS_RIGHT)
        print(f"{name:<16}{vals}")

    print("\n── Left leg (relative to GE_27_2) ───────────────────────────────────")
    hdr2 = f"{'pose':<16}" + "".join(f"  {j.replace('Revolute_','R'):<8}" for j in JOINTS_LEFT)
    print(hdr2); print("─"*len(hdr2))
    for name, pos in [("stand",     [0.00, 0.00,-0.197]),
                      ("step fwd",  [0.04, 0.00,-0.185]),
                      ("mid swing", [0.02, 0.00,-0.160])]:
        a = ik.solve_left(np.array(pos))
        vals = "".join(f"  {math.degrees(a.get(j,0)):>+6.1f}°" for j in JOINTS_LEFT)
        print(f"{name:<16}{vals}")

    print("\nDone.")