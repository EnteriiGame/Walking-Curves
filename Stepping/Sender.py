#!/usr/bin/env python3
"""
sender.py — keyboard -> YARP port

Controls:
  w  → "forward"   (HOLD — walks while held, stops on release)
  s  → "backward"  (HOLD — walks backward while held, stops on release)
  a  → "left"      (TAP  — each tap queues one turn step)
  d  → "right"     (TAP  — each tap queues one turn step)
  q  → quit

How turns work:
  - Each tap of a/d sends one "left"/"right" message.
  - gait.py counts them and executes one pivot step per message.
  - You can tap multiple times quickly to queue several turns.
  - After the turn(s) finish, gait resumes walking (if w held) or idle.
"""

import sys, os, tty, termios, time
import yarp

PORT_OUT = "/debug/sender"
TARGETS  = ["/debug/receiver", "/robot/cmd"]

RETRY_INTERVAL = 1.0   # seconds between reconnect attempts
TICK            = 0.02  # seconds per loop (50 Hz)

# ── terminal helpers ──────────────────────────────────────────────────────────

def set_raw(fd):
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old

def restore(fd, old):
    termios.tcsetattr(fd, termios.TCSADRAIN, old)

def read_key_nonblock(fd):
    """Return one byte if a key is available, else None."""
    os.set_blocking(fd, False)
    try:
        return os.read(fd, 1)
    except BlockingIOError:
        return None
    finally:
        os.set_blocking(fd, True)

# ── YARP helpers ──────────────────────────────────────────────────────────────

def send(port, msg: str):
    b = yarp.Bottle()
    b.addString(msg)
    port.write(b)

def try_connect_all(connected: dict):
    for target in TARGETS:
        if connected[target]:
            continue
        if yarp.Network.connect(PORT_OUT, target):
            connected[target] = True
            print(f"\n[sender] connected -> {target}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    yarp.Network.init()
    if not yarp.Network.checkNetwork(3.0):
        print("[sender] ERROR: yarpserver not reachable.")
        sys.exit(1)

    port = yarp.Port()
    port.open(PORT_OUT)

    print(f"[sender] port open: {PORT_OUT}")
    print(f"[sender] targeting: {TARGETS}")
    print()
    print("  HOLD  w / s   →  walk forward / backward")
    print("  TAP   a / d   →  turn left / right  (tap multiple times to queue turns)")
    print("  q             →  quit")
    print()
    print("[sender] messages sent immediately — connection retried in background")

    fd = sys.stdin.fileno()
    old_term = set_raw(fd)

    connected  = {t: False for t in TARGETS}
    last_retry = 0.0

    # keys that are HOLD-type (send on press, send "stop" on release)
    HOLD_KEYS = {b"w": "forward", b"s": "backward"}

    # keys that are TAP-type (send once on press, no release action)
    TAP_KEYS  = {b"a": "left",    b"d": "right"}

    held_key   = None   # currently physically held hold-key
    last_cmd   = None   # last thing we sent (to avoid repeating "stop")

    try:
        while True:
            now = time.monotonic()

            # ── connection retry ─────────────────────────────────────────────
            if now - last_retry > RETRY_INTERVAL:
                try_connect_all(connected)
                last_retry = now

            # ── read key ─────────────────────────────────────────────────────
            key = read_key_nonblock(fd)

            if key in (b"q", b"\x03"):          # q or Ctrl-C
                send(port, "stop")
                print(f"\n[sender] quit")
                break

            elif key in HOLD_KEYS:
                # New hold key pressed (or same one repeated — ignore)
                if key != held_key:
                    held_key = key
                    cmd      = HOLD_KEYS[key]
                    last_cmd = cmd
                    send(port, cmd)
                    _status(cmd, connected)

            elif key in TAP_KEYS:
                # Tap: send once, no state change
                cmd = TAP_KEYS[key]
                send(port, cmd)
                _status(f"{cmd} (tap)", connected)
                # Note: held_key is NOT changed — if w was held we keep walking
                # between/after turns.

            elif key is not None:
                # Any other key = release — treat as "stop hold"
                if held_key is not None:
                    held_key = None
                if last_cmd != "stop":
                    last_cmd = "stop"
                    send(port, "stop")
                    _status("stop", connected)

            else:
                # No key this tick
                if held_key is None and last_cmd != "stop":
                    # Nothing held and we haven't sent stop yet
                    last_cmd = "stop"
                    send(port, "stop")
                    _status("stop", connected)

            time.sleep(TICK)

    finally:
        restore(fd, old_term)
        port.close()
        yarp.Network.fini()
        print("\n[sender] bye.")


def _status(cmd: str, connected: dict):
    names = [t.split("/")[-1] for t, v in connected.items() if v]
    sys.stdout.write(f"\r[sender] -> {cmd:<20}  connected:{names}   ")
    sys.stdout.flush()


if __name__ == "__main__":
    main()