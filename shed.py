"""Nira Shed Helper - touchless dashboard. No pose classification, no PC keys.
Presence lock (hand exists = cursor on) + index fingertip drives everything:
  boxes (dwell 0.6s to toggle) + slider (tip x sets value) + counter.
uv run python shed.py [--test]
"""
import argparse
import sys
import time
import cv2
import numpy as np

MODEL = "hand_landmarker.task"


def _xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h])


class FakeLm:
    def __init__(self, x, y):
        self.x, self.y = x, y


def fingers_up(lms):
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    return sum(1 for t, p in zip(tips, pips) if lms[t].y < lms[p].y)


def is_palm(lms):
    return fingers_up(lms) >= 4


def is_fist(lms):
    return fingers_up(lms) == 0


def pinch_px(lms, w, h):
    a, b = _xy(lms[4], w, h), _xy(lms[8], w, h)
    return float(np.linalg.norm(a - b))


class FistPause:
    """Fist = kill-switch. Hold fist ~12 frames to freeze/unfreeze everything.
    Pointing (index up) never counts, so normal use can't trip it."""

    NEED = 12

    def __init__(self):
        self.n = 0
        self.paused = False
        self.armed = True

    def update(self, is_fist, index_up=False):
        if not is_fist or index_up:
            self.n = 0
            self.armed = True
            return None
        self.n += 1
        if self.n >= self.NEED and self.armed:
            self.armed = False
            self.paused = not self.paused
            return "paused" if self.paused else "resumed"
        return None

    def reset(self):
        self.n = 0
        self.armed = True


class PresenceLock:
    """Palm lock-in: needs N consecutive frames with hand to lock, M missing to drop."""

    def __init__(self, need=3, lose=5):
        self.need, self.lose = need, lose
        self.hit, self.miss, self.locked = 0, 0, False

    def update(self, present):
        if present:
            self.hit, self.miss = self.hit + 1, 0
            if self.hit >= self.need:
                self.locked = True
        else:
            self.miss, self.hit = self.miss + 1, 0
            if self.miss >= self.lose:
                self.locked = False
        return self.locked

    def reset(self):
        self.hit, self.miss, self.locked = 0, 0, False


class DwellButton:
    DWELL_S = 0.6
    DWELL_MAP = {"freeze": 0.8}  # freeze needs a deliberate hold
    REFIRE_S = 2.0  # same box can't refire within 2s (edge-jitter spam)

    def __init__(self):
        self.cur = None
        self.enter = 0.0
        self.armed = True
        self.cool = {}
        self.need = self.DWELL_S

    def update(self, tip, buttons_px, now=None):
        now = time.time() if now is None else now
        hit = None
        for (bid, _label, (x0, y0, x1, y1)) in buttons_px:
            if x0 <= tip[0] <= x1 and y0 <= tip[1] <= y1:
                hit = bid
                break
        if hit != self.cur:
            self.cur, self.enter, self.armed = hit, now, True
            self.need = self.DWELL_MAP.get(hit, self.DWELL_S)
            return None
        if hit is None or not self.armed:
            return None
        if now < self.cool.get(hit, 0):
            return None
        if now - self.enter >= self.need:
            self.armed = False
            self.cool[hit] = now + self.REFIRE_S
            return hit
        return None

    def progress(self, now=None):
        now = time.time() if now is None else now
        if self.cur is None:
            return 0.0
        return (now - self.enter) / self.need

    def reset(self):
        self.cur, self.armed = None, True


class Slider:
    """Tip x inside strip sets 0-100 with smoothing + end snap. No grab needed."""

    SNAP = 8.0

    def __init__(self, alpha=0.25):
        self.value = 0.0
        self.alpha = alpha

    def update(self, tip, rect):
        x0, y0, x1, y1 = rect
        if not (x0 <= tip[0] <= x1 and y0 <= tip[1] <= y1):
            return None
        target = 100.0 * (tip[0] - x0) / max(x1 - x0, 1)
        if target < self.SNAP:
            target = 0.0
        elif target > 100.0 - self.SNAP:
            target = 100.0
        self.value += self.alpha * (target - self.value)
        if abs(target - self.value) < 0.6:
            self.value = target
        return self.value

    def set(self, v):
        self.value = float(max(0, min(100, v)))
        return self.value


class CursorSmoother:
    """Smooth helper: EMA on fingertip so cursor doesn't shake."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.p = None

    def update(self, tip):
        if self.p is None:
            self.p = (float(tip[0]), float(tip[1]))
        else:
            self.p = (self.p[0] + self.alpha * (tip[0] - self.p[0]),
                      self.p[1] + self.alpha * (tip[1] - self.p[1]))
        return self.p

    def reset(self):
        self.p = None


class Board:
    def __init__(self):
        self.motor = False
        self.light = False
        self.sprinkler = False
        self.doors = {d: False for d in DOORS}
        self.sel = 0
        self.alarm = False
        self.frozen = False
        self.speed = 0.0

    @property
    def door_name(self):
        return DOORS[self.sel % len(DOORS)]

    @property
    def door_open(self):
        return self.doors[self.door_name]

    def toggle(self, bid):
        if bid == "motor":
            self.motor = not self.motor
        elif bid == "sprinkler":
            self.sprinkler = not self.sprinkler
        elif bid == "light":
            self.light = not self.light
        elif bid == "doorsel":
            self.sel = (self.sel + 1) % len(DOORS)
            return f"DOOR: {self.door_name} {'OPEN' if self.door_open else 'SHUT'}"
        elif bid == "open":
            if self.door_open:
                return f"{self.door_name} OPENED ALREADY"
            self.doors[self.door_name] = True
            return f"{self.door_name} OPEN"
        elif bid == "close":
            if not self.door_open:
                return f"{self.door_name} CLOSED ALREADY"
            self.doors[self.door_name] = False
            return f"{self.door_name} CLOSED"
        elif bid == "alarm":
            self.alarm = not self.alarm
        elif bid == "freeze":
            self.frozen = not self.frozen
            return "FROZEN" if self.frozen else "RESUMED"
        return f"{bid.upper()} {'ON' if getattr(self, bid, None) is True else ''}".strip()


BUTTONS = [
    ("motor", "MOTOR", (0.02, 0.13, 0.24, 0.20)),
    ("light", "LIGHT", (0.26, 0.13, 0.48, 0.20)),
    ("alarm", "ALARM", (0.50, 0.13, 0.72, 0.20)),
    ("freeze", "FREEZE", (0.74, 0.13, 0.95, 0.20)),
    ("doorsel", "DOOR", (0.02, 0.22, 0.24, 0.29)),
    ("open", "OPEN", (0.26, 0.22, 0.48, 0.29)),
    ("close", "CLOSE", (0.50, 0.22, 0.72, 0.29)),
    ("sprinkler", "SPRINKLER", (0.74, 0.22, 0.95, 0.29)),
]
DOORS = ["MAIN", "KITCHEN", "SHED"]
SLIDER_NORM = (0.16, 0.86, 0.74, 0.93)  # SPRINKLER SPEED strip (room for 0/100 sides)
MIN_NORM = (0.02, 0.86, 0.14, 0.93)  # touch = sprinkler 0
MAX_NORM = (0.76, 0.86, 0.93, 0.93)  # touch = sprinkler 100


def fit_label(frame, txt, box, color, thick=2, floor=0.6):
    """Draw text centered in box, shrunk to fit but never below floor."""
    X0, Y0, X1, Y1 = box
    bw, bh = max(X1 - X0 - 16, 10), max(Y1 - Y0 - 12, 10)
    fs = 0.9
    while fs > floor:
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        if tw <= bw and th <= bh:
            break
        fs = round(fs - 0.1, 1)
    fs = max(fs, floor)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    cv2.putText(frame, txt, (X0 + (X1 - X0 - tw) // 2, Y0 + (Y1 - Y0 + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick)
    return fs


def draw(frame, board, lock, buttons=None, dwell=None, slider_rect=None):
    h, w, _ = frame.shape
    state = "FROZEN" if board.frozen else ("LOCKED" if lock else "SHOW PALM")
    scol = (0, 0, 255) if board.frozen else ((0, 255, 0) if lock else (0, 165, 255))
    cv2.rectangle(frame, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.putText(frame, f"{state}   SPD:{board.speed:.0f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, scol, 2)
    cv2.putText(frame, "dwell a box | q=quit", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if buttons:
        overlay = frame.copy()
        for (bid, label, (x0, y0, x1, y1)) in buttons:
            X0, Y0, X1, Y1 = int(x0), int(y0), int(x1), int(y1)  # already pixels
            cv2.rectangle(overlay, (X0, Y0), (X1, Y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        for (bid, label, (x0, y0, x1, y1)) in buttons:
            X0, Y0, X1, Y1 = int(x0), int(y0), int(x1), int(y1)  # already pixels
            on = (board.__dict__.get(bid) is True or (bid == "freeze" and board.frozen)
                  or (bid == "open" and board.door_open) or (bid == "close" and not board.door_open)
                  or (bid == "doorsel" and board.door_open))
            hot = dwell is not None and dwell.cur == bid
            col = (0, 255, 0) if on else ((0, 255, 255) if hot else (255, 255, 255))
            cv2.rectangle(frame, (X0, Y0), (X1, Y1), col, 2)
            if bid in ("motor", "light", "alarm", "sprinkler"):
                txt = f"{label} {'ON' if on else ''}"
            elif bid == "freeze":
                txt = "FROZEN" if board.frozen else "FREEZE"
            elif bid == "doorsel":
                txt = f"{board.door_name}:{'OPEN' if board.door_open else 'SHUT'}"
            else:
                txt = label
            fit_label(frame, txt, (X0, Y0, X1, Y1 - 12), (0, 255, 255))
            if hot and dwell is not None:
                p = min(1.0, dwell.progress())
                cv2.rectangle(frame, (X0, Y1 - 10), (int(X0 + (X1 - X0) * p), Y1 - 4), (0, 255, 0), -1)
    if slider_rect:
        X0, Y0, X1, Y1 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(slider_rect)]
        mid = (Y0 + Y1) // 2
        cv2.rectangle(frame, (X0, mid - 3), (X1, mid + 3), (120, 120, 120), -1)
        fx = int(X0 + (X1 - X0) * board.speed / 100.0)
        cv2.rectangle(frame, (X0, mid - 3), (fx, mid + 3), (255, 150, 0), -1)
        cv2.circle(frame, (fx, mid), 13, (255, 150, 0), -1)
        cv2.circle(frame, (fx, mid), 13, (255, 255, 255), 2)
        for t in (0, 25, 50, 75, 100):
            tx = int(X0 + (X1 - X0) * t / 100.0)
            cv2.line(frame, (tx, Y1 - 2), (tx, Y1 + 5), (160, 160, 160), 1)
        cv2.putText(frame, f"SPRINKLER SPEED {board.speed:.0f}", (X0, Y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
        for (rx, label) in ((MIN_NORM, "0"), (MAX_NORM, "100")):
            BX0, BY0, BX1, BY1 = int(rx[0] * w), int(rx[1] * h), int(rx[2] * w), int(rx[3] * h)
            cv2.rectangle(frame, (BX0, BY0), (BX1, BY1), (40, 40, 40), -1)
            cv2.rectangle(frame, (BX0, BY0), (BX1, BY1), (255, 150, 0), 2)
            fit_label(frame, label, (BX0, BY0, BX1, BY1), (255, 150, 0))
    return frame


def fake_hand(pose):
    lms = [FakeLm(0.5, 0.7) for _ in range(21)]
    lms[0] = FakeLm(0.5, 0.75)
    if pose == "palm":
        for t, p, x in [(8, 6, 0.45), (12, 10, 0.5), (16, 14, 0.55), (20, 18, 0.6)]:
            lms[t] = FakeLm(x, 0.3)
            lms[p] = FakeLm(x, 0.5)
        lms[4] = FakeLm(0.35, 0.5)
    elif pose == "fist":
        for t, p, x in [(8, 6, 0.45), (12, 10, 0.5), (16, 14, 0.55), (20, 18, 0.6)]:
            lms[t] = FakeLm(x, 0.65)
            lms[p] = FakeLm(x, 0.5)
    return lms


def selftest():
    palm, fist = fake_hand("palm"), fake_hand("fist")
    assert is_palm(palm) and is_fist(fist)
    assert fingers_up(palm) == 4 and fingers_up(fist) == 0
    pl = PresenceLock(need=3, lose=5)
    assert [pl.update(False) for _ in range(6)] == [False] * 6
    assert [pl.update(True) for _ in range(2)] == [False] * 2
    assert pl.update(True) is True, "3 consecutive lock"
    assert pl.update(False) is True, "1 miss keeps lock"
    for _ in range(4):
        pl.update(False)
    assert pl.update(False) is False, "5 missing drops"
    db = DwellButton()
    box = [("motor", "MOTOR", (100, 100, 300, 300))]
    t0 = time.time()
    assert db.update((10, 10), box, now=t0) is None
    for i in range(4):
        assert db.update((200, 200), box, now=t0 + 0.1 * i) is None
    assert db.update((200, 200), box, now=t0 + 0.7) == "motor"
    assert db.update((200, 200), box, now=t0 + 1.2) is None
    assert db.update((10, 10), box, now=t0 + 1.3) is None
    assert db.update((200, 200), box, now=t0 + 1.4) is None
    assert db.update((200, 200), box, now=t0 + 2.1) is None, "2s refire cooldown blocks jitter"
    assert db.update((200, 200), box, now=t0 + 3.0) == "motor", "fires after cooldown"
    sl = Slider()
    assert sl.update((0, 0), (100, 100, 500, 200)) is None
    v = sl.update((300, 150), (100, 100, 500, 200))
    assert v is not None and 0 < v < 100, f"slider follows tip {v}"
    for _ in range(30):
        v = sl.update((500, 150), (100, 100, 500, 200))
    assert v == 100, f"end snaps exactly 100, got {v}"
    for _ in range(30):
        v = sl.update((100, 150), (100, 100, 500, 200))
    assert v == 0, f"end snaps exactly 0, got {v}"
    assert sl.set(150) == 100 and sl.set(-5) == 0, "MIN/MAX clamp"
    cs = CursorSmoother()
    p1 = cs.update((100, 100))
    p2 = cs.update((200, 100))
    assert p1 == (100.0, 100.0) and 100 < p2[0] < 200, f"smoother halves jump {p2}"
    cs.reset()
    assert cs.update((50, 50)) == (50.0, 50.0)
    b = Board()
    assert b.toggle("motor") == "MOTOR ON" and b.motor
    assert b.toggle("motor") == "MOTOR" and not b.motor
    assert b.toggle("open") == "MAIN OPEN" and b.door_open
    assert b.toggle("close") == "MAIN CLOSED" and not b.door_open
    assert b.toggle("alarm") == "ALARM ON" and b.alarm
    assert b.toggle("doorsel") == "DOOR: KITCHEN SHUT" and b.door_name == "KITCHEN"
    assert b.toggle("open") == "KITCHEN OPEN" and b.doors["KITCHEN"]
    assert b.toggle("doorsel") == "DOOR: SHED SHUT"
    assert b.toggle("doorsel") == "DOOR: MAIN SHUT", "cycles MAIN->KITCHEN->SHED->MAIN"
    assert b.doors["KITCHEN"] and not b.doors["MAIN"], "per-door state kept"
    fp = FistPause()
    for _ in range(11):
        assert fp.update(True) is None, "needs 12 steady fist frames"
    assert fp.update(True) == "paused" and fp.paused
    assert fp.update(True) is None, "holding fist doesn't re-toggle"
    assert fp.update(False) is None, "release re-arms"
    for _ in range(11):
        assert fp.update(True) is None
    assert fp.update(True) == "resumed" and not fp.paused
    fp2 = FistPause()
    for _ in range(20):
        assert fp2.update(True, index_up=True) is None, "pointing never freezes"
    assert not fp2.paused
    b2 = Board()
    assert b2.toggle("freeze") == "FROZEN" and b2.frozen
    assert b2.toggle("freeze") == "RESUMED" and not b2.frozen
    assert b2.toggle("sprinkler") == "SPRINKLER ON" and b2.sprinkler
    assert b2.toggle("sprinkler") == "SPRINKLER" and not b2.sprinkler
    b3 = Board()
    assert b3.toggle("close") == "MAIN CLOSED ALREADY" and not b3.door_open
    assert b3.toggle("open") == "MAIN OPEN" and b3.door_open
    assert b3.toggle("open") == "MAIN OPENED ALREADY" and b3.door_open
    assert b3.toggle("close") == "MAIN CLOSED" and not b3.door_open
    assert b3.toggle("close") == "MAIN CLOSED ALREADY" and not b3.door_open
    db3 = DwellButton()
    t2 = time.time()
    box_f = [("freeze", "FREEZE", (100, 100, 300, 300))]
    db3.update((200, 200), box_f, now=t2)
    assert db3.update((200, 200), box_f, now=t2 + 0.7) is None, "freeze needs 0.8s hold"
    assert db3.update((200, 200), box_f, now=t2 + 0.9) == "freeze"
    print("boxes self-test passed: presence/dwell/slider/toggles/fistpause")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--cam", type=int, default=0, help="camera index (try 1, 2 if 0 glitches)")
    ap.add_argument("--res", default="640x480", help="capture resolution WxH (default 640x480, safer than 720p)")
    ap.add_argument("--skip", type=int, default=2, help="detect every Nth frame (perf)")
    args = ap.parse_args()
    if args.test:
        selftest()
        return
    from mediapipe.tasks.python import BaseOptions, vision
    import mediapipe as mp
    import os
    if not os.path.exists(MODEL):
        sys.exit("missing hand_landmarker.task")
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4)
    cap = cv2.VideoCapture(args.cam)
    try:
        rw, rh = args.res.lower().split("x")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(rw))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(rh))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        sys.exit(f"No webcam {args.cam}. Try: uv run python shed.py --cam 1")
    ok, probe = cap.read()
    if ok:
        ph, pw, _ = probe.shape
        print(f"camera {args.cam} running at {pw}x{ph}")
    else:
        print(f"camera {args.cam} opened but no frames (try --cam 2)")
    board, lock, dwell, slider, smooth = Board(), PresenceLock(), DwellButton(), Slider(), CursorSmoother()
    print("farm-shed-helper. SHOW PALM to lock, point fingertip at boxes, slider follows tip. q=quit")
    cv2.namedWindow("farm-shed-helper (q quit)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("farm-shed-helper (q quit)", 1280, 720)
    frame_no, cached, fps, t_prev = 0, None, 0.0, time.time()
    with vision.HandLandmarker.create_from_options(opts) as lm:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            if frame.shape[0] < 100 or frame.shape[1] < 100:
                continue  # corrupt partial JPEG frame
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_no += 1
            now = time.time()
            fps += 0.2 * (1.0 / max(now - t_prev, 1e-3) - fps)
            t_prev = now
            if frame_no % max(args.skip, 1) == 1 or cached is None:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cached = lm.detect_for_video(mp_img, int(now * 1000))
            res = cached
            cv2.putText(frame, f"{fps:.0f}fps cam{args.cam}", (w - 170, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            hands = res.hand_landmarks or []
            locked = lock.update(bool(hands))
            msg = None
            bpx = [(b, l, (x0 * w, y0 * h, x1 * w, y1 * h)) for (b, l, (x0, y0, x1, y1)) in BUTTONS]
            srect = (SLIDER_NORM[0] * w, SLIDER_NORM[1] * h, SLIDER_NORM[2] * w, SLIDER_NORM[3] * h)
            minrect = (MIN_NORM[0] * w, MIN_NORM[1] * h, MIN_NORM[2] * w, MIN_NORM[3] * h)
            maxrect = (MAX_NORM[0] * w, MAX_NORM[1] * h, MAX_NORM[2] * w, MAX_NORM[3] * h)
            if hands:
                hand = hands[0]
                pts = [(int(p.x * w), int(p.y * h)) for p in hand]
                for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 0), 2)
                tip = smooth.update((hand[8].x * w, hand[8].y * h))
                cv2.circle(frame, (int(tip[0]), int(tip[1])), 10, (0, 255, 255), -1)
                if board.frozen:
                    cv2.putText(frame, "FROZEN - dwell FREEZE to resume", (w // 2 - 300, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    fired = dwell.update(tip, bpx)
                    if fired == "freeze":
                        msg = board.toggle("freeze")
                elif locked:
                    mx0, my0, mx1, my1 = minrect
                    gx0, gy0, gx1, gy1 = maxrect
                    if mx0 <= tip[0] <= mx1 and my0 <= tip[1] <= my1:
                        if board.speed != 0:
                            board.speed = slider.set(0)
                            msg = "SPRINKLER SPEED 0"
                    elif gx0 <= tip[0] <= gx1 and gy0 <= tip[1] <= gy1:
                        if board.speed != 100:
                            board.speed = slider.set(100)
                            msg = "SPRINKLER SPEED 100"
                    else:
                        v = slider.update(tip, srect)
                        if v is not None:
                            board.speed = v
                    fired = dwell.update(tip, bpx)
                    if fired:
                        msg = board.toggle(fired)
            else:
                dwell.reset()
                smooth.reset()
            if msg:
                print(msg, flush=True)
            cv2.imshow("farm-shed-helper (q quit)", draw(frame, board, locked, bpx, dwell, SLIDER_NORM))
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
