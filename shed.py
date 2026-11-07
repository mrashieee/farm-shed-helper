"""Nira Shed Helper - touchless dashboard. No pose classification, no PC keys.
Presence lock (hand exists = cursor on) + index fingertip drives everything:
  boxes (dwell 0.6s to toggle) + slider (tip x sets value) + counter.
uv run python shed.py [--test]
"""
import argparse
import csv
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

    def __init__(self):
        self.cur = None
        self.enter = 0.0
        self.armed = True

    def update(self, tip, buttons_px, now=None):
        now = time.time() if now is None else now
        hit = None
        for (bid, _label, (x0, y0, x1, y1)) in buttons_px:
            if x0 <= tip[0] <= x1 and y0 <= tip[1] <= y1:
                hit = bid
                break
        if hit != self.cur:
            self.cur, self.enter, self.armed = hit, now, True
            return None
        if hit is None or not self.armed:
            return None
        if now - self.enter >= self.DWELL_S:
            self.armed = False
            return hit
        return None

    def progress(self, now=None):
        now = time.time() if now is None else now
        if self.cur is None:
            return 0.0
        return (now - self.enter) / self.DWELL_S

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
        self.door = False
        self.alarm = False
        self.count = 0
        self.speed = 0.0

    def toggle(self, bid):
        if bid == "motor":
            self.motor = not self.motor
        elif bid == "light":
            self.light = not self.light
        elif bid == "door":
            self.door = not self.door
        elif bid == "open":
            self.door = True
        elif bid == "close":
            self.door = False
        elif bid == "alarm":
            self.alarm = not self.alarm
        elif bid == "count":
            self.count += 1
        elif bid in ("m10", "m1", "p1", "p10"):
            self.count = max(0, self.count + {"m10": -10, "m1": -1, "p1": 1, "p10": 10}[bid])
        elif bid == "reset":
            self.__init__()
            return "RESET"
        self.log(bid)
        if bid in ("m10", "m1", "p1", "p10", "count"):
            return f"COUNT {self.count}"
        if bid == "open":
            return "DOOR OPEN"
        if bid == "close":
            return "DOOR CLOSED"
        return f"{bid.upper()} {'ON' if getattr(self, bid, None) is True else ''}".strip()

    def log(self, what):
        with open("shed_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%H:%M:%S"), what, self.count,
                                    round(self.speed), self.motor, self.light,
                                    self.door, self.alarm])


BUTTONS = [
    ("motor", "MOTOR", (0.05, 0.16, 0.30, 0.24)),
    ("light", "LIGHT", (0.35, 0.16, 0.60, 0.24)),
    ("door", "DOOR", (0.65, 0.16, 0.90, 0.24)),
    ("alarm", "ALARM", (0.05, 0.26, 0.30, 0.34)),
    ("open", "OPEN", (0.35, 0.26, 0.60, 0.34)),
    ("close", "CLOSE", (0.65, 0.26, 0.90, 0.34)),
    ("m10", "-10", (0.02, 0.36, 0.18, 0.43)),
    ("m1", "-1", (0.20, 0.36, 0.36, 0.43)),
    ("reset", "RESET", (0.38, 0.36, 0.54, 0.43)),
    ("p1", "+1", (0.56, 0.36, 0.72, 0.43)),
    ("p10", "+10", (0.74, 0.36, 0.94, 0.43)),
]
SLIDER_NORM = (0.16, 0.86, 0.74, 0.93)  # SPEED strip (room for 0/100 sides)
MIN_NORM = (0.02, 0.86, 0.14, 0.93)  # touch = SPEED 0
MAX_NORM = (0.76, 0.86, 0.93, 0.93)  # touch = SPEED 100


def draw(frame, board, lock, buttons=None, dwell=None, slider_rect=None):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 72), (20, 20, 20), -1)
    cv2.putText(frame, f"{'LOCKED' if lock else 'SHOW PALM'}  "
                       f"M:{int(board.motor)} L:{int(board.light)} D:{int(board.door)} "
                       f"A:{int(board.alarm)} N:{board.count} SPD:{board.speed:.0f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if lock else (0, 165, 255), 2)
    cv2.putText(frame, "dwell box 0.6s | slider follows tip | q=quit",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if buttons:
        overlay = frame.copy()
        for (bid, label, (x0, y0, x1, y1)) in buttons:
            X0, Y0, X1, Y1 = int(x0), int(y0), int(x1), int(y1)  # already pixels
            cv2.rectangle(overlay, (X0, Y0), (X1, Y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        for (bid, label, (x0, y0, x1, y1)) in buttons:
            X0, Y0, X1, Y1 = int(x0), int(y0), int(x1), int(y1)  # already pixels
            on = board.__dict__.get(bid) is True
            hot = dwell is not None and dwell.cur == bid
            col = (0, 255, 0) if on else ((0, 255, 255) if hot else (255, 255, 255))
            cv2.rectangle(frame, (X0, Y0), (X1, Y1), col, 2)
            fs = 0.55 if len(label) <= 3 else 0.7
            cv2.putText(frame, f"{label} {'ON' if on else ''}", (X0 + 6, Y0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 2)
            if hot and dwell is not None:
                p = min(1.0, dwell.progress())
                cv2.rectangle(frame, (X0, Y1 - 10), (int(X0 + (X1 - X0) * p), Y1 - 4), (0, 255, 0), -1)
    if slider_rect:
        X0, Y0, X1, Y1 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(slider_rect)]
        cv2.rectangle(frame, (X0, Y0), (X1, Y1), (90, 90, 90), 2)
        fx = int(X0 + (X1 - X0) * board.speed / 100.0)
        cv2.rectangle(frame, (X0, Y0), (fx, Y1), (255, 150, 0), -1)
        cv2.putText(frame, f"SPEED {board.speed:.0f}", (X0 + 10, Y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
        for (rx, label) in ((MIN_NORM, "0"), (MAX_NORM, "100")):
            BX0, BY0, BX1, BY1 = int(rx[0] * w), int(rx[1] * h), int(rx[2] * w), int(rx[3] * h)
            cv2.rectangle(frame, (BX0, BY0), (BX1, BY1), (255, 150, 0), 2)
            cv2.putText(frame, label, (BX0 + 8, BY0 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 150, 0), 2)
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
    assert db.update((200, 200), box, now=t0 + 2.1) == "motor"
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
    assert b.toggle("open") == "DOOR OPEN" and b.door
    assert b.toggle("close") == "DOOR CLOSED" and not b.door
    assert b.toggle("p10") == "COUNT 10" and b.count == 10
    assert b.toggle("m1") == "COUNT 9" and b.count == 9
    assert b.toggle("m10") == "COUNT 0" and b.count == 0, "clamp at zero"
    print("boxes self-test passed: presence/dwell/slider/toggles")


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
            if not ok:
                continue
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
                if locked:
                    mx0, my0, mx1, my1 = minrect
                    gx0, gy0, gx1, gy1 = maxrect
                    if mx0 <= tip[0] <= mx1 and my0 <= tip[1] <= my1:
                        if board.speed != 0:
                            board.speed = slider.set(0)
                            board.log("speed_0")
                            msg = "SPEED 0"
                    elif gx0 <= tip[0] <= gx1 and gy0 <= tip[1] <= gy1:
                        if board.speed != 100:
                            board.speed = slider.set(100)
                            board.log("speed_100")
                            msg = "SPEED 100"
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
