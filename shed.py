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
    """Tip x inside strip sets 0-100 with smoothing. No grab gesture needed."""

    def __init__(self, alpha=0.35):
        self.value = 0.0
        self.alpha = alpha

    def update(self, tip, rect):
        x0, y0, x1, y1 = rect
        if not (x0 <= tip[0] <= x1 and y0 <= tip[1] <= y1):
            return None
        target = 100.0 * (tip[0] - x0) / max(x1 - x0, 1)
        self.value += self.alpha * (target - self.value)
        return self.value


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
        elif bid == "alarm":
            self.alarm = not self.alarm
        elif bid == "count":
            self.count += 1
        elif bid == "reset":
            self.__init__()
            return "RESET"
        self.log(bid)
        return f"{bid.upper()} {'ON' if getattr(self, bid, None) is True else ''}".strip()

    def log(self, what):
        with open("shed_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%H:%M:%S"), what, self.count,
                                    round(self.speed), self.motor, self.light,
                                    self.door, self.alarm])


BUTTONS = [
    ("motor", "MOTOR", (0.05, 0.30, 0.30, 0.46)),
    ("light", "LIGHT", (0.35, 0.30, 0.60, 0.46)),
    ("door", "DOOR", (0.65, 0.30, 0.90, 0.46)),
    ("alarm", "ALARM", (0.05, 0.50, 0.30, 0.66)),
    ("count", "COUNT+1", (0.35, 0.50, 0.60, 0.66)),
    ("reset", "RESET", (0.65, 0.50, 0.90, 0.66)),
]
SLIDER_NORM = (0.05, 0.78, 0.90, 0.90)  # SPEED slider strip


def draw(frame, board, lock, buttons=None, dwell=None, slider_rect=None):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.putText(frame, f"{'LOCKED - point fingertip' if lock else 'SHOW PALM TO LOCK'}  "
                       f"M:{int(board.motor)} L:{int(board.light)} D:{int(board.door)} "
                       f"A:{int(board.alarm)} N:{board.count} SPD:{board.speed:.0f}",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if lock else (0, 165, 255), 2)
    cv2.putText(frame, "dwell 0.6s on box | slider follows fingertip | q=quit",
                (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    if buttons:
        for (bid, label, (x0, y0, x1, y1)) in buttons:
            X0, Y0, X1, Y1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
            on = board.__dict__.get(bid) is True
            hot = dwell is not None and dwell.cur == bid
            col = (0, 255, 0) if on else ((0, 255, 255) if hot else (90, 90, 90))
            cv2.rectangle(frame, (X0, Y0), (X1, Y1), col, 3 if (hot or on) else 2)
            cv2.putText(frame, f"{label} {'ON' if on else ''}", (X0 + 10, Y0 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
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
    for _ in range(20):
        v = sl.update((500, 150), (100, 100, 500, 200))
    assert v > 90, f"slider converges right {v}"
    b = Board()
    assert b.toggle("motor") == "MOTOR ON" and b.motor
    assert b.toggle("motor") == "MOTOR" and not b.motor
    assert b.toggle("count").startswith("COUNT") and b.count == 1
    print("boxes self-test passed: presence/dwell/slider/toggles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("No webcam. Try: uv run python shed.py --test")
    board, lock, dwell, slider = Board(), PresenceLock(), DwellButton(), Slider()
    print("farm-shed-helper. SHOW PALM to lock, point fingertip at boxes, slider follows tip. q=quit")
    with vision.HandLandmarker.create_from_options(opts) as lm:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(mp_img, int(time.time() * 1000))
            hands = res.hand_landmarks or []
            locked = lock.update(bool(hands))
            msg = None
            bpx = [(b, l, (x0 * w, y0 * h, x1 * w, y1 * h)) for (b, l, (x0, y0, x1, y1)) in BUTTONS]
            srect = (SLIDER_NORM[0] * w, SLIDER_NORM[1] * h, SLIDER_NORM[2] * w, SLIDER_NORM[3] * h)
            if hands:
                hand = hands[0]
                pts = [(int(p.x * w), int(p.y * h)) for p in hand]
                for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 0), 2)
                tip = (hand[8].x * w, hand[8].y * h)
                cv2.circle(frame, (int(tip[0]), int(tip[1])), 10, (0, 255, 255), -1)
                if locked:
                    v = slider.update(tip, srect)
                    if v is not None:
                        board.speed = v
                    fired = dwell.update(tip, bpx)
                    if fired:
                        msg = board.toggle(fired)
            else:
                dwell.reset()
            if msg:
                print(msg, flush=True)
            cv2.imshow("farm-shed-helper (q quit)", draw(frame, board, locked, bpx, dwell, SLIDER_NORM))
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
