"""Nira Shed Buddy - touchless shed dashboard. Muddy hands, no touch.
Gestures (21 hand points, no training):
  OPEN PALM = START motor/timer | FIST = STOP | PINCH = count sack | SWIPE = switch mode
uv run python shed.py [--test]
"""
import argparse
import csv
import sys
import time
import cv2
import numpy as np

MODEL = "hand_landmarker.task"
PINCH_PX = 40
SWIPE_PX = 80
DEBOUNCE_S = 1.0


def _xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h])


class FakeLm:
    def __init__(self, x, y):
        self.x, self.y = x, y


def fingers_up(lms):
    """Count extended fingers (index..pinky tip above PIP). Thumb ignored."""
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    return sum(1 for t, p in zip(tips, pips) if lms[t].y < lms[p].y)


def is_palm(lms):
    return fingers_up(lms) >= 4


def is_fist(lms):
    return fingers_up(lms) == 0


def pinch_px(lms, w, h):
    a, b = _xy(lms[4], w, h), _xy(lms[8], w, h)
    return float(np.linalg.norm(a - b))


class Swipe:
    def __init__(self):
        self.hist = []

    def push(self, x, t):
        self.hist.append((x, t))
        self.hist = [(xx, tt) for xx, tt in self.hist if t - tt < 0.6]
        if len(self.hist) >= 5:
            dx = self.hist[-1][0] - self.hist[0][0]
            if abs(dx) > SWIPE_PX:
                self.hist.clear()
                return "right" if dx > 0 else "left"
        return None


class PinchHold:
    """Pinch held 2s = CONFIRM (door). Testable with explicit times."""
    HOLD_S = 2.0

    def __init__(self):
        self.start = None

    def update(self, pinching, now=None):
        now = time.time() if now is None else now
        if pinching:
            if self.start is None:
                self.start = now
            if now - self.start >= self.HOLD_S:
                self.start = None
                return "confirmed"
            return "holding"
        self.start = None
        return None


class Shed:
    """Modular farm assistant core (test build). Modules switch by swipe."""
    HOLD_S = 2.0

    def __init__(self):
        self.motor = False
        self.door_open = False
        self.light = False
        self.alarm = False
        self.sacks = 0
        self.t0 = None
        self.elapsed = 0.0
        self.mode = 0
        self.modes = ["MOTOR", "DOOR", "LIGHT", "ALARM", "COUNT"]
        self.last = {}
        self.hold = PinchHold()

    def allow(self, key):
        now = time.time()
        if now - self.last.get(key, 0) < DEBOUNCE_S:
            return False
        self.last[key] = now
        return True

    def event(self, name):
        mod = self.modes[self.mode]
        if name == "palm" and self.allow("palm"):
            if mod == "MOTOR":
                self.motor = True
                if self.t0 is None:
                    self.t0 = time.time()
                return "MOTOR ON"
            if mod == "DOOR":
                self.door_open = True
                self.log("door_open")
                return "DOOR OPEN"
            if mod == "LIGHT":
                self.light = True
                return "LIGHT ON"
            if mod == "ALARM":
                self.alarm = True
                return "ALARM ON"
            if mod == "COUNT":
                self.motor = True
                return "START"
        if name == "fist" and self.allow("fist"):
            if mod == "MOTOR":
                self.motor = False
                if self.t0 is not None:
                    self.elapsed += time.time() - self.t0
                    self.t0 = None
                return "STOP"
            if mod == "DOOR":
                self.door_open = False
                self.log("door_lock")
                return "DOOR LOCKED"
            if mod == "LIGHT":
                self.light = False
                return "LIGHT OFF"
            if mod == "ALARM":
                self.alarm = False
                return "ALARM OFF"
            self.motor = False
            return "STOP"
        if name == "pinch" and self.allow("pinch"):
            if mod == "DOOR":
                return "DOOR: hold pinch 2s to confirm"
            self.sacks += 1
            self.log("sack")
            return f"SACK #{self.sacks}"
        if name == "pinch_confirm":
            if mod == "DOOR":
                self.door_open = not self.door_open
                self.log("door_toggle_confirm")
                return f"DOOR {'OPEN' if self.door_open else 'LOCKED'} (confirmed)"
            return None
        if name == "both_up" and self.allow("emg"):
            self.motor = False
            self.alarm = True
            if self.t0 is not None:
                self.elapsed += time.time() - self.t0
                self.t0 = None
            self.log("EMERGENCY")
            return "EMERGENCY ALL-STOP + ALARM"
        if name in ("left", "right") and self.allow("swipe"):
            self.mode = (self.mode + (1 if name == "right" else -1)) % len(self.modes)
            return f"MODE {self.modes[self.mode]}"
        return None

    def runtime(self):
        e = self.elapsed
        if self.t0 is not None:
            e += time.time() - self.t0
        return e

    def log(self, what):
        with open("shed_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%H:%M:%S"), what, self.sacks,
                                    round(self.runtime(), 1), self.modes[self.mode]])


def draw(frame, shed, gesture):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 110), (20, 20, 20), -1)
    cv2.putText(frame, f"{shed.modes[shed.mode]} motor={'ON' if shed.motor else 'OFF'} door={'OPEN' if shed.door_open else 'LOCK'} light={'ON' if shed.light else 'OFF'} sacks={shed.sacks} t={shed.runtime():.0f}s",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "palm=ON fist=OFF pinch=+1 hold2s=confirm swipe=module 2hands=EMG q=quit",
                (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return frame


def fake_hand(pose):
    """Synthetic 21 landmarks for tests. y down (0 top)."""
    lms = [FakeLm(0.5, 0.7) for _ in range(21)]
    lms[0] = FakeLm(0.5, 0.75)  # wrist
    if pose == "palm":
        for t, p, x in [(8, 6, 0.45), (12, 10, 0.5), (16, 14, 0.55), (20, 18, 0.6)]:
            lms[t] = FakeLm(x, 0.3)
            lms[p] = FakeLm(x, 0.5)
        lms[4] = FakeLm(0.35, 0.5)
    elif pose == "fist":
        for t, p, x in [(8, 6, 0.45), (12, 10, 0.5), (16, 14, 0.55), (20, 18, 0.6)]:
            lms[t] = FakeLm(x, 0.65)
            lms[p] = FakeLm(x, 0.5)
    elif pose == "pinch":
        for t, p, x in [(8, 6, 0.45), (12, 10, 0.5), (16, 14, 0.55), (20, 18, 0.6)]:
            lms[t] = FakeLm(x, 0.3 if t != 8 else 0.5)
            lms[p] = FakeLm(x, 0.5)
        lms[4] = FakeLm(0.46, 0.5)
        lms[8] = FakeLm(0.47, 0.5)
    return lms


def selftest():
    palm, fist, pinch = fake_hand("palm"), fake_hand("fist"), fake_hand("pinch")
    assert is_palm(palm) and not is_fist(palm), "palm detect fail"
    assert is_fist(fist) and not is_palm(fist), "fist detect fail"
    assert fingers_up(palm) == 4 and fingers_up(fist) == 0
    d = pinch_px(pinch, 640, 480)
    assert d < PINCH_PX, f"pinch should be <{PINCH_PX}, got {d:.0f}"
    assert pinch_px(palm, 640, 480) > PINCH_PX, "palm must not pinch"
    sw = Swipe()
    base = time.time()
    got = None
    for i in range(6):
        r = sw.push(100 + i * 25, base + i * 0.08)
        got = r if r else got
    assert got == "right", f"swipe fail {got}"
    s = Shed()
    assert s.event("palm") == "MOTOR ON" and s.motor
    assert s.event("pinch") == "SACK #1" and s.sacks == 1
    assert s.event("pinch") is None, "debounce fail"
    assert s.event("fist") == "STOP" and not s.motor
    # modular: swipe to DOOR, pinch tap asks confirm, hold confirms, both-up emergency
    s2 = Shed()
    s2.last.clear()
    assert s2.event("right") == "MODE DOOR"
    s2.last.clear()
    assert "hold pinch" in s2.event("pinch")
    assert not s2.door_open
    assert "OPEN" in s2.event("pinch_confirm") and s2.door_open
    s3 = Shed()
    s3.event("palm")
    assert "EMERGENCY" in s3.event("both_up") and not s3.motor and s3.alarm
    ph = PinchHold()
    t0 = time.time()
    assert ph.update(True, t0) == "holding"
    assert ph.update(True, t0 + 2.1) == "confirmed"
    print("shed self-test passed: palm/fist/pinch/swipe/debounce/modules/hold/emergency")


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
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("No webcam. Try: uv run python shed.py --test")
    shed, swipe = Shed(), Swipe()
    print("Farm assistant TEST BUILD. palm=ON fist=OFF pinch=+1 hold2s=door-confirm swipe=module 2hands=EMG q=quit")
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
            gesture, msg = "none", None
            hands = res.hand_landmarks or []
            for hand in hands:  # draw all
                pts = [(int(p.x * w), int(p.y * h)) for p in hand]
                for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 0), 2)
            if len(hands) >= 2 and all(is_palm(hd) for hd in hands[:2]):
                gesture = "BOTH UP"
                msg = shed.event("both_up")
            elif hands:
                hand = hands[0]
                wx = hand[0].x * w
                sd = swipe.push(wx, time.time())
                pinching = pinch_px(hand, w, h) < PINCH_PX
                hold_state = shed.hold.update(pinching)
                if is_palm(hand):
                    gesture = "PALM"
                    msg = shed.event("palm")
                elif is_fist(hand):
                    gesture = "FIST"
                    msg = shed.event("fist")
                elif hold_state == "confirmed":
                    gesture = "PINCH HOLD 2s"
                    msg = shed.event("pinch_confirm")
                elif pinching:
                    gesture = "PINCH" if hold_state != "holding" else "PINCH..."
                    if hold_state != "holding" or shed.modes[shed.mode] != "DOOR":
                        m = shed.event("pinch")
                        msg = m
                    elif shed.modes[shed.mode] == "DOOR" and hold_state == "holding" and not msg:
                        msg = "DOOR: hold pinch 2s..."
                if sd:
                    gesture = f"SWIPE {sd}"
                    m2 = shed.event(sd)
                    msg = m2 if m2 else msg
            if msg:
                print(msg)
            cv2.imshow("Nira Shed Buddy (q quit)", draw(frame, shed, gesture))
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
