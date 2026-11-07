"""Nira - webcam + hue checker (step 1). uv run python app.py [--test]"""
import argparse
import sys
import cv2
import numpy as np

ROI = 120  # center square size


def analyze_roi(bgr):
    h, w, _ = bgr.shape
    x0, y0 = w // 2 - ROI // 2, h // 2 - ROI // 2
    roi = bgr[y0:y0 + ROI, x0:x0 + ROI]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mh, ms, mv, _ = cv2.mean(hsv)
    # masks in full-frame HSV for veg-relevant hues (OpenCV H: 0-179)
    full = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(full, np.array([0, 70, 50]), np.array([10, 255, 255])) | \
          cv2.inRange(full, np.array([170, 70, 50]), np.array([180, 255, 255]))
    green = cv2.inRange(full, np.array([35, 70, 50]), np.array([85, 255, 255]))
    yellow = cv2.inRange(full, np.array([20, 70, 50]), np.array([35, 255, 255]))
    brown = cv2.inRange(full, np.array([8, 50, 20]), np.array([20, 200, 150]))  # defects
    total = bgr.shape[0] * bgr.shape[1]
    pct = lambda m: 100.0 * cv2.countNonZero(m) / total
    return (x0, y0), (mh, ms, mv), {"red": pct(red), "green": pct(green),
                                     "yellow": pct(yellow), "brown": pct(brown)}


def draw(frame, box, hsv_mean, pcts):
    x0, y0 = box
    cv2.rectangle(frame, (x0, y0), (x0 + ROI, y0 + ROI), (255, 255, 255), 2)
    lines = [f"H:{hsv_mean[0]:.0f} S:{hsv_mean[1]:.0f} V:{hsv_mean[2]:.0f}",
             f"R:{pcts['red']:.1f}% G:{pcts['green']:.1f}% Y:{pcts['yellow']:.1f}% D:{pcts['brown']:.1f}%"]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="headless self-test, no webcam/display")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    if args.test:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (60, 0, 200)  # fake red tomato BGR
        box, hsv_mean, pcts = analyze_roi(img)
        print(f"TEST OK H:{hsv_mean[0]:.0f} S:{hsv_mean[1]:.0f} V:{hsv_mean[2]:.0f} {pcts}")
        assert pcts["red"] > 50, "red mask should dominate fake tomato"
        print("hue checker self-test passed")
        return

    cap = cv2.VideoCapture(0)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"cannot read {args.image}")
        box, hsv_mean, pcts = analyze_roi(frame)
        print(f"H:{hsv_mean[0]:.0f} S:{hsv_mean[1]:.0f} V:{hsv_mean[2]:.0f} {pcts}")
        cv2.imwrite("hue_preview.jpg", draw(frame, box, hsv_mean, pcts))
        print("wrote hue_preview.jpg")
        return

    if not cap.isOpened():
        sys.exit("No webcam. Try: uv run python app.py --test  OR  uv run python app.py --image veg.jpg")
    print("Webcam on. q=quit s=snapshot")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        box, hsv_mean, pcts = analyze_roi(frame)
        cv2.imshow("Nira - hue checker (q quit, s save)", draw(frame, box, hsv_mean, pcts))
        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            cv2.imwrite("hue_snapshot.jpg", frame)
            print("saved hue_snapshot.jpg", pcts)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
