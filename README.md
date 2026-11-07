# Nira
![Project Cover](assets/cover.jpeg)

### 💡 What's the idea?
Touchless vegetable checker for small vendors and kitchens. Point webcam at vegetables, get live ripeness and freshness score with Grade A / B / Reject badge. Control it with hand gestures — built for dirty mandis and kitchens. 100% on-device, works with no internet.

### 🌐 Check it out!
- **Live Website:** N/A
- **GitHub Repo:** https://github.com/mrashieee/nira

### 🛠️ What did you use?
- Python 3.11 + `uv` (`uv run`)
- OpenCV 5.0 (`opencv-python`), NumPy
- MediaPipe 1.2.2 Tasks API — `HandLandmarker` (`hand_landmarker.task`)
- HSV color masks + contour area for grading

### 👥 The Humans Behind It
- Muhammad Rashid K – mrashieee

### ⚙️ Hardware Corner
- Laptop webcam (no webcam in container, tested via image/file mode)
- White paper for one-click color calibration, optional desk lamp lightbox

### 📸 Show Us What You Built!
[Add screenshots, photos, GIFs or demo video links here.]

### ✨ Cool Things It Does
- Live ripeness + freshness % with HSV color ratios (lighting-stable)
- `c` to calibrate, hand gestures to control + save to CSV log
- Hand-as-scale for size estimate + fully touchless UI
- Honest limitation: controlled light prototype, HSV presets not lab-grade

### ▶️ Run it
```bash
uv run python app.py
```

---

**Built with curiosity at Saturday Hack Night 💜**
