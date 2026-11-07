# farm-shed-helper
![Project Cover](assets/cover.jpeg)

### 💡 What's the idea?
Touchless farm-shed dashboard for muddy/gloved hands. Show palm to lock, point your index fingertip at on-screen boxes to toggle MOTOR / LIGHT / ALARM, pick a door (MAIN/KITCHEN/SHED) and OPEN/CLOSE it, and drag a SPEED slider — no touching. 100% on-device MediaPipe hand tracking. Built for sheds where touchscreens die.

### 🌐 Check it out!
- **Live Website:** N/A
- **GitHub Repo:** https://github.com/mrashieee/nira

### 🛠️ What did you use?
- Python 3.11 + `uv` (`uv run`)
- OpenCV 5.0, NumPy
- MediaPipe 1.2.2 Tasks API — `HandLandmarker` (`hand_landmarker.task`)
- Fingertip position only: presence lock + dwell buttons + follow slider (no gesture classification)

### 👥 The Humans Behind It
- Muhammad Rashid K – mrashieee

### ⚙️ Hardware Corner
- Laptop webcam. No extra hardware. (Relay/Pi wiring = future.)

### 📸 Show Us What You Built!
Demo video (2 min, face + app walkthrough): [assets/Demo.mp4](assets/Demo.mp4) — click to play in GitHub's player.

### ✨ Cool Things It Does
- Palm lock-in (3 frames to lock, 5 to drop) — flicker never triggers
- Boxes: MOTOR / LIGHT / ALARM / FREEZE, door selector (MAIN/KITCHEN/SHED) + OPEN / CLOSE
- SPEED slider with 0 / 100 side buttons
- No cloud, works offline in the shed

### ▶️ Run it
```bash
uv run python shed.py --test   # self-test, no webcam
uv run python shed.py          # live: SHOW PALM, point at boxes
```

---

**Built with curiosity at Saturday Hack Night 💜**
