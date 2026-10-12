# 🧠 Cognitive Load Detector (Final Year AI/ML Project)

A real-time webcam-based AI system that estimates a person's **cognitive load (attention state)** using passive behavioral signals—without requiring any self-reporting.

## ✨ Features

The system analyzes the following real-time signals:

- 👁️ **Blink Rate** (using Eye Aspect Ratio - EAR)
- 🎯 **Gaze Drift** (iris position relative to the eye socket)
- 🧭 **Head Movement** (Pitch, Yaw, Roll using `solvePnP`)
- 😴 **Drowsiness Detection** (sustained eye closure)

These signals are combined into a live **Cognitive Load Score (0–100)** and classified as:

- 🟢 Low
- 🟡 Medium
- 🔴 High

The score is displayed as a real-time overlay on the webcam feed.

---

# 🚀 Setup in VS Code

## Step 1: Install Python

Install **Python 3.9–3.11** (Python 3.10 is recommended because MediaPipe has the best compatibility).

Check your installation:

```bash
python --version
```

---

## Step 2: Open the Project

Open VS Code.

Go to:

```
File → Open Folder
```

Select the project folder:

```
cognitive-load-detector
```

---

## Step 3: Create a Virtual Environment

Open the VS Code Terminal.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

If VS Code asks you to **Select Interpreter**, choose the Python interpreter inside the `venv` folder.

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5: Run the Project

```bash
python main.py
```

A webcam window will open showing:

- Live face detection
- Cognitive Load Score
- Low / Medium / High status

### Keyboard Shortcuts

- **Q** → Quit
- **R** → Reset the session

If the wrong webcam opens, change:

```python
CAM_INDEX = 0
```

to

```python
CAM_INDEX = 1
```

or

```python
CAM_INDEX = 2
```

inside `main.py`.

---

# 📂 Project Structure

```text
cognitive-load-detector/
│
├── main.py               # Webcam loop and UI
├── face_utils.py         # EAR, gaze, head pose calculations
├── cognitive_load.py     # Cognitive Load scoring model
├── requirements.txt
└── README.md
```

---

# ⚙️ How the Scoring Works

The project uses an **Explainable Heuristic Scoring Model** over a rolling **15-second window**.

| Signal | Measurement | Effect on Load |
|---------|-------------|----------------|
| Blink Rate | Blinks per minute using EAR | Very low (<8/min) or very high (>25/min) increases cognitive load |
| Gaze Variance | Standard deviation of iris position | High variance indicates attention drift |
| Head Movement | Frame-to-frame Pitch, Yaw, Roll change | Frequent movement increases cognitive load |
| Drowsiness | Eyes closed for ≥1.5 seconds | Score is forced to at least **85** |

### Final Score Formula

```text
Score =
0.40 × Blink Score
+ 0.35 × Gaze Score
+ 0.25 × Head Movement Score
```

The final value is clipped between **0 and 100**.

Because the model is **explainable**, every score can be justified during demonstrations or project viva.

---

# 📈 Future Scope

Some possible improvements include:

### 1. Machine Learning Model

Collect labeled data and train models such as:

- Random Forest
- XGBoost

instead of using a heuristic formula.

---

### 2. Typing and Quiz Analysis

Include typing speed and response time to estimate cognitive load during online learning.

---

### 3. Session Logging

Save session data into CSV files and generate graphs showing cognitive load over time.

---

### 4. Streamlit Dashboard

Replace the OpenCV interface with a modern Streamlit dashboard.

---

### 5. Multi-Face Detection

Support multiple users and calculate a classroom-wide attention score.

---

# 🛠 Troubleshooting

### MediaPipe Module Not Found

```text
ModuleNotFoundError: mediapipe
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Then reinstall:

```bash
pip install -r requirements.txt
```

---

### Webcam Doesn't Open

- Check camera permissions.
- Ensure no other application is using the webcam.

---

### Low FPS

Reduce webcam resolution:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

Add these lines after:

```python
cv2.VideoCapture(...)
```

inside `main.py`.

---

# 🎓 Project Summary

This project demonstrates a real-time, explainable AI approach for estimating cognitive load using webcam-based behavioral signals.

It combines:

- Blink Detection
- Eye Gaze Tracking
- Head Pose Estimation
- Drowsiness Detection

to generate an interpretable **Cognitive Load Score**, making it suitable for AI/ML final-year projects, demonstrations, and academic presentations.