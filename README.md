# Human Pose Estimation & Activity Classification Pipeline

> **Computer Vision Assignment **
> MediaPipe Tasks API | Rule-Based Classifier | 

---

##  Project Overview

This project implements a complete pipeline that:
1. Detects human body poses using a pre-trained MediaPipe model
2. Computes joint angles (Knee, Hip, Shoulder) across all video frames
3. Classifies physical activities (Squat / Standing) using a rule-based threshold system

**Video Source:** `https://youtube.com/shorts/-5LhNSMBrEs?si=D0eUTqKjlbvmV_-k` — Pre-recorded gym clip showing squat and standing activities.

---

## Repository Structure

```
├── pose_activity_classification.ipynb   # Main Jupyter notebook (all 3 tasks)
├── exercise.mp4                         # Input video
├── pose_landmarker.task                 # MediaPipe model file
├── output_annotated.mp4                 # Output video with skeleton + angle overlay
├── plot_angles.png                      # Knee/Hip/Shoulder angle curves over time
├── plot_classification.png              # Per-frame predicted vs ground-truth chart
├── skeleton_overlay.png                 # MediaPipe skeleton on 5 sample frames
└── README.md                            # This file
```

---

##  Setup & Installation

### Google Colab (Recommended)

```python
# Cell 0 — Run this first, then restart runtime
!pip install mediapipe -q
!wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task -O pose_landmarker.task
```

>  **Important:** After running Cell 0, go to **Runtime → Restart session**, then run all remaining cells.

### Local Environment

```bash
pip install mediapipe opencv-python numpy matplotlib
```

---

##  How to Run

1. Upload `exercise.mp4` to your Colab session
2. Run **Cell 0** (install + download model)
3. **Restart Runtime**
4. Run all cells top to bottom (`Ctrl+F9`)

All output files will be saved automatically in the working directory.

---

##  Pipeline Details

### Task 1 — Pose Detection & Pre-processing

- **Model:** MediaPipe PoseLandmarker Lite (`pose_landmarker.task`)
- **API:** MediaPipe Tasks API (modern, TensorFlow-independent)
- **Output:** 33 body landmarks per frame in normalized (x, y) coordinates
- **Smoothing:** 5-frame rolling-mean filter to reduce jitter

### Task 2 — Joint Angle Computation

Angles computed using the **dot-product formula** at the vertex joint:

```
angle = arccos( (BA · BC) / (|BA| × |BC|) )
```

| Joint | Landmarks | Purpose |
|-------|-----------|---------|
| Knee  | Hip(24) → Knee(26) → Ankle(28) | Primary classifier feature |
| Hip   | Shoulder(12) → Hip(24) → Knee(26) | Secondary confirmation |
| Shoulder | Elbow(14) → Shoulder(12) → Hip(24) | Tertiary indicator |

**Observed ranges:**

| Angle | Squat | Standing |
|-------|-------|----------|
| Knee  | ~42° – 105° | ~140° – 165° |
| Hip   | ~32° – 110° | ~130° – 180° |

### Task 3 — Rule-Based Classifier

```python
if knee_angle > 140 and hip_angle > 130:
    activity = "Standing"
elif knee_angle < 105 and hip_angle < 110:
    activity = "Squat"
else:
    activity = "Transition"
```

---

##  Results

| Metric | Value |
|--------|-------|
| Total Frames | 254 |
| Pose Detected | 254 / 254 (100%) |
| Non-Transition Frames | 228 |
| Correctly Classified | 215 |
| **Overall Accuracy** | **94.3% ** |
| Target Accuracy | 90% |

> **Target exceeded by 4.3%**

---

##  Output Files

| File | Description |
|------|-------------|
| `output_annotated.mp4` | Video with skeleton, knee/hip angles, and activity label on each frame |
| `plot_angles.png` | Angle curves over time with threshold lines marked |
| `plot_classification.png` | Per-frame predicted vs ground-truth activity comparison |
| `skeleton_overlay.png` | Sample frames showing MediaPipe skeleton overlay |

---

##  Dependencies

| Package | Purpose |
|---------|---------|
| `mediapipe` | Pose landmark detection |
| `opencv-python` | Video I/O and frame annotation |
| `numpy` | Numerical computations |
| `matplotlib` | Plotting angle and classification graphs |

---

##  Notes

- Only **right-side landmarks** were used for angle computation
- **Transition frames** were excluded from accuracy evaluation (inherently ambiguous)
- The classifier uses **fixed thresholds** — may need tuning for different camera angles or body types
- Each student must submit independently

---

##  Assignment Info

| Field | Detail |
|-------|--------|
| Course | Computer Vision |
| Activities Detected | Squat, Standing, Transition |
| Model Used | MediaPipe PoseLandmarker Lite |
| Accuracy Achieved | 94.3% |
