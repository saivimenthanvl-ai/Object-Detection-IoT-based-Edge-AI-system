# AI Edge Tracking System

## Overview

This project is a minimal AI video analytics pipeline for **person and vehicle tracking** using **Ultralytics YOLO** and **ByteTrack**.

It is designed as a safe starter system for:

* person tracking
* vehicle tracking
* track ID assignment across frames
* annotated video output generation

This version does **not** perform:

* helmet violation detection
* number plate OCR
* automatic fines
* identity recognition

Those features can be added later after the tracking pipeline is stable.

---

## Features

* Detects people and common vehicle classes
* Tracks objects across frames with persistent IDs
* Draws bounding boxes and labels on each frame
* Saves an annotated output video
* Works with local MP4 input files

---

## Project Structure

```text
Object Detection IoT-based Edge AI system/
├── main.py
├── helmet.mp4
├── no helmet.mp4
├── outputs/
│   └── tracked_output.mp4
├── .venv/
└── README.md
```

---

## Requirements

* Python 3.11
* VS Code
* pip
* virtual environment support

Python packages:

* ultralytics
* opencv-python

---

## Installation

### 1. Create the project folder

```bash
mkdir "Object Detection IoT-based Edge AI system"
cd "Object Detection IoT-based Edge AI system"
```

### 2. Create a virtual environment

#### Windows PowerShell

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install ultralytics opencv-python
```

---

## Main Script

Create a file named `main.py` and paste your tracking code into it.

The minimal configuration section should look like this:

```python
VIDEO_SOURCE = "helmet.mp4"
MODEL_PATH = "yolov8n.pt"
TRACKER_CFG = "bytetrack.yaml"
CONF_THRESHOLD = 0.25
```

---

## Input Video

Place your input video in the project root.

Examples:

* `helmet.mp4`
* `no helmet.mp4`

If the filename contains spaces, rename it to avoid path issues.

Example:

```text
no helmet.mp4  ->  no_helmet.mp4
```

Then update `VIDEO_SOURCE` in `main.py`:

```python
VIDEO_SOURCE = "no_helmet.mp4"
```

---

## How to Run

Make sure the virtual environment is active, then run:

```bash
python main.py
```

On first run, Ultralytics may automatically download `yolov8n.pt`.

---

## Output

The script will:

* open the input video
* detect and track people and vehicles
* assign object IDs
* display the annotated video in a window
* save the result to:

```text
outputs/tracked_output.mp4
```

Press `Esc` to stop playback early.

---

## Tracked Classes

The default model tracks these COCO classes:

* person
* car
* motorcycle
* bus
* truck

These are filtered in code using:

```python
TARGET_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}
```

---

## Troubleshooting

### 1. Video not opening

Error:

```text
Could not open video
```

Check:

* the file exists in the project folder
* the filename in `VIDEO_SOURCE` is correct
* the video format is supported

### 2. Module not found

Example:

```text
ModuleNotFoundError: No module named 'ultralytics'
```

Fix:

```bash
pip install ultralytics opencv-python
```

### 3. Virtual environment not active

If VS Code shows missing imports, select the correct interpreter:

* `Ctrl + Shift + P`
* `Python: Select Interpreter`
* choose `.venv`

### 4. Output video not saved

Make sure the `outputs/` folder exists. The script usually creates it automatically.

---

## Git Setup

### 1. Initialize git

```bash
git init
```

### 2. Add files

```bash
git add .
```

### 3. Create first commit

```bash
git commit -m "Initial commit for AI edge tracking system"
```

---

## Add a `.gitignore`

Create a `.gitignore` file to avoid pushing temporary and large files:

```gitignore
.venv/
__pycache__/
*.pyc
outputs/
*.pt
```

If your videos are large, also add:

```gitignore
*.mp4
```

---

## Connect to GitHub

### 1. Create an empty GitHub repository

Create a new repository on GitHub, for example:

```text
ai-edge-tracking-system
```

### 2. Add remote origin

Replace the URL with your repository URL:

```bash
git remote add origin https://github.com/saivimenthanvl-ai/Object-Detection-IoT-based-Edge-AI-system.git
```

### 3. Rename branch to main

```bash
git branch -M main
```

### 4. Push code

```bash
git push -u origin main
```

---

## Typical Git Workflow

After making changes:

```bash
git status
git add .
git commit -m "Updated tracking pipeline"
git push
```

---

## Future Improvements

Possible next steps:

* helmet detection
* vehicle number plate detection
* OCR integration
* JSON/CSV event logging
* email alerts for human review
* TensorRT / ONNX optimization for edge deployment
* custom YOLO training on helmet and plate datasets

---

## Safety Note

This project should be used with **human review** for any enforcement-related workflow. Model predictions can be wrong, especially in low light, occlusion, motion blur, or crowded scenes.

---
