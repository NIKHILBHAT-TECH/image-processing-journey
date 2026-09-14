# Computer Vision & Image Processing — Learning Journey

This repository documents my journey of learning **image processing and computer vision with Python**.

I started with basic OpenCV operations and gradually moved towards real-time color detection, image segmentation, hand tracking, and gesture-based control.

Rather than following only tutorials, I’m using small projects and experiments to understand how different computer vision techniques actually work and how they can be combined to build interactive applications.

---

## Repository Structure

### 01 — Basic OpenCV

The first set of projects focuses on understanding the fundamentals of image processing and working with live webcam input.

#### `blur.py`

Explores **Gaussian blur** by applying different kernel sizes side-by-side to see how the strength of the blur changes the image.

#### `customblur.py`

Combines **Gaussian blur and Canny edge detection**, with trackbars for changing the blur parameters in real time.

#### `rgb2hsv.py`

Converts and reads pixel information in the **HSV color space**. Clicking on the webcam feed returns the HSV value of that pixel, which is useful when calibrating colors for detection.

**Concepts covered:**

* OpenCV basics
* Webcam input
* Image filtering
* Gaussian blur
* Kernel size
* Canny edge detection
* RGB vs HSV color spaces
* Trackbars
* Pixel-level color analysis

---

### 02 — Color Detection & Contours

The next step was using color information to identify objects or regions in a live video stream.

#### `colourdetection_withcountor.py`

Detects a predefined HSV color range, identifies the matching regions, and finds the largest detected blob before drawing a bounding circle around it.

#### `CUSTOMCOLORDETCTIONWITHBLUR.py`

Extends the previous project by adding interactive controls for adjusting the HSV range and blur parameters while the camera is running.

**Concepts covered:**

* HSV thresholding
* Color segmentation
* Binary masks
* Contours
* Blob detection
* Largest contour selection
* Bounding geometry
* Real-time parameter tuning
* Noise reduction using blur

---

### 03 — Hand Tracking & Gesture Control

The later projects move from traditional image processing into **landmark-based computer vision** using MediaPipe.

Instead of only detecting pixels or colors, these projects use hand landmarks to understand the position and movement of the hand and convert those movements into commands.

#### `hand.py`

Tracks hand landmarks and uses a **thumb-index pinch gesture** to control the mouse cursor and perform clicks.

#### `clicker.py`

Tracks thumb movement and detects a quick flick gesture by checking whether the thumb crosses predefined lines in sequence.

#### `dino1game.py`

Uses the index finger's position relative to threshold lines to control **jump and duck actions** in the Chrome Dino game.

#### `dinosimulator.py`

A refined version of the Dino controller that uses a timed flick-up-and-down gesture for jumping and a hold gesture for ducking.

#### `brightness_control.py`

Uses a thumb-index pinch as the starting gesture and horizontal hand movement to control laptop screen brightness in real time.

**Concepts covered:**

* MediaPipe Hand Landmarker
* Hand landmarks
* Landmark coordinates
* Gesture detection
* Distance and movement calculations
* Position-based thresholds
* Motion tracking
* Real-time video processing
* Gesture-to-action mapping
* Human-computer interaction
* Computer automation

---

softwares and libraries Used

* **Python**
* **OpenCV**
* **MediaPipe**
* **NumPy**
* **PyAutoGUI**
* **Screen Brightness Control**

---

## Progression

The projects in this repository follow a gradual progression:

```text
OpenCV Basics
      ↓
Image Filtering
      ↓
Edge Detection
      ↓
RGB → HSV
      ↓
Color Segmentation
      ↓
Contour Detection
      ↓
Hand Landmark Detection
      ↓
Gesture Recognition
      ↓
Real-Time Computer Control
```

The goal is to keep building on these concepts and eventually move towards more advanced computer vision applications, including **object detection, pose estimation, tracking, and computer vision with embedded systems such as ESP32 and Raspberry Pi**.

This repository will continue to grow as I learn and build more.
