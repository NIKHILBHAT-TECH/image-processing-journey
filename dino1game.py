import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import pyautogui

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Done.")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open webcam")
    exit()

frame_timestamp_ms = 0
previous_index_point = None

# ---- Jump (same simple style as duck) ----
JUMP_LINE_RATIO = 0.40
JUMP_HOLD_TIME = 0.10          # how long finger must stay above the line
JUMP_COOLDOWN = 0.45
above_jump_since = None
last_jump_time = 0.0

# ---- Duck (unchanged) ----
DUCK_LINE_RATIO = 0.75
DUCK_HOLD_TIME = 0.15
below_duck_since = None
is_ducking = False

print("Jump  → hold index finger ABOVE the green line")
print("Duck  → hold index finger BELOW the red line")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 33

    h, w, _ = frame.shape
    jump_line_y = int(h * JUMP_LINE_RATIO)
    duck_line_y = int(h * DUCK_LINE_RATIO)

    # Guide lines
    cv2.line(frame, (0, jump_line_y), (w, jump_line_y), (0, 200, 0), 2)
    cv2.putText(frame, "Jump line (hold above)", (10, jump_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    cv2.line(frame, (0, duck_line_y), (w, duck_line_y), (0, 0, 200), 2)
    cv2.putText(frame, "Duck line (hold below)", (10, duck_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    finger_seen = False
    now = time.time()

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            index_point = points[8]
            finger_seen = True
            cy = index_point[1]

            # Trail
            if previous_index_point is not None:
                cv2.line(frame, previous_index_point, index_point, (0, 255, 255), 3)
            previous_index_point = index_point

            # ---------- JUMP (same logic style as duck) ----------
            if cy < jump_line_y:                       # finger is ABOVE the line
                if above_jump_since is None:
                    above_jump_since = now
                elif (now - above_jump_since > JUMP_HOLD_TIME and
                      now - last_jump_time > JUMP_COOLDOWN):
                    pyautogui.press('space')
                    last_jump_time = now
                    above_jump_since = None            # reset so it doesn't spam
            else:
                above_jump_since = None

            # ---------- DUCK (unchanged) ----------
            if cy > duck_line_y:
                if below_duck_since is None:
                    below_duck_since = now
                elif now - below_duck_since > DUCK_HOLD_TIME and not is_ducking:
                    pyautogui.keyDown('down')
                    is_ducking = True
            else:
                below_duck_since = None
                if is_ducking:
                    pyautogui.keyUp('down')
                    is_ducking = False

            # Visual feedback
            if is_ducking:
                dot_color = (0, 0, 200)
                status = "DUCK"
            elif cy < jump_line_y:
                dot_color = (0, 255, 0)
                status = "JUMP"
            else:
                dot_color = (0, 0, 255)
                status = ""

            cv2.circle(frame, index_point, 12, dot_color, -1)
            if status:
                cv2.putText(frame, status, (index_point[0] + 15, index_point[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, dot_color, 2)

            # Skeleton
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)
            for idx, (cx, cy) in enumerate(points):
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # Hand lost → clean up
    if not finger_seen:
        if is_ducking:
            pyautogui.keyUp('down')
            is_ducking = False
        below_duck_since = None
        above_jump_since = None
        previous_index_point = None

    cv2.imshow("Dino Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if is_ducking:
            pyautogui.keyUp('down')
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()