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

# ---- Jump gesture state (flick up then down across JUMP_LINE) ----
JUMP_LINE_RATIO = 0.4       # jump trigger line, 40% down the frame
CROSS_TIMEOUT = 0.5         # must complete up->down within this window
armed = False
armed_time = 0.0
prev_side = None
last_jump_time = 0
JUMP_COOLDOWN = 0.4

# ---- Duck gesture state (index finger held below DUCK_LINE) ----
DUCK_LINE_RATIO = 0.75      # duck trigger line, lower on the frame
DUCK_HOLD_TIME = 0.15       # must stay below this long before ducking (avoids jitter)
below_duck_since = None
is_ducking = False


print("Move your index finger UP across the green line then back DOWN quickly to JUMP.")
print("Hold your index finger below the red line to DUCK.")
print("Press 'q' in the video window to quit.")


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

    cv2.line(frame, (0, jump_line_y), (w, jump_line_y), (0, 200, 0), 2)
    cv2.putText(frame, "Jump line", (10, jump_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    cv2.line(frame, (0, duck_line_y), (w, duck_line_y), (0, 0, 200), 2)
    cv2.putText(frame, "Duck line", (10, duck_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    finger_seen = False

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            index_point = points[8]
            finger_seen = True

            if previous_index_point is not None:
                cv2.line(frame, previous_index_point, index_point, (0, 255, 255), 3)
            previous_index_point = index_point

            now = time.time()

            # ---- Jump detection ----
            current_side = "above" if index_point[1] < jump_line_y else "below"

            if prev_side is not None and prev_side != current_side:
                if prev_side == "below" and current_side == "above":
                    armed = True
                    armed_time = now
                elif prev_side == "above" and current_side == "below" and armed:
                    if now - armed_time <= CROSS_TIMEOUT and now - last_jump_time > JUMP_COOLDOWN:
                        pyautogui.press('space')
                        last_jump_time = now
                    armed = False

            if armed and (now - armed_time > CROSS_TIMEOUT):
                armed = False

            prev_side = current_side

            # ---- Duck detection ----
            if index_point[1] > duck_line_y:
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

            dot_color = (0, 255, 255) if armed else (0, 0, 255)
            if is_ducking:
                dot_color = (0, 0, 200)
            cv2.circle(frame, index_point, 10, dot_color, -1)

            status_text = ""
            if is_ducking:
                status_text = "DUCK"
            elif armed:
                status_text = "ARMED"
            cv2.putText(frame, status_text, (index_point[0] + 10, index_point[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, dot_color, 2)

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)

            for idx, (cx, cy) in enumerate(points):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(idx), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # release duck key if hand disappears mid-duck
    if not finger_seen and is_ducking:
        pyautogui.keyUp('down')
        is_ducking = False
        below_duck_since = None

    cv2.imshow("Dino Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if is_ducking:
            pyautogui.keyUp('down')
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()