import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import math
import screen_brightness_control as sbc

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

# ---- Pinch-and-slide brightness control ----
PINCH_THRESHOLD_RATIO = 0.35   # thumb-index distance / hand-scale below this = "pinching"
SENSITIVITY = 0.35              # pixels of horizontal movement -> % brightness per pixel
is_pinching = False
last_pinch_x = None

try:
    current_brightness = sbc.get_brightness()[0]
except Exception as e:
    print(f"Could not read current brightness ({e}). Defaulting to 50.")
    current_brightness = 50

last_set_time = 0.0
SET_INTERVAL = 0.05  # throttle how often we actually push a brightness change (seconds)

print("Pinch thumb + index finger together, then slide left/right to change brightness.")
print("Slide right -> brighter | Slide left -> dimmer")
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
    hand_found = False
    now = time.time()

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            hand_found = True

            thumb_tip = points[4]
            index_tip = points[8]
            wrist = points[0]
            middle_mcp = points[9]

            # Hand-scale reference so the pinch threshold works at any distance from the camera
            hand_scale = math.hypot(wrist[0] - middle_mcp[0], wrist[1] - middle_mcp[1])
            hand_scale = max(hand_scale, 1)  # avoid divide-by-zero

            pinch_dist = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
            pinch_ratio = pinch_dist / hand_scale

            mid_x = (thumb_tip[0] + index_tip[0]) // 2
            mid_y = (thumb_tip[1] + index_tip[1]) // 2

            currently_pinching = pinch_ratio < PINCH_THRESHOLD_RATIO

            if currently_pinching:
                if not is_pinching:
                    # Pinch just started
                    is_pinching = True
                    last_pinch_x = mid_x
                else:
                    delta_x = mid_x - last_pinch_x
                    last_pinch_x = mid_x

                    if delta_x != 0 and (now - last_set_time) > SET_INTERVAL:
                        current_brightness += delta_x * SENSITIVITY
                        current_brightness = max(0, min(100, current_brightness))
                        try:
                            sbc.set_brightness(int(current_brightness))
                        except Exception as e:
                            print(f"Failed to set brightness: {e}")
                        last_set_time = now
            else:
                is_pinching = False
                last_pinch_x = None

            # Visual feedback
            pinch_color = (0, 255, 0) if currently_pinching else (0, 0, 255)
            cv2.line(frame, thumb_tip, index_tip, pinch_color, 2)
            cv2.circle(frame, thumb_tip, 8, pinch_color, -1)
            cv2.circle(frame, index_tip, 8, pinch_color, -1)
            cv2.circle(frame, (mid_x, mid_y), 5, (255, 255, 0), -1)

            status = "PINCH - slide to adjust" if currently_pinching else "Pinch fingers to start"
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, pinch_color, 2)

            # Skeleton
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)
            for cx, cy in points:
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    if not hand_found:
        is_pinching = False
        last_pinch_x = None

    # Brightness bar UI
    bar_x, bar_y, bar_w, bar_h = 10, h - 40, 300, 20
    fill_w = int(bar_w * (current_brightness / 100))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 0), -1)
    cv2.putText(frame, f"Brightness: {int(current_brightness)}%",
                (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Pinch-Slide Brightness Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()