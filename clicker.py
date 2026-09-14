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
    num_hands=2,
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
previous_thumb_point = None
screen_width, screen_height = pyautogui.size()

# ---- Gesture detection state ----
LINE_Y_RATIO = 0.4          # line drawn at 40% down the frame height
CROSS_TIMEOUT = 0.6         # must complete up->down within 0.6 sec
armed = False               # True once thumb has crossed upward, waiting for return
armed_time = 0.0
prev_side = None            # 'above' or 'below' the line
last_click_time = 0
CLICK_COOLDOWN = 0.5        # prevent rapid re-fire


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
    line_y = int(h * LINE_Y_RATIO)

    # draw the reference line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 200, 0), 2)
    cv2.putText(frame, "Flick line", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    if result.hand_landmarks:
        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            thumb_point = points[4]

            if hand_index == 0:
                if previous_thumb_point is not None:
                    cv2.line(frame, previous_thumb_point, thumb_point, (0, 255, 255), 3)
                previous_thumb_point = thumb_point

                current_side = "above" if thumb_point[1] < line_y else "below"
                now = time.time()

                if prev_side is not None and prev_side != current_side:
                    if prev_side == "below" and current_side == "above":
                        # crossed upward -> arm the gesture
                        armed = True
                        armed_time = now
                    elif prev_side == "above" and current_side == "below" and armed:
                        # crossed back down within the timeout -> fire click
                        if now - armed_time <= CROSS_TIMEOUT and now - last_click_time > CLICK_COOLDOWN:
                            screen_x = int(thumb_point[0] * screen_width / w)
                            screen_y = int(thumb_point[1] * screen_height / h)
                            pyautogui.moveTo(screen_x, screen_y, duration=0.05)
                            pyautogui.click()
                            last_click_time = now
                        armed = False

                # cancel the arm if it's taking too long
                if armed and (now - armed_time > CROSS_TIMEOUT):
                    armed = False

                prev_side = current_side

                cv2.circle(frame, thumb_point, 10, (0, 255, 255) if armed else (0, 0, 255), -1)
                cv2.putText(frame, "ARMED" if armed else "",
                            (thumb_point[0] + 10, thumb_point[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)

            for idx, (cx, cy) in enumerate(points):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(idx), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imshow("MediaPipe Hands", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()