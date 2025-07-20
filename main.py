
import cv2
import mediapipe as mp
import time
from controlkeys import right_pressed, left_pressed, up_pressed, down_pressed
from controlkeys import KeyOn, KeyOff
import os

# # Fix for Matplotlib config error
# os.environ['MPLCONFIGDIR'] = os.getcwd()

# Hand tracking setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)
current_keys_pressed = set()
prev_time = 0
gesture_last_time = 0
gesture_cooldown = 0.5  # seconds
last_action = "None"

def get_fingers_status(lm_list):
    fingers = []
    # Thumb
    fingers.append(1 if lm_list[tipIds[0]][0] > lm_list[tipIds[0] - 1][0] else 0)
    # Other fingers
    for id in range(1, 5):
        fingers.append(1 if lm_list[tipIds[id]][1] < lm_list[tipIds[id] - 2][1] else 0)
    return fingers

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        keys_triggered = set()

        curr_time = time.time()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                fingers = get_fingers_status(lm_list)

                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Debounce gestures
                if curr_time - gesture_last_time > gesture_cooldown:
                    if fingers[1] == 1 and all(f == 0 for i, f in enumerate(fingers) if i != 1):
                        keys_triggered.add(right_pressed)
                        last_action = "RIGHT"
                        cv2.putText(frame, "RIGHT", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

                    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                        keys_triggered.add(left_pressed)
                        last_action = "LEFT"
                        cv2.putText(frame, "LEFT", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                    elif fingers == [0, 0, 0, 0, 0]:
                        keys_triggered.add(down_pressed)
                        last_action = "DOWN"
                        cv2.putText(frame, "DOWN", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

                    elif fingers == [1, 1, 1, 1, 1]:
                        keys_triggered.add(up_pressed)
                        last_action = "UP"
                        cv2.putText(frame, "UP", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)

                    gesture_last_time = curr_time

                # Debug
                cv2.putText(frame, f"Fingers: {fingers}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Release old keys
        for key in current_keys_pressed - keys_triggered:
            KeyOff(key)
        # Press new keys
        for key in keys_triggered - current_keys_pressed:
            KeyOn(key)

        current_keys_pressed = keys_triggered

        # FPS
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gesture help
        cv2.putText(frame, "Palm: UP | Fist: DOWN | Peace: LEFT | Index: RIGHT",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Current Action display
        cv2.putText(frame, f"Current Action: {last_action}",
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Display
        cv2.imshow("One-Hand Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
