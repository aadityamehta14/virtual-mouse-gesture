import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
click_threshold = 30  # Distance threshold for pinch
right_click_cooldown = 1
last_right_click_time = 0

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror image
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Tip IDs: Index - 8, Middle - 12, Thumb - 4
            x_index, y_index = lm_list[8][1], lm_list[8][2]
            x_thumb, y_thumb = lm_list[4][1], lm_list[4][2]
            x_middle, y_middle = lm_list[12][1], lm_list[12][2]

            # Map coordinates to screen
            screen_x = np.interp(x_index, (0, w), (0, screen_width))
            screen_y = np.interp(y_index, (0, h), (0, screen_height))

            # Smoothen the cursor movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Pinch Gesture = Left Click
            distance_thumb_index = np.hypot(x_thumb - x_index, y_thumb - y_index)
            if distance_thumb_index < click_threshold:
                cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                time.sleep(0.2)

            # Peace Sign = Right Click (Index + Middle finger up)
            distance_index_middle = np.hypot(x_index - x_middle, y_index - y_middle)
            if distance_index_middle < 50:
                current_time = time.time()
                if current_time - last_right_click_time > right_click_cooldown:
                    pyautogui.rightClick()
                    last_right_click_time = current_time
                    time.sleep(0.2)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
