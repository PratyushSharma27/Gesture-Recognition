# Made by Pratyush Sharma
# On 27/01/2024

import cv2
import mediapipe as mp
import pyautogui

class HandLabels:
    LEFT = "Left"
    RIGHT = "Right"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
prev_x = None
prev_y = None

CLICK_THRESHOLD = 50
QUIT_KEY = ord('q')

while cap.isOpened():
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret or frame is None:
        continue

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for idx, landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label

            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mid = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            if handedness == HandLabels.LEFT:
                mcp_x = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                mcp_y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

                cursor_x = int(mcp_x * screen_width)
                cursor_y = int(mcp_y * screen_height)

                pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

                if index_tip.y >= index_mid.y:
                    pyautogui.click()

            elif handedness == HandLabels.RIGHT:
                x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
                if prev_x is not None and prev_y is not None:
                    dx = x - prev_x
                    dy = y - prev_y

                    if abs(dx) > abs(dy):
                        if dx > CLICK_THRESHOLD:
                            pyautogui.press('right')
                        elif dx < -CLICK_THRESHOLD:
                            pyautogui.press('left')
                    else:
                        if dy > CLICK_THRESHOLD:
                            pyautogui.press('down')
                        elif dy < -CLICK_THRESHOLD:
                            pyautogui.press('up')

                prev_x, prev_y = x, y

    cv2.imshow("Gesture Recognition - Pratyush Sharma", frame)

    if cv2.waitKey(10) & 0xFF == QUIT_KEY:
        break

cap.release()
cv2.destroyAllWindows()