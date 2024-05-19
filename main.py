import cv2
import mediapipe as mp
import serial


ser = serial.Serial("COM16", 9600, timeout=1)  # Change your port name COM... and your baudrate


def send_signal_to_robot(comand):
    if comand == 'f':
        ser.write(b'f')
    elif comand == 'b':
        ser.write(b'b')
    elif comand == 'r':
        ser.write(b'r')
    elif comand == 'l':
        ser.write(b'l')
    else:
        ser.write(b's')


mapa = {
    "Fist": "f",
    "Pointing": "b",
    "Victory": "r",
    "Thumbs Up": "l",
    "Open Hand": "s",
    "Unclassified": "s"}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.95,
    min_tracking_confidence=0.65)
mp_drawing = mp.solutions.drawing_utils


def classify_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    if (thumb_tip.y < index_mcp.y and
            index_tip.y < index_mcp.y and
            middle_tip.y < index_mcp.y and
            ring_tip.y < index_mcp.y and
            pinky_tip.y < index_mcp.y):
        return "Open Hand"

    if (thumb_tip.y > index_mcp.y and
            index_tip.y > index_mcp.y and
            middle_tip.y > index_mcp.y and
            ring_tip.y > index_mcp.y and
            pinky_tip.y > index_mcp.y):
        return "Fist"

    if thumb_tip.y < index_mcp.y and index_tip.y > index_mcp.y:
        return "Thumbs Up"

    if (index_tip.y < index_mcp.y and
            middle_tip.y < index_mcp.y and
            ring_tip.y > index_mcp.y and
            pinky_tip.y > index_mcp.y):
        return "Victory"

    if (index_tip.y < index_mcp.y and
            middle_tip.y > index_mcp.y and
            ring_tip.y > index_mcp.y and
            pinky_tip.y > index_mcp.y):
        return "Pointing"

    return "Unclassified"


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        mirrored_frame = cv2.flip(frame, 1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    lm.x = 1 - lm.x

                gesture = classify_gesture(hand_landmarks)
                send_signal_to_robot(mapa[gesture])

                mp_drawing.draw_landmarks(
                    mirrored_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                cv2.putText(mirrored_frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Recognition', mirrored_frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
