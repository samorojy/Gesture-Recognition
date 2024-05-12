import cv2
import numpy as np
from tensorflow.keras.models import load_model


def load_gesture_model(model_path):
    return load_model(model_path)


def prepare_frame(frame):
    # Assuming you need to convert the frame to a different size or color space before prediction
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0)  # Expand dimensions if needed for the model


def predict_gesture(frame, model):
    processed_frame = prepare_frame(frame)
    prediction = model.predict(processed_frame)
    # Let's assume the model outputs an integer or a one-hot encoded array
    gesture_id = np.argmax(prediction)  # for one-hot encoded output
    return gesture_id  # Just an example; you might want to tailor this according to your model


def main():
    #model = load_gesture_model('path_to_your_model.h5')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gesture = "asfsa" #predict_gesture(frame, model)
        gesture_name = f'Gesture ID: {gesture}'  # Map to your gesture names or IDs

        cv2.putText(frame, gesture_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (10, 10), (220, 60), (255, 255, 255), -1)
        cv2.putText(frame, gesture_name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video Feed with Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
