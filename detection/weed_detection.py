import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('./models/weed.h5')

class_names_dict = {
    0: 'weed',
    1: 'clean',
}

def preprocess_image(image):
    image = cv2.resize(image, (100, 100))
    image = image / 255.0
    return image

def detect_objects():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_image = preprocess_image(frame)
        preprocessed_image_resized = cv2.resize(preprocessed_image, (139, 139))
        predictions = model.predict(np.expand_dims(preprocessed_image_resized, axis=0))
        class_indices = np.argmax(predictions, axis=1)
        class_names = [class_names_dict[i] for i in class_indices]

        print("Predicted class:", class_names[0])

        cv2.imshow('Webcam Input', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_objects()
