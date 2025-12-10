import os
import cv2
import face_recognition


def image_exists(path: str) -> bool:
    return os.path.isfile(path)


def load_face_encoding_from_image(image_path: str):
    if not image_exists(image_path):
        print(f"ERROR: Image path does not exist → {image_path}")
        return None

    try:
        img = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return None

    encodings = face_recognition.face_encodings(img)

    if not encodings:
        print("No face detected in the image.")
        return None

    return encodings[0]


def webcam_available() -> bool:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False
    cap.release()
    return True


def webcam_capture_one_frame():
    if not webcam_available():
        print("Webcam is NOT available on this system.")
        return None

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not capture a frame from the webcam.")
        return None

    return frame