import cv2
import face_recognition

def load_face_encoding_from_image(image_path: str):
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return None

    return encodings[0]


def webcam_capture_one_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None