from modules.face_classes import Person, FaceRecognizer
from modules.face_utils import (
    load_face_encoding_from_image,
    webcam_available,
    webcam_capture_one_frame,
)
import face_recognition
import cv2
import os


def menu(webcam_enabled: bool):
    print("\n===== FACIAL RECOGNITION SYSTEM =====")
    print("1. Register new face")
    print("2. Recognize face from image")
    
    if webcam_enabled:
        print("3. Recognize face from webcam")
    else:
        print("3. (Webcam unavailable)")

    print("4. Exit")


def prompt_image_path(prompt_text="Enter path to image: "):
    while True:
        path = input(prompt_text).strip()
        if os.path.isfile(path):
            return path
        print("Invalid path. Try again.")


def main():
    recognizer = FaceRecognizer()
    webcam_enabled = webcam_available()

    if not webcam_enabled:
        print("\nNOTE: Webcam not detected. Running in image-only mode.")

    while True:
        menu(webcam_enabled)
        choice = input("Enter choice: ").strip()

        if choice == "1":
            name = input("Enter person's name: ").strip()
            image_path = prompt_image_path("Enter path to face image: ")

            encoding = load_face_encoding_from_image(image_path)
            if encoding is None:
                continue

            recognizer.add_person(Person(name, encoding))
            print(f"✔ {name} successfully registered!")


        elif choice == "2":
            image_path = prompt_image_path("Enter path to unknown face image: ")
            encoding = load_face_encoding_from_image(image_path)

            if encoding is None:
                continue

            result = recognizer.recognize(encoding)
            print(f"Recognition Result → {result}")

        elif choice == "3":
            if not webcam_enabled:
                print("Webcam not available. Choose option 2 instead.")
                continue

            print("Capturing frame...")
            frame = webcam_capture_one_frame()

            if frame is None:
                continue

            rgb = frame[:, :, ::-1]
            encodings = face_recognition.face_encodings(rgb)

            if not encodings:
                print("No face detected in webcam frame.")
                continue

            result = recognizer.recognize(encodings[0])
            print(f"Recognition Result → {result}")

            cv2.imshow("Webcam Frame", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")
