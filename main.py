from modules.face_classes import Person, FaceRecognizer
from modules.face_utils import load_face_encoding_from_image, webcam_capture_one_frame
import face_recognition
import cv2


def menu():
    print("\nFACIAL RECOGNITION SYSTEM")
    print("1. Register new face")
    print("2. Recognize face from image")
    print("3. Recognize face from webcam")
    print("4. Exit")


def main():
    recognizer = FaceRecognizer()

    while True:
        menu()
        choice = input("Enter choice: ")

    
        if choice == "1":
            name = input("Enter person's name: ")
            path = input("Enter path to face image: ")
            encoding = load_face_encoding_from_image(path)

            if encoding is None:
                print("No face found in image.")
                continue

            recognizer.add_person(Person(name, encoding))
            print(f"{name} successfully registered!")

        elif choice == "2":
            path = input("Enter path to unknown face image: ")
            frame_encoding = load_face_encoding_from_image(path)

            if frame_encoding is None:
                print("No face detected.")
                continue

            result = recognizer.recognize(frame_encoding)
            print("Recognition result:", result)

        elif choice == "3":
            print("Capturing webcam frame...")
            frame = webcam_capture_one_frame()

            if frame is None:
                print("Could not capture frame.")
                continue

            rgb_frame = frame[:, :, ::-1]
            encodings = face_recognition.face_encodings(rgb_frame)

            if not encodings:
                print("No face detected from webcam.")
                continue

            result = recognizer.recognize(encodings[0])
            print("Recognition result:", result)

            cv2.imshow("Captured Frame", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        elif choice == "4":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
