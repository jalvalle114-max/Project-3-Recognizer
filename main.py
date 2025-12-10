from modules.face_classes import Person, FaceRecognizer
from modules.face_utils import load_face_encoding_from_image
from modules.storage import load_database, save_database

def main():
    recognizer = FaceRecognizer()
    recognizer.people = load_database() or []

    while True:
        print("\n===== FACIAL RECOGNITION SYSTEM =====")
        print("1. Register new face")
        print("2. Recognize face from image")
        print("3. Exit")
        choice = input("Enter choice: ")

        if choice == "1":
            name = input("Enter person's name: ")
            path = input("Enter image filename from images/ folder: ")
            full_path = f"images/{path}"
            _, encoding = load_face_encoding_from_image(full_path)
            if encoding is not None:
                recognizer.add_person(Person(name, encoding))
                save_database(recognizer.people)
                print(f"{name} registered successfully.")

        elif choice == "2":
            path = input("Enter unknown image filename from images/ folder: ")
            full_path = f"images/{path}"
            _, encoding = load_face_encoding_from_image(full_path)
            if encoding is not None:
                name = recognizer.recognize(encoding)
                print(f"Recognized: {name}")

        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
