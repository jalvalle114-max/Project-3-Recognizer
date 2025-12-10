import face_recognition
from PIL import Image

def load_face_encoding_from_image(image_path):
    try:
        img = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) == 0:
            print(f"No faces found in {image_path}")
            return None, None
        return Image.fromarray(img), encodings[0]
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None, None
