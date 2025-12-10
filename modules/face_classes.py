import face_recognition
import numpy as np
from modules.storage import save_database, load_database


class Person:

    def __init__(self, name: str, encoding: np.ndarray):
        self.name = name
        self.encoding = encoding

    def __repr__(self):
        return f"Person(name='{self.name}')"


class FaceDatabase:

    def __init__(self):
        self.people = load_database()

    def add_person(self, person: Person):
        self.people.append(person)
        save_database(self.people)

    def __len__(self):
        return len(self.people)


class FaceRecognizer(FaceDatabase):

    def recognize(self, unknown_encoding: np.ndarray, tolerance=0.45):
        if not self.people:
            return "Database empty"

        encodings = [p.encoding for p in self.people]
        names = [p.name for p in self.people]

        matches = face_recognition.compare_faces(encodings, unknown_encoding, tolerance)

        if True in matches:
            return names[matches.index(True)]
        return "Unknown"
