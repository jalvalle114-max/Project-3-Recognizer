import face_recognition
import numpy as np


class Person:

    def __init__(self, name: str, encoding: np.ndarray):
        self.name = name
        self.encoding = encoding

    def __repr__(self):
        return f"Person(name='{self.name}')"


class FaceDatabase:

    def __init__(self):
        self.people = []

    def add_person(self, person: Person):
        self.people.append(person)

    def __len__(self):
        return len(self.people)


class FaceRecognizer(FaceDatabase):

    def recognize(self, unknown_encoding: np.ndarray, tolerance=0.45):
        
        known_encodings = [p.encoding for p in self.people]
        known_names = [p.name for p in self.people]

        if not known_encodings:
            return "No registered faces"

        matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance)

        if True in matches:
            matched_index = matches.index(True)
            return known_names[matched_index]

        return "Unknown"