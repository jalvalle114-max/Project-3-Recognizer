import numpy as np
from modules.face_classes import Person, FaceDatabase


def test_person_creation():
    encoding = np.zeros(128)
    person = Person("Alice", encoding)
    assert person.name == "Alice"


def test_database_add_person():
    db = FaceDatabase()
    encoding = np.zeros(128)
    db.add_person(Person("Bob", encoding))
    assert len(db) == 1
