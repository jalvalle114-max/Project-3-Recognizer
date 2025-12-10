import pickle
import os


DB_PATH = "face_database.pkl"


def save_database(people_list):
    with open(DB_PATH, "wb") as f:
        pickle.dump(people_list, f)
    print(f" Database saved to {DB_PATH}")


def load_database():
    if not os.path.isfile(DB_PATH):
        return []

    with open(DB_PATH, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} faces from database.")
    return data
