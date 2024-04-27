import json
import spacy

JSON_FILE_PATH = "./dataset.json"
MODEL_NAME = "en_core_web_sm"

COURSE_PREREQUISITES = {
    "frontend": ["Html", "Css", "JavaScript"],
    "backend": ["Python", "Django"],
    "data_science": ["Python", "Data Science", "Machine Learning"],
}

COURSE_RECOMMENDATIONS = {
    "frontend": ["Html", "Css", "JavaScript", "React", "Ajax"],
    "backend": ["SQL", "RESTful APIs"],
    "data_science": ["Statistics", "Data Visualization"],
}

def find_remaining_skills(data, board, prerequisite_skills, completed_skills):
    highest_class = None
    remaining_skills = set(prerequisite_skills)

    if board in data:
        for class_name, subjects in data[board].items():
            for subject_name, topics in subjects.items():
                for topic in topics:
                    doc = nlp(topic)

                    # Create a copy of the set to avoid RuntimeError
                    for skill in set(remaining_skills):
                        if skill.lower() in [token.text.lower() for token in doc]:
                            completed_skills.add(skill)
                            remaining_skills.remove(skill)

            if not remaining_skills:
                highest_class = class_name
                break  # No need to check further if all skills are completed

    return highest_class, completed_skills

def load_dataset(file_path):
    try:
        with open(file_path, 'r') as file:
            dataset = json.load(file)
        return dataset
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Unable to decode JSON from {file_path}")
        return None

def load_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Error: SpaCy model '{model_name}' not found. Make sure it's installed or download it using `python -m spacy download en_core_web_sm`.")
        return None

def print_results(course, prerequisite_skills, completed_skills, recommended_skills, uncovered_skills, highest_class):
    print(f"Highest class for {course} where all skills are matched: {highest_class}")
    print(f"Prerequisite skills: {prerequisite_skills}")
    print(f"Completed skills: {completed_skills}")

    if uncovered_skills:
        print(f"Uncovered recommended skills: {uncovered_skills}")
    else:
        print("All recommended skills are covered in the syllabus.")

def find_uncovered_skills(recommended_skills, completed_skills):
    return set(skill for skill in recommended_skills if skill not in completed_skills)

data = load_dataset(JSON_FILE_PATH)
if data is None:
    exit()

nlp = load_spacy_model(MODEL_NAME)
if nlp is None:
    exit()

course = input("Enter the course (e.g., frontend, backend, data_science): ").lower()

if course in COURSE_PREREQUISITES:
    prerequisite_skills = COURSE_PREREQUISITES[course]
    recommended_skills = COURSE_RECOMMENDATIONS.get(course, [])

    board = "CBSE"
    completed_skills = set()

    highest_class, completed_skills = find_remaining_skills(data, board, prerequisite_skills, completed_skills)

    if highest_class:
        uncovered_skills = find_uncovered_skills(recommended_skills, completed_skills)
        print_results(course, prerequisite_skills, completed_skills, recommended_skills, uncovered_skills, highest_class)
    else:
        print(f"No class found for {course} where all prerequisite skills are matched.")
else:
    print("Invalid course. Please check your input.")
