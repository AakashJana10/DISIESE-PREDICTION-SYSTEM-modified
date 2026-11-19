"""
create_dataset.py - Generates data/Training.csv and data/Testing.csv with realistic symptom-disease mappings.
100% standard library (random, csv, os) - fast, reliable, and zero external binary dependencies.
"""
import random
import csv
import os

random.seed(42)

SYMPTOMS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
    "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
    "vomiting", "burning_micturition", "spotting_ urination", "fatigue", "weight_gain",
    "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
    "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
    "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes",
    "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever",
    "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload",
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision",
    "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose",
    "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity",
    "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid",
    "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts",
    "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain",
    "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness",
    "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side",
    "loss_of_smell", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine",
    "passage_of_gases", "internal_itching", "toxic_look_(typhos)", "depression",
    "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain",
    "abnormal_menstruation", "dischromic _patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum",
    "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload.1", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples",
    "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
]

assert len(SYMPTOMS) == 132, f"Expected 132 symptoms, got {len(SYMPTOMS)}"

DISEASE_PROFILES = {
    "Fungal infection": {
        "primary":    ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic _patches"],
        "secondary":  ["fatigue", "lethargy", "sweating"],
        "secondary_p": 0.5,
    },
    "Allergy": {
        "primary":    ["continuous_sneezing", "shivering", "runny_nose", "watering_from_eyes", "redness_of_eyes"],
        "secondary":  ["fatigue", "cough", "throat_irritation", "congestion", "sinus_pressure", "headache"],
        "secondary_p": 0.6,
    },
    "GERD": {
        "primary":    ["acidity", "indigestion", "stomach_pain", "ulcers_on_tongue"],
        "secondary":  ["vomiting", "cough", "chest_pain", "fatigue", "loss_of_appetite"],
        "secondary_p": 0.5,
    },
    "Chronic cholestasis": {
        "primary":    ["itching", "fatigue", "loss_of_appetite", "yellowish_skin", "dark_urine", "yellowing_of_eyes"],
        "secondary":  ["vomiting", "nausea", "abdominal_pain", "constipation"],
        "secondary_p": 0.5,
    },
    "Drug Reaction": {
        "primary":    ["itching", "skin_rash", "swelled_lymph_nodes", "fatigue"],
        "secondary":  ["vomiting", "nausea", "diarrhoea", "headache", "blurred_and_distorted_vision"],
        "secondary_p": 0.5,
    },
    "Peptic ulcer disease": {
        "primary":    ["vomiting", "indigestion", "abdominal_pain", "loss_of_appetite"],
        "secondary":  ["fatigue", "nausea", "passage_of_gases", "headache", "diarrhoea"],
        "secondary_p": 0.55,
    },
    "AIDS": {
        "primary":    ["muscle_wasting", "fatigue", "high_fever", "extra_marital_contacts", "receiving_blood_transfusion", "receiving_unsterile_injections"],
        "secondary":  ["weight_loss", "sweating", "swelled_lymph_nodes", "patches_in_throat", "diarrhoea"],
        "secondary_p": 0.6,
    },
    "Diabetes": {
        "primary":    ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "excessive_hunger", "polyuria", "increased_appetite"],
        "secondary":  ["blurred_and_distorted_vision", "obesity", "family_history", "sweating"],
        "secondary_p": 0.5,
    },
    "Gastroenteritis": {
        "primary":    ["vomiting", "diarrhoea", "abdominal_pain", "dehydration", "nausea"],
        "secondary":  ["fatigue", "mild_fever", "stomach_pain", "sunken_eyes"],
        "secondary_p": 0.55,
    },
    "Bronchial Asthma": {
        "primary":    ["cough", "breathlessness", "chest_pain", "phlegm", "fatigue"],
        "secondary":  ["family_history", "mucoid_sputum", "sweating", "weakness_in_limbs"],
        "secondary_p": 0.5,
    },
    "Hypertension": {
        "primary":    ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
        "secondary":  ["fatigue", "obesity", "sweating", "fast_heart_rate"],
        "secondary_p": 0.5,
    },
    "Migraine": {
        "primary":    ["headache", "nausea", "blurred_and_distorted_vision", "fatigue"],
        "secondary":  ["fatigue", "vomiting", "loss_of_appetite", "dizziness"],
        "secondary_p": 0.55,
    },
    "Cervical spondylosis": {
        "primary":    ["back_pain", "neck_pain", "weakness_in_limbs", "dizziness"],
        "secondary":  ["fatigue", "loss_of_balance", "headache", "stiff_neck"],
        "secondary_p": 0.55,
    },
    "Paralysis (brain hemorrhage)": {
        "primary":    ["vomiting", "weakness_of_one_body_side", "altered_sensorium", "slurred_speech", "loss_of_balance"],
        "secondary":  ["headache", "dizziness", "stiff_neck", "blurred_and_distorted_vision"],
        "secondary_p": 0.5,
    },
    "Jaundice": {
        "primary":    ["itching", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "yellowing_of_eyes", "loss_of_appetite"],
        "secondary":  ["nausea", "abdominal_pain", "dehydration", "constipation", "yellow_urine"],
        "secondary_p": 0.6,
    },
    "Malaria": {
        "primary":    ["chills", "high_fever", "sweating", "headache", "nausea", "muscle_pain"],
        "secondary":  ["vomiting", "diarrhoea", "fatigue", "dehydration", "shivering"],
        "secondary_p": 0.6,
    },
    "Chicken pox": {
        "primary":    ["itching", "skin_rash", "fatigue", "mild_fever", "swelled_lymph_nodes"],
        "secondary":  ["headache", "loss_of_appetite", "malaise", "red_spots_over_body", "blister"],
        "secondary_p": 0.6,
    },
    "Dengue": {
        "primary":    ["high_fever", "headache", "joint_pain", "pain_behind_the_eyes", "nausea", "fatigue", "red_spots_over_body"],
        "secondary":  ["vomiting", "sweating", "malaise", "loss_of_appetite", "back_pain"],
        "secondary_p": 0.6,
    },
    "Typhoid": {
        "primary":    ["high_fever", "headache", "nausea", "loss_of_appetite", "abdominal_pain", "toxic_look_(typhos)"],
        "secondary":  ["vomiting", "fatigue", "diarrhoea", "constipation", "belly_pain"],
        "secondary_p": 0.6,
    },
    "hepatitis A": {
        "primary":    ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite"],
        "secondary":  ["abdominal_pain", "mild_fever", "yellowing_of_eyes", "malaise"],
        "secondary_p": 0.55,
    },
    "Hepatitis B": {
        "primary":    ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "yellowing_of_eyes", "receiving_blood_transfusion"],
        "secondary":  ["vomiting", "nausea", "abdominal_pain", "malaise", "receiving_unsterile_injections"],
        "secondary_p": 0.55,
    },
    "Hepatitis C": {
        "primary":    ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "yellowing_of_eyes", "receiving_blood_transfusion"],
        "secondary":  ["vomiting", "abdominal_pain", "dark_urine", "malaise", "receiving_unsterile_injections"],
        "secondary_p": 0.5,
    },
    "Hepatitis D": {
        "primary":    ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "yellowing_of_eyes", "receiving_blood_transfusion", "receiving_unsterile_injections"],
        "secondary":  ["nausea", "loss_of_appetite", "abdominal_pain", "malaise"],
        "secondary_p": 0.55,
    },
    "Hepatitis E": {
        "primary":    ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea"],
        "secondary":  ["loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "malaise", "diarrhoea"],
        "secondary_p": 0.55,
    },
    "Alcoholic hepatitis": {
        "primary":    ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "history_of_alcohol_consumption"],
        "secondary":  ["fatigue", "nausea", "loss_of_appetite", "yellowing_of_eyes", "fluid_overload", "distention_of_abdomen"],
        "secondary_p": 0.55,
    },
    "Tuberculosis": {
        "primary":    ["cough", "fatigue", "weight_loss", "sweating", "high_fever", "phlegm", "blood_in_sputum"],
        "secondary":  ["chest_pain", "loss_of_appetite", "malaise", "breathlessness", "mild_fever", "mucoid_sputum"],
        "secondary_p": 0.6,
    },
    "Common Cold": {
        "primary":    ["continuous_sneezing", "shivering", "fatigue", "cough", "headache", "runny_nose", "congestion"],
        "secondary":  ["chills", "throat_irritation", "sinus_pressure", "mild_fever", "loss_of_smell"],
        "secondary_p": 0.6,
    },
    "Pneumonia": {
        "primary":    ["cough", "chest_pain", "breathlessness", "phlegm", "high_fever", "fatigue"],
        "secondary":  ["chills", "sweating", "malaise", "nausea", "rusty_sputum", "mucoid_sputum"],
        "secondary_p": 0.55,
    },
    "Dimorphic hemorrhoids": {
        "primary":    ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
        "secondary":  ["fatigue"],
        "secondary_p": 0.4,
    },
    "Heart attack": {
        "primary":    ["chest_pain", "sweating", "breathlessness", "fast_heart_rate", "vomiting"],
        "secondary":  ["fatigue", "nausea", "weakness_in_limbs", "palpitations", "anxiety"],
        "secondary_p": 0.6,
    },
    "Varicose veins": {
        "primary":    ["fatigue", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf", "painful_walking"],
        "secondary":  ["cramps", "bruising", "obesity", "swelling_joints"],
        "secondary_p": 0.5,
    },
    "Hypothyroidism": {
        "primary":    ["fatigue", "weight_gain", "cold_hands_and_feets", "constipation", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails"],
        "secondary":  ["mood_swings", "depression", "lethargy", "dizziness", "swollen_extremeties"],
        "secondary_p": 0.55,
    },
    "Hyperthyroidism": {
        "primary":    ["fatigue", "weight_loss", "restlessness", "fast_heart_rate", "sweating", "diarrhoea", "enlarged_thyroid"],
        "secondary":  ["mood_swings", "anxiety", "irritability", "neck_pain", "weakness_in_limbs"],
        "secondary_p": 0.55,
    },
    "Hypoglycemia": {
        "primary":    ["vomiting", "fatigue", "anxiety", "sweating", "headache", "blurred_and_distorted_vision", "slurred_speech"],
        "secondary":  ["irregular_sugar_level", "excessive_hunger", "drying_and_tingling_lips", "palpitations", "nausea"],
        "secondary_p": 0.6,
    },
    "Osteoarthritis": {
        "primary":    ["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "movement_stiffness"],
        "secondary":  ["obesity", "painful_walking", "back_pain", "muscle_weakness", "cramps"],
        "secondary_p": 0.55,
    },
    "Arthritis": {
        "primary":    ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "loss_of_balance", "unsteadiness"],
        "secondary":  ["joint_pain", "fatigue", "back_pain", "knee_pain"],
        "secondary_p": 0.55,
    },
    "(Vertigo) Paroymsal Positional Vertigo": {
        "primary":    ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"],
        "secondary":  ["dizziness", "fatigue", "blurred_and_distorted_vision"],
        "secondary_p": 0.5,
    },
    "Acne": {
        "primary":    ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
        "secondary":  ["fatigue", "mood_swings", "irregular_sugar_level"],
        "secondary_p": 0.4,
    },
    "Urinary tract infection": {
        "primary":    ["burning_micturition", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine"],
        "secondary":  ["fatigue", "mild_fever", "back_pain", "abdominal_pain"],
        "secondary_p": 0.5,
    },
    "Psoriasis": {
        "primary":    ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
        "secondary":  ["itching", "fatigue", "back_pain"],
        "secondary_p": 0.5,
    },
    "Impetigo": {
        "primary":    ["skin_rash", "blister", "red_sore_around_nose", "yellow_crust_ooze"],
        "secondary":  ["itching", "fatigue", "nodal_skin_eruptions"],
        "secondary_p": 0.45,
    },
    "Paroymsal Positional Vertigo": {
        "primary":    ["dizziness", "spinning_movements", "loss_of_balance", "unsteadiness", "vomiting"],
        "secondary":  ["nausea", "headache", "blurred_and_distorted_vision", "neck_pain"],
        "secondary_p": 0.5,
    },
}

def gen_rows(disease, profile, n_rows):
    rows = []
    primary = [s for s in profile["primary"] if s in SYMPTOMS]
    secondary = [s for s in profile.get("secondary", []) if s in SYMPTOMS]
    sec_p = profile.get("secondary_p", 0.5)
    
    for _ in range(n_rows):
        row = {s: 0 for s in SYMPTOMS}
        for s in primary:
            row[s] = 1 if random.random() > 0.05 else 0
        for s in secondary:
            row[s] = 1 if random.random() < sec_p else 0
        noise_syms = [s for s in SYMPTOMS if s not in primary and s not in secondary]
        noise_count = random.randint(0, 2)
        for s in random.sample(noise_syms, min(noise_count, len(noise_syms))):
            row[s] = 1
        row["prognosis"] = disease
        rows.append(row)
    return rows

all_train_rows = []
all_test_rows = []
for disease, profile in DISEASE_PROFILES.items():
    train_rows = gen_rows(disease, profile, 120)
    test_rows = gen_rows(disease, profile, 10)
    all_train_rows.extend(train_rows)
    all_test_rows.extend(test_rows)

random.shuffle(all_train_rows)
random.shuffle(all_test_rows)

cols = SYMPTOMS + ["prognosis"]

os.makedirs("data", exist_ok=True)

with open("data/Training.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(all_train_rows)

with open("data/Testing.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(all_test_rows)

print(f"Training set: {len(all_train_rows)} rows")
print(f"Testing set:  {len(all_test_rows)} rows")
print(f"Diseases:     {len(DISEASE_PROFILES)}")
print(f"Columns:      {len(cols)}")
print("Done! Saved data/Training.csv and data/Testing.csv successfully.")
