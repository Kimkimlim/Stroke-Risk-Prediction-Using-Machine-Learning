import pickle
import numpy as np

print("Enter the following information to determine if you are at risk of a stroke.\n")

age = float(input("Enter age: "))
hypertension = int(input("Enter hypertension [0 for 'NO', 1 for 'YES']: "))
heart_disease = int(input("Enter heart disease [0 for 'NO', 1 for 'YES']: "))
ever_married = int(input("Enter marital status [0 for 'NO', 1 for 'YES']: "))
avg_glucose_level = float(input("Enter average glucose level: "))
bmi = float(input("Enter BMI: "))
gender = input("Enter gender (female, male, or other): ")
work_type = input("Enter work type (government, private, self-employed, children, or never worked): ")
residence_type = input("Enter residence type (rural or urban): ")
smoking_status = input("Enter smoking status (formerly smoked, never smoked, or smokes): ")

user_data = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'female': gender == "female",
    'male': gender == "male",
    'Other': gender == "other",
    'government_work': work_type == "government",
    'private_work': work_type == "private",
    'self_employed': work_type == "self-employed",
    'children_work': work_type == "children",
    'never_worked': work_type == "never worked",
    'rural_resident': residence_type == "rural",
    'urban_resident': residence_type == "urban",
    'formerly_smoked': smoking_status == "formerly smoked",
    'never_smoked': smoking_status == "never smoked",
    'smokes': smoking_status == "smokes"
}


with open('Models/RandomForest.pkl', 'rb') as file:
    predictor = pickle.load(file)

X_test = np.array([list(user_data.values())])

# Make predictions
predictions = predictor.predict(X_test)

# Predictions

if predictions[0] == 1:
    print("\n[=] You are at risk of a stroke!")
else:
    print("\n[=] You are not at risk of a stroke.")
