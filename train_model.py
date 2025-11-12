from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import missingno as msno
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings(action='ignore')


# ---------------- Data Preprocessing ----------------

# Dataset Information
df = pd.read_csv('stroke_data.csv')
print("Dataset shape:", df.shape)

print("\nDataset Information:")
print(df.info())
print("--------------------------------------- Dataset Description:--------------------------------------")
print(df.describe())

# missing values in each column of the DataFrame
print(df.isna().sum())

# Droping unnecessary columns
df.drop(columns=['id'], axis=1, inplace=True)

# removing null values and handling missing data
print("Shape of dataframe before dropping:", df.shape)
df = df.dropna(axis=0, subset=['smoking_status'])
print("Shape after dropping:", df.shape)

# Imputing missing values with median
df['bmi'].fillna(df['bmi'].median(), inplace=True)
print(df['bmi'].isna().sum())

# ---------- Encoding categorical variables ----------
df = df.join(pd.get_dummies(df['gender']))
df.drop(columns=['gender'], inplace=True)
df.rename(columns={'Female': 'female', 'Male': 'male'}, inplace=True)

df = df.join(pd.get_dummies(df['work_type']))
df.drop(columns=['work_type'], inplace=True)
df.rename(columns={
    'Private': 'private_work', 'Self-employed': 'self_employed',
    'Govt_job': 'government_work', 'children': 'children_work',
    'Never_worked': 'never_worked'}, inplace=True)

df = df.join(pd.get_dummies(df['Residence_type']))
df.drop(columns=['Residence_type'], inplace=True)
df.rename(columns={'Urban': 'urban_resident',
                   'Rural': 'rural_resident'}, inplace=True)

df = df.join(pd.get_dummies(df['smoking_status']))
df.drop(columns=['smoking_status'], inplace=True)
df.rename(columns={'formerly smoked': 'formerly_smoked',
                   'never smoked': 'never_smoked',
                   'Unknown': 'smoking_unknown'}, inplace=True)
df['ever_married'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['ever_married'].dtype
df.head().T

# ---------- Correlation Matrix ----------
# corr = df.corr()
# plt.figure(figsize=(15, 10))
# plt.title("Correlation Matrix")
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()

# Checking Imabalanced Dataset
print(df['stroke'].value_counts())

# Oversampling the minority class
X = df.drop(columns=['stroke'])
y = df['stroke']

categorical_features = [
    'hypertension',
    'heart_disease',
    'ever_married',
    'female',
    'male',
    'government_work',
    'never_worked',
    'private_work',
    'self_employed',
    'children_work',
    'rural_resident',
    'urban_resident',
    'smoking_unknown',
    'formerly_smoked',
    'never_smoked',
    'smokes'
]


categorical_features_indices = []
for feature in X.columns:
    if feature in categorical_features:
        categorical_features_indices.append(True)
    else:
        categorical_features_indices.append(False)

smote = SMOTENC(categorical_features=categorical_features_indices)


X_resampled, y_resampled = smote.fit_resample(X, y)

print("Before OverSampling, counts of label '1': {}".format(sum(y == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y == 0)))

print("After OverSampling, counts of label '1': {}".format(sum(y_resampled == 1)))
print("After OverSampling, counts of label '0': {}\n".format(sum(y_resampled == 0)))

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------ Model Training ------------------
print("\n------------------ Training Models ------------------")

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("\nLogistic Regression:")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred) * 100))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("\nDecision Tree:")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred) * 100))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("\nRandom Forest:")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred) * 100))

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("\nKNN:")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred) * 100))

# Neural Network trained.
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print("\nNeural Network:")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred) * 100))

# Linear SVC
svc = LinearSVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("\nLinear SVC:")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred) * 100))

# SVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("\nSVC:")
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
print("F1 Score: {:.2f}%".format(f1_score(y_test, y_pred) * 100))

# save all the models according to their accuracy
pickle.dump(rf, open('Models/RandomForest.pkl', 'wb'))
pickle.dump(log_reg, open('Models/LogisticRegression.pkl', 'wb'))
pickle.dump(dt, open('Models/DecisionTree.pkl', 'wb'))
pickle.dump(knn, open('Models/KNN.pkl', 'wb'))
pickle.dump(mlp, open('Models/NeuralNetwork.pkl', 'wb'))
pickle.dump(svc, open('Models/SVC.pkl', 'wb'))
pickle.dump(svc, open('Models/LinearSVC.pkl', 'wb'))

print("\n[=] Models Trained Successfully!")
print("[=] Models saved successfully!")