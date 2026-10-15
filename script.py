#1. Importation of the Py Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Generate Patient Record using the Seeded Data
np.random.seed(42)
n = 400 #Generate 400 Records

# Simulate the Features (X) => Age, Glucose_level, Steps, Body Mass
glucose = np.random.normal(loc=110, scale=25, size=n).round(2)
bmi = np.random.normal(loc=25, scale=5, size=n).round(2) #BMI (18.5, 40.2)
age = np.random.randint(20,80, size=n) #Age(20-80)
steps = np.random.randint(1500, 12000, size=n) #Daily Steps

# Make a Biological Risk Score
risk_score = (glucose * 0.04) + (bmi * 0.08) + (age * 0.03) - (steps * 0.0003)
# Obtaining the Target (Y)
y_target = (risk_score > 6.5).astype(int)

# Combine into a single Pandas DataFrame
df = pd.DataFrame({
    'Glucose_Level': glucose,
    'BMI': bmi,
    'Age': age,
    'Daily_Steps': steps,
    'Diabetes_Risk': y_target
})

print("---SAMPLE RAW DATASET")
print(df.head())

# Separete the inputs clues and the answers
X = df[['Glucose_Level', 'BMI', 'Age', 'Daily_Steps']]
y = df['Diabetes_Risk']

# Train/ Split
# 80 percent used to train the AI Doctor, 20 percent for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.20, random_state=42
)

# Train and Build the Rndiandom Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model Accuracy on this Test Data
y_pred = model.predict(X_test)

print(" --MODEL DIAGONISTICS PERFORMANCE----")
print(f"Overoll Accuracy is: {accuracy_score(y_test,y_pred)*100}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Model:\n", classification_report(y_test,y_pred))


joblib.dump(model, 'diabetes_model.joblib')
print("Model saved successfully saved as diabetes_model.joblib")








