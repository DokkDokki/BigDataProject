import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from dotenv import load_dotenv
import certifi
from pymongo import MongoClient
import os

load_dotenv()
uri = os.getenv("MONGO_URI")

client = MongoClient(
    uri,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000
)

db = client["student_dropout_db"]
collection = db["students"]

df = pd.DataFrame(list(collection.find()))

# Dropped useless columns
df = df.drop(columns=[ 'Inflation rate', 'Unemployment rate', 'GDP'])

# Renamed columns
rename_map = {
    'Marital status': 'marital_status',
    'Application mode': 'application_mode',
    'Application order': 'application_order',
    'Course': 'course',
    'Daytime/evening attendance': 'attendance_type',
    'Previous qualification': 'prev_qualification',
    'Nacionality': 'nationality',
    "Mother's qualification": 'mother_qualification',
    "Father's qualification": 'father_qualification',
    "Mother's occupation": 'mother_occupation',
    "Father's occupation": 'father_occupation',
    'Displaced': 'displaced',
    'Educational special needs': 'special_needs',
    'Debtor': 'debtor',
    'Tuition fees up to date': 'tuition_paid',
    'Gender': 'gender',
    'Scholarship holder': 'scholarship',
    'Age at enrollment': 'age_enrollment',
    'International': 'international',

    # 1st semester
    'Curricular units 1st sem (credited)': 'cu_1_credited',
    'Curricular units 1st sem (enrolled)': 'cu_1_enrolled',
    'Curricular units 1st sem (evaluations)': 'cu_1_evaluations',
    'Curricular units 1st sem (approved)': 'cu_1_approved',
    'Curricular units 1st sem (grade)': 'cu_1_grade',
    'Curricular units 1st sem (without evaluations)': 'cu_1_no_eval',

    # 2nd semester
    'Curricular units 2nd sem (credited)': 'cu_2_credited',
    'Curricular units 2nd sem (enrolled)': 'cu_2_enrolled',
    'Curricular units 2nd sem (evaluations)': 'cu_2_evaluations',
    'Curricular units 2nd sem (approved)': 'cu_2_approved',
    'Curricular units 2nd sem (grade)': 'cu_2_grade',
    'Curricular units 2nd sem (without evaluations)': 'cu_2_no_eval',
    
    # Target
    'Target': 'target',
}

df = df.rename(columns = rename_map)

# Encoded 'target'
label_encoder = LabelEncoder()
df['target_encoded'] = label_encoder.fit_transform(df['target'])

cat_cols = [
    'marital_status',
    'application_mode',
    'application_order',
    'course',
    'attendance_type',
    'prev_qualification',
    'nationality',
    'mother_qualification',
    'father_qualification',
    'mother_occupation',
    'father_occupation',
    'displaced',
    'special_needs',
    'debtor',
    'tuition_paid',
    'gender',
    'scholarship',
    'international',
]

num_cols = [
    'age_enrollment',

    # 1st Semester
    'cu_1_credited',
    'cu_1_enrolled',
    'cu_1_evaluations',
    'cu_1_approved',
    'cu_1_grade',
    'cu_1_no_eval',

    # 2nd Semester
    'cu_2_credited',
    'cu_2_enrolled',
    'cu_2_evaluations',
    'cu_2_approved',
    'cu_2_grade',
    'cu_2_no_eval',
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

x = df[cat_cols + num_cols]
y = df['target_encoded']

x_processed = preprocessor.fit_transform(x)

#Step 3
# x_train, x_test, y_train, y_test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size=0.2, random_state=42)

from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
# accuracy and precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")

# F1 score and confusion matrix
f1 = f1_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)

print(f"F1-score: {f1}")
print("Confusion matrix:")
print(cm)

# Visualizations
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#Histogram of predictions
axes[0, 0].hist(y_pred, bins=len(np.unique(y_pred)), edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Histogram of Predictions')
axes[0, 0].set_xlabel('Predicted Class')
axes[0, 0].set_ylabel('Frequency')

#Bar chart of metrics
metrics = ['Accuracy', 'Precision', 'F1-score']
values = [accuracy, precision, f1]
axes[0, 1].bar(metrics, values, color=['blue', 'green', 'red'], alpha=0.7)
axes[0, 1].set_title('Model Metrics')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_ylim([0, 1])

#Pie chart of true labels distribution
unique_labels, counts = np.unique(y_test, return_counts=True)
axes[1, 0].pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('True Labels Distribution')

#Confusion matrix heatmap
im = axes[1, 1].imshow(cm, cmap='Blues', aspect='auto')
axes[1, 1].set_title('Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')

#Add text annotations to confusion matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

fig.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.show()

# Calculate Risk Score (0-100, higher = more at-risk)
def calculate_risk_score(row):
    risk_score = 0
    
    # Grade-based risk (most important)
    risk_score += (10 - row['cu_1_grade']) * 5  # Low 1st sem grade
    risk_score += (10 - row['cu_2_grade']) * 5  # Low 2nd sem grade
    
    # Unevaluated units (high risk)
    risk_score += row['cu_1_no_eval'] * 2
    risk_score += row['cu_2_no_eval'] * 2
    
    # Approval rates
    if row['cu_1_enrolled'] > 0:
        approval_rate_1 = row['cu_1_approved'] / row['cu_1_enrolled']
        risk_score += (1 - approval_rate_1) * 15
    
    if row['cu_2_enrolled'] > 0:
        approval_rate_2 = row['cu_2_approved'] / row['cu_2_enrolled']
        risk_score += (1 - approval_rate_2) * 15
    
    # Debtor status
    if row['debtor'] == 1:
        risk_score += 10
    
    # Age (older = slightly higher risk)
    if row['age_enrollment'] > 30:
        risk_score += 5
    elif row['age_enrollment'] > 25:
        risk_score += 3
    
    # Attendance type (evening = higher risk)
    if row['attendance_type'] == 1:  # Usually 1 = evening
        risk_score += 5
    
    # Displaced status
    if row['displaced'] == 1:
        risk_score += 5
    
    # International status
    if row['international'] == 1:
        risk_score += 5
    
    # No scholarship (higher risk)
    if row['scholarship'] == 0:
        risk_score += 8
    
    # Tuition not paid
    if row['tuition_paid'] == 0:
        risk_score += 10
    
    return min(risk_score, 100)  # Cap at 100

df['risk_score'] = df.apply(calculate_risk_score, axis=1)
df['success_score'] = 100 - df['risk_score']

print("=" * 80)
print("TOP 10 STUDENTS AT RISK OF DROPPING")
print("=" * 80)

at_risk = df.nlargest(10, 'risk_score')[['risk_score', 'target', 'cu_1_grade', 'cu_2_grade', 'age_enrollment', 'debtor', 'scholarship', 'tuition_paid']]
at_risk['rank'] = range(1, 11)

for idx, (i, row) in enumerate(at_risk.iterrows(), 1):
    print(f"\n{idx}. Risk Score: {row['risk_score']:.1f}/100")
    print(f"   Status: {row['target']}")
    print(f"   1st Sem Grade: {row['cu_1_grade']:.2f}, 2nd Sem Grade: {row['cu_2_grade']:.2f}")
    print(f"   Age: {row['age_enrollment']}, Debtor: {row['debtor']}, Scholarship: {row['scholarship']}")
    print(f"   Tuition Paid: {row['tuition_paid']}")

print("\n" + "=" * 80)
print("TOP 10 STUDENTS WHO DON'T NEED TO DROP (Low Risk)")
print("=" * 80)

doing_well = df.nsmallest(10, 'risk_score')[['risk_score', 'target', 'cu_1_grade', 'cu_2_grade', 'age_enrollment', 'debtor', 'scholarship', 'tuition_paid']]

for idx, (i, row) in enumerate(doing_well.iterrows(), 1):
    print(f"\n{idx}. Risk Score: {row['risk_score']:.1f}/100")
    print(f"   Status: {row['target']}")
    print(f"   1st Sem Grade: {row['cu_1_grade']:.2f}, 2nd Sem Grade: {row['cu_2_grade']:.2f}")
    print(f"   Age: {row['age_enrollment']}, Debtor: {row['debtor']}, Scholarship: {row['scholarship']}")
    print(f"   Tuition Paid: {row['tuition_paid']}")

print("\n" + "=" * 80)

# TREND ANALYSIS OVER TIME (1st Semester vs 2nd Semester)
print("\n" + "=" * 80)
print("TREND ANALYSIS: SEMESTER 1 vs SEMESTER 2")
print("=" * 80)

# Calculate trends for each student
df['grade_trend'] = df['cu_2_grade'] - df['cu_1_grade']  # Positive = improvement
df['approval_trend'] = (df['cu_2_approved'] / (df['cu_2_enrolled'] + 1)) - (df['cu_1_approved'] / (df['cu_1_enrolled'] + 1))
df['enrollment_trend'] = df['cu_2_enrolled'] - df['cu_1_enrolled']

# Overall statistics
avg_sem1_grade = df['cu_1_grade'].mean()
avg_sem2_grade = df['cu_2_grade'].mean()
avg_sem1_approval = (df['cu_1_approved'] / (df['cu_1_enrolled'] + 1)).mean()
avg_sem2_approval = (df['cu_2_approved'] / (df['cu_2_enrolled'] + 1)).mean()

print(f"\nOVERALL TRENDS:")
print(f"Average Grade Sem 1: {avg_sem1_grade:.2f} → Sem 2: {avg_sem2_grade:.2f}")
print(f"Average Approval Rate Sem 1: {avg_sem1_approval:.2%} → Sem 2: {avg_sem2_approval:.2%}")

# Students with positive trend (improving)
print(f"\n" + "-" * 80)
print("TOP 10 STUDENTS WITH POSITIVE GRADE TREND (Improving)")
print("-" * 80)

improving = df.nlargest(10, 'grade_trend')[['cu_1_grade', 'cu_2_grade', 'grade_trend', 'target']]

for idx, (i, row) in enumerate(improving.iterrows(), 1):
    print(f"\n{idx}. Grade Trend: {row['grade_trend']:+.2f}")
    print(f"   Sem 1: {row['cu_1_grade']:.2f} → Sem 2: {row['cu_2_grade']:.2f}")
    print(f"   Status: {row['target']}")

# Students with negative trend (declining)
print(f"\n" + "-" * 80)
print("TOP 10 STUDENTS WITH NEGATIVE GRADE TREND (Declining)")
print("-" * 80)

declining = df.nsmallest(10, 'grade_trend')[['cu_1_grade', 'cu_2_grade', 'grade_trend', 'target']]

for idx, (i, row) in enumerate(declining.iterrows(), 1):
    print(f"\n{idx}. Grade Trend: {row['grade_trend']:+.2f}")
    print(f"   Sem 1: {row['cu_1_grade']:.2f} → Sem 2: {row['cu_2_grade']:.2f}")
    print(f"   Status: {row['target']}")

# Trend visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Grade trend distribution
axes[0, 0].hist(df['grade_trend'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].set_title('Distribution of Grade Trends (Sem2 - Sem1)')
axes[0, 0].set_xlabel('Grade Change')
axes[0, 0].set_ylabel('Number of Students')
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
axes[0, 0].legend()

# Average grades by semester
semesters = ['Semester 1', 'Semester 2']
avg_grades = [avg_sem1_grade, avg_sem2_grade]
axes[0, 1].bar(semesters, avg_grades, color=['lightcoral', 'lightgreen'], alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Average Grade by Semester')
axes[0, 1].set_ylabel('Average Grade')
axes[0, 1].set_ylim([0, 10])

# Trend by target status
target_grades_sem1 = df.groupby('target')['cu_1_grade'].mean()
target_grades_sem2 = df.groupby('target')['cu_2_grade'].mean()
x_pos = np.arange(len(target_grades_sem1))
width = 0.35
axes[1, 0].bar(x_pos - width/2, target_grades_sem1, width, label='Semester 1', alpha=0.7)
axes[1, 0].bar(x_pos + width/2, target_grades_sem2, width, label='Semester 2', alpha=0.7)
axes[1, 0].set_xlabel('Target Status')
axes[1, 0].set_ylabel('Average Grade')
axes[1, 0].set_title('Grade Trends by Student Status')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(target_grades_sem1.index)
axes[1, 0].legend()

# Approval rate trend by target status
approval_sem1_by_target = (df.groupby('target')['cu_1_approved'].sum() / (df.groupby('target')['cu_1_enrolled'].sum() + 1))
approval_sem2_by_target = (df.groupby('target')['cu_2_approved'].sum() / (df.groupby('target')['cu_2_enrolled'].sum() + 1))
x_pos = np.arange(len(approval_sem1_by_target))
axes[1, 1].bar(x_pos - width/2, approval_sem1_by_target, width, label='Semester 1', alpha=0.7, color='orange')
axes[1, 1].bar(x_pos + width/2, approval_sem2_by_target, width, label='Semester 2', alpha=0.7, color='purple')
axes[1, 1].set_xlabel('Target Status')
axes[1, 1].set_ylabel('Approval Rate')
axes[1, 1].set_title('Approval Rate Trends by Student Status')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(approval_sem1_by_target.index)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)

