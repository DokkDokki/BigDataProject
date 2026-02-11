import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from load_data import load_data

df = load_data()

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
