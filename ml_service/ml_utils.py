import joblib
import pandas as pd

def calculate_risk_score(row):
    # Ensure values are numbers (Node.js might send them as strings)
    # We use a helper to prevent errors if a key is missing
    def get_val(key, default=0):
        try:
            return float(row.get(key, default))
        except (ValueError, TypeError):
            return default

    risk_score = 0
    
    # 1. Grade-based risk
    risk_score += (10 - get_val('cu_1_grade')) * 5
    risk_score += (10 - get_val('cu_2_grade')) * 5
    
    # 2. Unevaluated units
    risk_score += get_val('cu_1_no_eval') * 2
    risk_score += get_val('cu_2_no_eval') * 2
    
    # 3. Approval rates
    cu1_enrolled = get_val('cu_1_enrolled')
    if cu1_enrolled > 0:
        approval_rate_1 = get_val('cu_1_approved') / cu1_enrolled
        risk_score += (1 - approval_rate_1) * 15
    
    cu2_enrolled = get_val('cu_2_enrolled')
    if cu2_enrolled > 0:
        approval_rate_2 = get_val('cu_2_approved') / cu2_enrolled
        risk_score += (1 - approval_rate_2) * 15
    
    # 4. Binary status flags (1 = Yes, 0 = No)
    if get_val('debtor') == 1: risk_score += 10
    if get_val('tuition_paid') == 0: risk_score += 10
    if get_val('scholarship') == 0: risk_score += 8
    if get_val('attendance_type') == 1: risk_score += 5
    if get_val('displaced') == 1: risk_score += 5
    if get_val('international') == 1: risk_score += 5

    # 5. Age logic
    age = get_val('age_enrollment')
    if age > 30:
        risk_score += 5
    elif age > 25:
        risk_score += 3
    
    return min(risk_score, 100)

# Optional: Add a function to load the model easily
def load_saved_model():
    try:
        model = joblib.load('models/dropout_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None