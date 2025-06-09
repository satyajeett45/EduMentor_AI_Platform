# inference.py

import joblib
import pandas as pd
from config import (
    REGRESSOR_MODEL_PATH, CLASSIFIER_MODEL_PATH, SCALER_PATH,
    ENCODERS_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from preprocessing import preprocess_data
from explaination import generate_explanation  # we'll build this next

# Load models and preprocessing tools
regressor = joblib.load(REGRESSOR_MODEL_PATH)
classifier = joblib.load(CLASSIFIER_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

EXPECTED_FEATURE_ORDER = [
    'math_grade', 'english_grade', 'science_grade', 'history_grade', 'overall_grade',
    'assignment_completion', 'engagement_score', 'math_lec_present', 'science_lec_present',
    'history_lec_present', 'english_lec_present', 'attendance_ratio', 'login_frequency_per_week',
    'average_session_duration_minutes', 'completed_lessons', 'practice_tests_taken', 'lms_test_scores',
    'learning_style', 'content_type_preference', 'teacher_comments_summary'
]

def predict_student_risk(student_row: pd.Series) -> dict:
    df = pd.DataFrame([student_row])
    
    # Preprocess
    processed = preprocess_data(df, fit=False)
     # Drop non-feature columns
    processed = processed.drop(
        columns=["student_name", "email_id", "password", "suggestions_required",
                 "student_id", "risk_score", "is_at_risk", "std"],
        errors="ignore"
    )
    processed = processed[EXPECTED_FEATURE_ORDER]
    risk_score = regressor.predict(processed)[0]
    risk_score = round(risk_score, 2)

    # Predict binary classification (is_at_risk)
    is_at_risk = classifier.predict(processed)[0]
    is_at_risk_label = "At-Risk" if is_at_risk == 1 else "Not At-Risk"
    confidence = max(classifier.predict_proba(processed)[0]) * 100

    prediction_data = {
    "risk_score": risk_score,
    "classification": is_at_risk_label,
    "confidence": f"{confidence:.1f}%"
    }
    explanation, suggestions = generate_explanation(student_row.to_dict(), prediction_data)

    return {
        "student_id": student_row.get("student_id"),
        "risk_score": (risk_score),
        "is_at_risk": (is_at_risk_label),
        "explanation": explanation,
        "suggestions": suggestions
    }
