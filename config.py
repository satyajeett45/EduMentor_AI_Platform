#config.py 
import os
from datetime import datetime

# -------------------------------
# Feature Configuration
# -------------------------------

NUMERICAL_FEATURES = [
    'math_grade', 'english_grade', 'science_grade', 'history_grade', 'overall_grade',
'assignment_completion', 'engagement_score', 'math_lec_present', 'science_lec_present',
'history_lec_present', 'english_lec_present', 'attendance_ratio', 'login_frequency_per_week',
'average_session_duration_minutes', 'completed_lessons', 'practice_tests_taken', 'lms_test_scores'

]

CATEGORICAL_FEATURES = [
    'learning_style',
    'content_type_preference',
    'teacher_comments_summary'
]

TARGET_COLUMN = 'risk_score'
CLASSIFICATION_COLUMN = 'is_at_risk'

# -------------------------------
# File Paths
# -------------------------------

MODEL_PATH = 'artifacts/best_model.pkl'
REGRESSOR_MODEL_PATH =  'artifacts/best_regression_model.pkl'
CLASSIFIER_MODEL_PATH = 'artifacts/best_classification_model.pkl'
SCALER_PATH = 'artifacts/scaler.pkl'
ENCODERS_PATH = 'artifacts/label_encoders.pkl'
STUDENT_DATA_PATH = r'C:\Users\Admin\OneDrive\Desktop\Edumentor platform\Data\edu_mentor_dataset_final.csv'
TEACHERS_CSV = "Data/users.csv"
FEEDBACK_FILE = "Data/teacher_feedback.csv"

# -------------------------------
# LLM & API Keys (Set via .env or environment)
# -------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# -------------------------------
# Thresholds & Flags
# -------------------------------

RISK_THRESHOLD = 30       # Below this = low risk
CLASSIFICATION_THRESHOLD = 0.5   # Probability threshold for binary output
SUGGESTIONS_ENABLED = True