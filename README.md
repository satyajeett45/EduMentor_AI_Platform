# EduMentor_AI_Platform

# 🎓 EduMentor: Student Risk Analysis and Feedback System

EduMentor is an AI-powered Streamlit web application designed for educational institutions to predict student dropout risk and assist teachers in monitoring and improving student performance through real-time feedback.

---

## 📌 Features

### ✅ Student Login
- Students can log in using their ID and password.
- View personalized:
  - Risk score
  - Risk classification (At Risk / Not At Risk)
  - AI-generated rationale
  - Actionable improvement suggestions
  - Ask follow-up questions using an AI assistant
  - View teacher feedback (if provided)

### 🧑‍🏫 Teacher Login
- Teachers log in using credentials stored in `users.csv`.
- After login:
  - View all students scoring **below 60** in their subject.
  - Select a student to:
    - View prediction details
    - Ask AI follow-up questions
    - Add feedback for the student
- All feedback is saved to `teacher_feedback.csv`.

---

## 🗂️ Project Structure

EduMentor/
│
├── app.py  # Main Streamlit app
├── inference.py  # Model loading and risk prediction logic
├── preprocessing.py  # Data preprocessing utilities
├── explaination.py  # Handles follow-up Q&A using Groq LLM
├── train_model.py  # Model training script
│
│
├── artifacts/
| └── best_classification_model.pkl
| └── best_regression_model.pkl
| └── label_encoder.pkl
│ └── scaler.pkl 
│
├── Data/
│ ├── users.csv # Teacher login credentials
│ ├── teacher_feedback.csv # Stores teacher feedback per student
│ └── student_data.csv # Dataset used in prediction and login
│
├── config.py # Central configuration file
├── requirements.txt # Python dependencies