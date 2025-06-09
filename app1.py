# app1.py

import streamlit as st
import pandas as pd
from inference import predict_student_risk
from config import STUDENT_DATA_PATH
from explaination import answer_followup_question

# Load student data
@st.cache_data
def load_student_data():
    return pd.read_csv(STUDENT_DATA_PATH)

# Streamlit app title
st.title("🎓 EduMentor: Academic Risk Prediction")
st.markdown("Enter a Student ID to analyze academic risk and get personalized recommendations.")

# Initialize session state to persist student data
if "student_row" not in st.session_state:
    st.session_state.student_row = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Student ID input
student_id_input = st.text_input("Enter Student ID:", "")

# Predict button
if st.button("Predict"):
    if not student_id_input.strip():
        st.warning("Please enter a Student ID.")
    else:
        data = load_student_data()
        matching_student = data[data["student_id"].astype(str) == str(student_id_input).strip()]
        
        if not matching_student.empty:
            st.session_state.student_row = matching_student.iloc[0]
            st.session_state.prediction = predict_student_risk(st.session_state.student_row)
        else:
            st.error("❌ Student ID not found. Please check and try again.")

# Display prediction if available
if st.session_state.student_row is not None and st.session_state.prediction is not None:
    student_row = st.session_state.student_row
    prediction = st.session_state.prediction

    st.subheader(f"👤 Student Name: {student_row.get('student_name', 'Name not available')}")
    st.subheader("📊 Prediction Result")
    st.write(f"**Risk Score:** `{prediction['risk_score']}`")
    st.write(f"**Is At Risk:** `{prediction['is_at_risk']}`")

    st.subheader("🧠 Explanation")
    st.markdown(prediction["explanation"])

    st.subheader("✅ Personalized Suggestions")
    st.markdown(prediction["suggestions"])

    # Follow-up question
    st.subheader("❓ Ask a Follow-Up Question")
    followup_question = st.text_input("Ask a question about this student's performance:")

    if followup_question:
        with st.spinner("Analyzing student data..."):
            answer = answer_followup_question(student_row.to_dict(), followup_question)
        st.markdown(f"**Answer:** {answer}")
