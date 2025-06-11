import streamlit as st
import pandas as pd
from inference import predict_student_risk
from config import STUDENT_DATA_PATH,TEACHERS_CSV, FEEDBACK_FILE
from explaination import answer_followup_question
import datetime
import os
# ---------------- CONFIG ----------------
  # path to teacher credentials

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_student_data():
    return pd.read_csv(STUDENT_DATA_PATH)

@st.cache_data
def load_teacher_data():
    return pd.read_csv(TEACHERS_CSV)

student_data = load_student_data()
teachers_data = load_teacher_data()

# ---------------- SESSION STATE INIT ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "is_teacher" not in st.session_state:
    st.session_state.is_teacher = False
if "role" not in st.session_state:
    st.session_state.role = None
if "student_row" not in st.session_state:
    st.session_state.student_row = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "teacher_username" not in st.session_state:
    st.session_state.teacher_username = None
if "teacher_subject" not in st.session_state:
    st.session_state.teacher_subject = None
if "selected_student" not in st.session_state:
    st.session_state.selected_student = None

# ---------------- LOGIN SCREEN ----------------
if not st.session_state.logged_in:
    st.title("🎓 EduMentor Login")
    login_option = st.radio("Login as", ["Student", "Teacher"])

    if login_option == "Student":
        student_id_input = st.text_input("Student ID")
        password_input = st.text_input("Password", type="password")

        if st.button("🔐 Login"):
            student_match = student_data[
                (student_data["student_id"].astype(str) == student_id_input.strip()) &
                (student_data["password"] == password_input)
            ]

            if not student_match.empty:
                st.session_state.logged_in = True
                st.session_state.is_teacher = False
                st.session_state.role = "student"
                st.session_state.student_row = student_match.iloc[0]
                st.session_state.prediction = predict_student_risk(st.session_state.student_row)
                st.success("✅ Student login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid Student ID or Password.")

    else:  # Teacher login
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("🔐 Login"):
            match = teachers_data[
                (teachers_data["username"] == username) &
                (teachers_data["password"] == password)
            ]
            if not match.empty:
                st.session_state.logged_in = True
                st.session_state.is_teacher = True
                st.session_state.role = "teacher"
                st.session_state.teacher_username = username
                st.session_state.teacher_subject = match.iloc[0]["subject"]
                st.success(f"✅ Logged in as {username} ({match.iloc[0]['subject'].title()} Teacher)")
                st.rerun()
            else:
                st.error("❌ Invalid Teacher Credentials.")

# ---------------- STUDENT DASHBOARD ----------------
if st.session_state.logged_in and st.session_state.role == "student":
    student = st.session_state.student_row
    pred = st.session_state.prediction

    st.title("🎓 EduMentor: Academic Risk Prediction")
    st.subheader(f"👤 Welcome, {student.get('student_name', 'Student')}")

    st.subheader("📊 Prediction Result")
    st.write(f"**Risk Score:** `{pred['risk_score']:.2f}`")
    st.write(f"**Is At Risk:** `{pred['is_at_risk']}`")

    st.subheader("🧠 Explanation")
    st.markdown(pred["explanation"])

    st.subheader("✅ Personalized Suggestions")
    st.markdown(pred["suggestions"])
    
    st.subheader("📬 Feedback from Your Teachers")

    feedback_file = FEEDBACK_FILE
    if os.path.exists(feedback_file):
        feedback_data = pd.read_csv(feedback_file)
        student_feedback = feedback_data[feedback_data["student_id"] == student["student_id"]]

        if not student_feedback.empty:
            for _, row in student_feedback.iterrows():
                st.markdown(f"""
                    **🧑‍🏫 {row['teacher_username'].title()} ({row['subject'].capitalize()})**
                    - *{row['timestamp']}*
                    > {row['feedback']}
                    ---
                """)
        else:
            st.info("No feedback received yet.")
    else:
        st.info("Feedback file not found.")


    st.subheader("❓ Ask a Follow-Up Question")
    followup_question = st.text_input("Ask a question about this student's performance:")
    if followup_question:
        with st.spinner("Analyzing student data..."):
            answer = answer_followup_question(student.to_dict(), followup_question)
        st.markdown(f"**Answer:** {answer}")

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# ---------------- TEACHER DASHBOARD ----------------
elif st.session_state.logged_in and st.session_state.role == "teacher":
    subject = st.session_state.teacher_subject
    st.title(f"👩‍🏫 EduMentor: Teacher Dashboard ({subject.capitalize()})")

    # Filter students with marks < 60 in teacher's subject
    subject_students = student_data[student_data[subject] < 60]

    st.subheader(f"📋 Students Scoring Below 60 in {subject.capitalize()}")
    st.dataframe(subject_students[["student_id", "student_name", subject]], use_container_width=True)

    student_id_input = st.text_input("Enter Student ID to view details:")
    student_info = subject_students[subject_students["student_id"].astype(str) == student_id_input.strip()]

    if not student_info.empty:
        student = student_info.iloc[0]
        prediction = predict_student_risk(student)

        st.subheader(f"📊 Prediction Result for {student['student_name']}")
        st.write(f"**Risk Score:** `{prediction['risk_score']:.2f}`")
        st.write(f"**Is At Risk:** `{prediction['is_at_risk']}`")

        # st.subheader("🧠 Explanation")
        # st.markdown(prediction["explanation"])
        
        st.subheader("❓ Ask a Follow-Up Question")
        followup_question = st.text_input("Ask a question about this student's performance:")
        if followup_question:
            with st.spinner("Analyzing student data..."):
                answer = answer_followup_question(student.to_dict(), followup_question)
            st.markdown(f"**Answer:** {answer}")

        st.subheader("📌 Add Feedback")
        feedback = st.text_area("Add comments or feedback about this student (optional):")

        if st.button("💾 Save Feedback"):
            new_feedback = {
                "student_id": student["student_id"],
                "student_name": student["student_name"],
                "teacher_username": st.session_state.teacher_username,
                "subject": subject,
                "feedback": feedback,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            feedback_df = pd.DataFrame([new_feedback])

            # Append to file if exists, else create
            if os.path.exists(FEEDBACK_FILE):
                feedback_df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(FEEDBACK_FILE, mode='w', header=True, index=False)

            st.success("✅ Feedback saved successfully!")

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()
