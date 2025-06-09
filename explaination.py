# explanation.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import dotenv

dotenv.load_dotenv()  # Load GROQ_API_KEY from .env

def get_groq_api_key():
    return os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(api_key=get_groq_api_key(), model_name="llama-3.3-70b-versatile")

def generate_explanation(student_data: dict, prediction: dict) -> tuple[str, str]:
    """
    Generates a natural language explanation and improvement suggestions
    based on student features and prediction results.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an educational AI assistant that helps teachers understand why a student may be at risk of dropping out."),
        ("user", """Here is the student's data:

Grades:
- Overall: {overall_grade}
- Math: {math_grade}
- English: {english_grade}
- Science: {science_grade}
- History: {history_grade}

Engagement:
- Score: {engagement_score}
- Logins/week: {login_frequency_per_week}
- Avg session duration: {average_session_duration_minutes} min
- Lessons completed: {completed_lessons}
- Practice tests taken: {practice_tests_taken}
- Assignment completion: {assignment_completion}%
- LMS test scores: {lms_test_scores}%
- Attendance ratio: {attendance_ratio}%
- Learning Style: {learning_style}
- Content Preference: {content_type_preference}

Model Prediction:
- Risk Score: {risk_score}/100
- At Risk: {is_at_risk}

Based on this data, explain *why* this student is considered '{is_at_risk}' and provide 2–3 personalized suggestions to help them improve. Make it readable for teachers and counselors.""")
    ])
    
    input_data = {**student_data, **prediction}
    chain = prompt | llm
    response = chain.invoke(input_data)
    content = response.content

    split_marker = "**Personalized Suggestions to Improve**"

    if split_marker in content:
        explanation_text, suggestions_text = content.split(split_marker, 1)
        return explanation_text.strip(), suggestions_text.strip()
    else:
        return content.strip(), ""


def answer_followup_question(student_data: dict, question: str) -> str:
    """
    Answers a follow-up question based on a student's data.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant helping teachers understand student performance based on provided data."),
        ("user", """Student Data:
{student_info}

Question: {question}

Please answer the question using only the student’s data. Be clear, concise, and informative.""")
    ])

    student_info = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in student_data.items()])
    input_data = {
        "student_info": student_info,
        "question": question
    }

    chain = prompt | llm
    response = chain.invoke(input_data)
    return response.content.strip()
