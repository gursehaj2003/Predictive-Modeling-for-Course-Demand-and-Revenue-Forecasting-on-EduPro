import os
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

st.set_page_config(page_title="EduPro Demand Forecast", layout="wide", page_icon="🎓")
st.title("🎓 EduPro Course Demand & Revenue Predictor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= LOAD MODELS =================
@st.cache_data
def load_models():
    enroll_model = joblib.load(os.path.join(BASE_DIR,'enroll_model.pkl'))
    revenue_model = joblib.load(os.path.join(BASE_DIR,'revenue_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR,'scaler.pkl'))
    return enroll_model, revenue_model, scaler

enroll_model, revenue_model, scaler = load_models()

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    courses = pd.read_csv(os.path.join(BASE_DIR, "Courses.csv"))
    teachers = pd.read_csv(os.path.join(BASE_DIR, "Teachers.csv"))
    transactions = pd.read_csv(os.path.join(BASE_DIR, "Transactions.csv"))

    # Aggregate transactions
    course_rev = transactions.groupby("CourseID")["Amount"].agg(["count", "sum"]).reset_index()
    course_rev.columns = ["CourseID", "EnrollmentCount", "Revenue"]

    # Merge datasets
    df = courses.merge(course_rev, on="CourseID", how="left")
    if "TeacherID" in df.columns and "TeacherID" in teachers.columns:
        df = df.merge(teachers, on="TeacherID", how="left")

    df.fillna(0, inplace=True)

    # Use actual column names from your CSVs
    feature_cols = []
    if "CoursePrice" in df.columns: feature_cols.append("CoursePrice")
    if "CourseDuration" in df.columns: feature_cols.append("CourseDuration")
    if "CourseRating" in df.columns: feature_cols.append("CourseRating")
    if "Experience" in df.columns: feature_cols.append("Experience")  # optional

    features = df[feature_cols]
    return df, features

df, features = load_data()

# ================= DEMO PLOT =================
st.subheader("📊 Course Revenue vs Enrollments")
fig = px.scatter(df, x="EnrollmentCount", y="Revenue", size="CoursePrice",
                 color="CourseCategory", hover_data=["CourseName"])
st.plotly_chart(fig, use_container_width=True)
