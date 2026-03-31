import os
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

st.set_page_config(page_title="EduPro Demand Forecast", layout="wide", page_icon="🎓")
st.title("🎓 EduPro Course Demand & Revenue Predictor")
st.markdown("Forecast course demand and revenue using course, teacher, and transaction data.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ================= HELPER FUNCTION =================
def check_file_exists(file_path, file_label):
    if not os.path.exists(file_path):
        st.error(f"Missing file: {file_label}")
        st.error(f"Expected path: {file_path}")
        st.stop()

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    enroll_model_path = os.path.join(MODEL_DIR, "enroll_model.pkl")
    revenue_model_path = os.path.join(MODEL_DIR, "revenue_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    check_file_exists(enroll_model_path, "models/enroll_model.pkl")
    check_file_exists(revenue_model_path, "models/revenue_model.pkl")
    check_file_exists(scaler_path, "models/scaler.pkl")

    enroll_model = joblib.load(enroll_model_path)
    revenue_model = joblib.load(revenue_model_path)
    scaler = joblib.load(scaler_path)

    return enroll_model, revenue_model, scaler

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    courses_path = os.path.join(DATA_DIR, "Courses.csv")
    teachers_path = os.path.join(DATA_DIR, "Teachers.csv")
    transactions_path = os.path.join(DATA_DIR, "Transactions.csv")

    check_file_exists(courses_path, "data/Courses.csv")
    check_file_exists(teachers_path, "data/Teachers.csv")
    check_file_exists(transactions_path, "data/Transactions.csv")

    courses = pd.read_csv(courses_path)
    teachers = pd.read_csv(teachers_path)
    transactions = pd.read_csv(transactions_path)

    if "CourseID" not in transactions.columns:
        st.error("Transactions.csv must contain 'CourseID'")
        st.stop()

    if "Amount" not in transactions.columns:
        st.error("Transactions.csv must contain 'Amount'")
        st.stop()

    course_rev = transactions.groupby("CourseID")["Amount"].agg(["count", "sum"]).reset_index()
    course_rev.columns = ["CourseID", "EnrollmentCount", "Revenue"]

    if "CourseID" not in courses.columns:
        st.error("Courses.csv must contain 'CourseID'")
        st.stop()

    df = courses.merge(course_rev, on="CourseID", how="left")

    if "TeacherID" in df.columns and "TeacherID" in teachers.columns:
        df = df.merge(teachers, on="TeacherID", how="left")

    df["EnrollmentCount"] = df["EnrollmentCount"].fillna(0)
    df["Revenue"] = df["Revenue"].fillna(0)
    df.fillna(0, inplace=True)

    return df

# ================= PREPARE FEATURES =================
def prepare_features(df):
    df_model = df.copy()

    if "CoursePrice" in df_model.columns:
        df_model["PriceBand"] = pd.cut(
            df_model["CoursePrice"],
            bins=[0, 500, 1000, float("inf")],
            labels=["Low", "Medium", "High"]
        ).astype(str)
    else:
        df_model["PriceBand"] = "Unknown"

    if "CourseDuration" in df_model.columns:
        unique_vals = df_model["CourseDuration"].nunique()
        if unique_vals >= 3:
            df_model["DurationBucket"] = pd.qcut(
                df_model["CourseDuration"],
                3,
                labels=["Short", "Medium", "Long"],
                duplicates="drop"
            ).astype(str)
        else:
            df_model["DurationBucket"] = "Medium"
    else:
        df_model["DurationBucket"] = "Unknown"

    if "CourseRating" in df_model.columns:
        df_model["RatingTier"] = pd.cut(
            df_model["CourseRating"],
            bins=[0, 3, 4, 5],
            labels=["Low", "Medium", "High"],
            include_lowest=True
        ).astype(str)
    else:
        df_model["RatingTier"] = "Unknown"

    if "YearsOfExperience" in df_model.columns:
        df_model["ExpBucket"] = pd.cut(
            df_model["YearsOfExperience"],
            bins=[0, 5, 10, float("inf")],
            labels=["Junior", "Mid", "Senior"],
            include_lowest=True
        ).astype(str)
    else:
        df_model["ExpBucket"] = "Unknown"

    cat_cols = [
        "CourseCategory",
        "CourseType",
        "CourseLevel",
        "PriceBand",
        "DurationBucket",
        "RatingTier",
        "ExpBucket",
        "Expertise"
    ]

    for col in cat_cols:
        if col not in df_model.columns:
            df_model[col] = "Unknown"
        df_model[col] = df_model[col].astype("category").cat.codes

    drop_cols = [
        "CourseID",
        "TeacherID",
        "EnrollmentCount",
        "Revenue",
        "TransactionID",
        "TransactionDate",
        "CourseName"
    ]

    X = df_model.drop(columns=drop_cols, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X

# ================= MAIN APP =================
try:
    enroll_model, revenue_model, scaler = load_models()
    df = load_data()

    X = prepare_features(df)

    try:
        X_scaled = scaler.transform(X)
    except Exception:
        X_scaled = X

    try:
        df["PredictedEnrollments"] = np.maximum(enroll_model.predict(X_scaled), 0)
    except Exception:
        df["PredictedEnrollments"] = 0

    try:
        df["PredictedRevenue"] = np.maximum(revenue_model.predict(X_scaled), 0)
    except Exception:
        df["PredictedRevenue"] = 0

    # ================= KPI METRICS =================
    st.subheader("📌 Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Courses", len(df))
    col2.metric("Total Predicted Enrollments", f"{df['PredictedEnrollments'].sum():,.0f}")
    col3.metric("Total Predicted Revenue", f"${df['PredictedRevenue'].sum():,.2f}")

    # ================= FILTERS =================
    st.sidebar.header("Filters")

    if "CourseCategory" in df.columns:
        category_options = ["All"] + sorted(df["CourseCategory"].astype(str).unique().tolist())
        selected_category = st.sidebar.selectbox("Select Course Category", category_options)
    else:
        selected_category = "All"

    filtered_df = df.copy()

    if selected_category != "All" and "CourseCategory" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["CourseCategory"].astype(str) == selected_category]

    # ================= CHARTS =================
    st.subheader("📊 Course Revenue vs Enrollments")

    x_col = "EnrollmentCount" if "EnrollmentCount" in filtered_df.columns else "PredictedEnrollments"
    y_col = "Revenue" if "Revenue" in filtered_df.columns else "PredictedRevenue"
    size_col = "CoursePrice" if "CoursePrice" in filtered_df.columns else None
    color_col = "CourseCategory" if "CourseCategory" in filtered_df.columns else None
    hover_cols = []

    if "CourseName" in filtered_df.columns:
        hover_cols.append("CourseName")
    elif "CourseID" in filtered_df.columns:
        hover_cols.append("CourseID")

    fig = px.scatter(
        filtered_df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        hover_data=hover_cols,
        title="Revenue vs Enrollment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ================= CATEGORY PERFORMANCE =================
    if "CourseCategory" in filtered_df.columns:
        st.subheader("📚 Category-Level Forecast")

        category_summary = filtered_df.groupby("CourseCategory", as_index=False).agg({
            "PredictedEnrollments": "sum",
            "PredictedRevenue": "sum"
        })

        fig_bar = px.bar(
            category_summary,
            x="CourseCategory",
            y="PredictedRevenue",
            color="PredictedEnrollments",
            title="Predicted Revenue by Course Category",
            text_auto=".2s"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ================= DATA TABLE =================
    st.subheader("🧾 Forecast Table")

    display_cols = []
    for col in [
        "CourseID",
        "CourseName",
        "CourseCategory",
        "CoursePrice",
        "EnrollmentCount",
        "Revenue",
        "PredictedEnrollments",
        "PredictedRevenue"
    ]:
        if col in filtered_df.columns:
            display_cols.append(col)

    st.dataframe(filtered_df[display_cols], use_container_width=True)

    # ================= DEBUG INFO =================
    with st.expander("🔍 Debug File Check"):
        st.write("BASE_DIR:", BASE_DIR)
        st.write("MODEL_DIR:", MODEL_DIR)
        st.write("DATA_DIR:", DATA_DIR)
        st.write("Files in BASE_DIR:", os.listdir(BASE_DIR))
        if os.path.exists(MODEL_DIR):
            st.write("Files in models/:", os.listdir(MODEL_DIR))
        if os.path.exists(DATA_DIR):
            st.write("Files in data/:", os.listdir(DATA_DIR))

except Exception as e:
    st.error("Application failed to load.")
    st.exception(e)
