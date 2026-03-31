# Predictive-Modeling-for-Course-Demand-and-Revenue-Forecasting-on-EduPro
This repository contains an end-to-end machine learning project for forecasting course demand and revenue on EduPro. The workflow covers data preparation, feature engineering, regression modeling, evaluation, and deployment with a Streamlit web application hosted through Streamlit Community Cloud.
​

Business Problem
EduPro currently depends on historical intuition to decide which courses may perform well. Without predictive models for enrollment and revenue, course launches and pricing decisions carry avoidable business risk.
​

This project addresses that gap by answering the following questions:

Which courses are likely to attract higher enrollments?

Which courses are likely to generate stronger revenue?

Which categories appear most promising for future launches?

How do course and instructor attributes influence demand outcomes?

Objectives
Predict course-level enrollment count.

Predict course-level revenue.

Estimate category-wise revenue trends.

Identify the main drivers of course demand.

Deliver an interactive Streamlit dashboard for business users.
​
​

Dataset
The project uses multiple linked tables or sheets that are merged into one modeling dataset:

Source	Key fields	Purpose
Courses sheet	CourseID, CourseCategory, CourseType, CourseLevel, CoursePrice, CourseDuration, CourseRating	Core course attributes
Teachers sheet	TeacherID, Expertise, YearsOfExperience, TeacherRating	Instructor-level features
Transactions sheet	TransactionID, CourseID, TransactionDate, Amount	Historical demand and revenue signals
These inputs are merged so that transaction history can be aggregated at the course level and combined with course and instructor features for supervised learning.

Project Workflow
1. Data collection
Export the Google Sheet data into CSV files or read the sheets into pandas.

Store the raw data in a data/ folder for reproducibility.

Verify that primary keys such as CourseID and TeacherID are consistent across sheets.
​

2. Data preparation
Merge Courses, Teachers, and Transactions tables.

Aggregate transactions to compute enrollment count and total revenue per course.

Handle missing values in ratings, expertise, or sparse transaction records.

Standardize column names and data types before feature engineering.

3. Feature engineering
Typical engineered features include:

Price band: low, medium, high.

Duration bucket: short, medium, long.

Rating tier from course ratings.

Encoded course level.

Experience bucket for instructors.

Teacher rating score.

Historical average revenue.

Revenue per enrollment.

Expertise-to-category match score, if mapped through a rule-based feature.
​

4. Preprocessing
Encode categorical variables using label encoding or one-hot encoding.

Normalize or standardize numeric features where needed.

Remove redundant columns and identifiers not useful for training.

Split the dataset into train and test sets for unbiased evaluation.

5. Modeling
Recommended model progression:

Baseline: Linear Regression.

Regularized baseline: Ridge and Lasso Regression.

Advanced: Random Forest Regressor.

Advanced: Gradient Boosting Regressor.
​

This staged approach is useful because linear models provide interpretable baselines, while tree-based regressors can capture nonlinear relationships between pricing, rating, experience, and revenue.

6. Deployment
The final model is deployed using Streamlit Community Cloud, which can run apps directly from a GitHub repository. Streamlit Community Cloud has Streamlit installed by default, but projects commonly include a requirements.txt file for non-default packages used by the app.

Suggested Repository Structure
text
edupro-course-demand-forecasting/
│
├── app.py
├── train_model.py
├── feature_engineering.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── courses.csv
│   ├── teachers.csv
│   └── transactions.csv
├── models/
│   ├── enroll_model.pkl
│   ├── revenue_model.pkl
│   └── scaler.pkl
└── outputs/
    ├── processed_features.csv
    └── evaluation_metrics.csv
A clear repository layout makes Streamlit deployment and GitHub presentation easier to understand for reviewers and evaluators.

Installation
Clone the repository and install the project dependencies.

bash
git clone https://github.com/your-username/edupro-course-demand-forecasting.git
cd edupro-course-demand-forecasting
pip install -r requirements.txt
For Streamlit deployments, a concise requirements.txt is preferred over dumping every package from the local environment, because extra packages can make deployment unstable or unnecessarily heavy.

Example requirements.txt
text
pandas
numpy
scikit-learn
plotly
streamlit
openpyxl
joblib
Streamlit Community Cloud does not require built-in Python libraries in requirements.txt, and Streamlit itself is available by default unless a specific version needs to be pinned.
​

How to Run Locally
bash
python feature_engineering.py
python train_model.py
streamlit run app.py
Local testing is recommended before deployment because it makes debugging faster than relying only on cloud deployment logs.

Streamlit App Features
The web app can include the following modules:

Course demand prediction dashboard.

Revenue forecast visualization.

Category comparison charts.

Feature importance explorer.

Interactive input form for price, duration, level, instructor rating, and experience.

Deployment on Streamlit Community Cloud
Push the project to a GitHub repository.

Sign in to Streamlit Community Cloud.

Connect the GitHub account.

Select the repository, branch, and main file path such as app.py.

Add a unique app URL or subdomain if available.

Deploy and monitor the build logs for dependency or file path issues.

A requirements.txt file is strongly recommended for non-standard libraries, even though Streamlit itself is preinstalled on Community Cloud.

Example GitHub Description
Use this short repository description on GitHub:

Machine learning and Streamlit project for forecasting course enrollments, course revenue, and category demand on EduPro using course, instructor, and transaction data.

Key Technologies
Python

Pandas

NumPy

Scikit-learn

Plotly

Streamlit

openpyxl

