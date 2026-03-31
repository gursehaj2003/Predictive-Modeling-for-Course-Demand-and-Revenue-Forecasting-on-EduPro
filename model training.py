import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# --- Load your dataset(s) ---
courses = pd.read_csv("Courses.csv")
users = pd.read_csv("EduPro Online Platform.xlsx - Users.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge datasets
df = transactions.merge(users, on="UserID", how="left").merge(courses, on="CourseID", how="left")

# --- Preprocessing ---
df.fillna(0, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# --- Define target variables ---
# Revenue = transaction amount
y_revenue = df["Amount"]

# Enrollment count = Amount / CoursePrice (avoid division by zero)
df["EnrollCount"] = np.where(df["CoursePrice"] > 0, df["Amount"] / df["CoursePrice"], 0)
y_enroll = df["EnrollCount"]

# Features = everything except targets
X = df.drop(columns=["Amount", "EnrollCount"], errors="ignore")

# --- Split both targets together ---
X_train, X_test, y_enroll_train, y_enroll_test, y_revenue_train, y_revenue_test = train_test_split(
    X, y_enroll, y_revenue, test_size=0.2, random_state=42
)

# --- Scale ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train enrollment model ---
enroll_model = RandomForestRegressor(n_estimators=100, random_state=42)
enroll_model.fit(X_train_scaled, y_enroll_train)

joblib.dump(enroll_model, 'enroll_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'encoder.pkl')

# --- Train revenue model ---
rev_model = RandomForestRegressor(n_estimators=100, random_state=42)
rev_model.fit(X_train_scaled, y_revenue_train)

joblib.dump(rev_model, 'revenue_model.pkl')

# --- Evaluate enrollment model ---
enroll_pred = enroll_model.predict(X_test_scaled)
print(f"Enrollment MAE: {mean_absolute_error(y_enroll_test, enroll_pred):.2f}")
print(f"Enrollment R2: {r2_score(y_enroll_test, enroll_pred):.3f}")

# --- Evaluate revenue model ---
rev_pred = rev_model.predict(X_test_scaled)
print(f"Revenue MAE: {mean_absolute_error(y_revenue_test, rev_pred):.2f}")
print(f"Revenue R2: {r2_score(y_revenue_test, rev_pred):.3f}")
