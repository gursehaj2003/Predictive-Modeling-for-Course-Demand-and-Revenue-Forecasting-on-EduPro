import pandas as pd

# Load data
courses = pd.read_csv('Courses.csv')
teachers = pd.read_csv('Teachers.csv')
transactions = pd.read_csv('Transactions.csv')

# Aggregate transactions by CourseID
course_rev = transactions.groupby('CourseID')['Amount'].agg(['count', 'sum']).reset_index()
course_rev.columns = ['CourseID', 'EnrollmentCount', 'Revenue']

# Check columns
print("Courses columns:", courses.columns.tolist())
print("Teachers columns:", teachers.columns.tolist())
print("Transactions columns:", transactions.columns.tolist())

# Merge courses with revenue
df = courses.merge(course_rev, on='CourseID', how='left')

# Merge with teachers only if both have TeacherID
if 'TeacherID' in df.columns and 'TeacherID' in teachers.columns:
    df = df.merge(teachers, on='TeacherID', how='left')
else:
    print("⚠️ TeacherID not found in both datasets. Skipping teacher merge.")

# Handle missing values
df.fillna(0, inplace=True)

# Save processed file for use in your app
df.to_csv("processed_courses.csv", index=False)
print("✅ Processed courses saved as processed_courses.csv")
print(df.head())
