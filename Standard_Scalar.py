import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "./placement_predict_50k Dataset.csv"
df = pd.read_csv(file_path)

# Numeric columns (make sure these match your CSV exactly)
nums_cols = [
    "CGPA",
    "AttendancePercent",
    "AptitudeTestScore",
    "MockInterviewScore",
    "Internships"
]

# Split into train and test sets
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["PlacementStatus"]
)

print("\nTraining Data before scaling")
print(train_df[nums_cols].head())

print("\nTesting Data before scaling")
print(test_df[nums_cols].head())

# Apply Standard Scaler (z-score normalization)
scaler = StandardScaler()

# Fit on training data and transform both sets
X_train_scaled = scaler.fit_transform(train_df[nums_cols])
X_test_scaled = scaler.transform(test_df[nums_cols])

print("\nOriginal Data shape:", df.shape)

print("\nTraining Data after scaling (array)")
print(X_train_scaled[:5])   # first 5 rows of scaled training data

print("\nTesting Data after scaling (array)")
print(X_test_scaled[:5])    # first 5 rows of scaled test data
