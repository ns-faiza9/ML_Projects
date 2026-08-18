import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "./placement_predict_50k Dataset.csv"
df = pd.read_csv(file_path)

# print(df)
# print("Available columns:", df.columns.tolist())


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

# Apply Min-Max scaling
scaler = MinMaxScaler()

train_df[nums_cols] = scaler.fit_transform(train_df[nums_cols])
test_df[nums_cols] = scaler.transform(test_df[nums_cols])

print("Original Data shape: ", df.shape)
print(train_df[nums_cols].head())

print("\nTraining Data after scaling")
print(train_df[nums_cols].head())

print("\nTesting Data after scaling")
print(test_df[nums_cols].head())
