import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "./placement_predict_50k Dataset.csv"
df = pd.read_csv(file_path)

feature = "CodingTestScore"

print("Original Statistics")
print(df[feature].describe())

# Calculate IQR
Q1 = df[feature].quantile(0.25)
Q3 = df[feature].quantile(0.75)
IQR = Q3 - Q1

# Fences
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print("\nQ1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Fence:", lower_fence)
print("Upper Fence:", upper_fence)

# Detect outliers
outliers = df[(df[feature] < lower_fence) | (df[feature] > upper_fence)]

print("\nNumber of outliers:", len(outliers))
print("\nOutlier values:")
print(outliers[feature].head())

# Clip outliers
df["CodingScore_Clipped"] = df[feature].clip(lower=lower_fence, upper=upper_fence)

print("\nMinimum BEFORE clipping:", df[feature].min())
print("Maximum BEFORE clipping:", df[feature].max())

print("\nMinimum AFTER clipping:", df["CodingScore_Clipped"].min())
print("Maximum AFTER clipping:", df["CodingScore_Clipped"].max())

# Apply Min-Max scaling
scaler = MinMaxScaler()
df["CodingScore_Scaled"] = scaler.fit_transform(
    df[["CodingScore_Clipped"]]
)

print("\nScaled values (first 5 rows):")
print(df[["CodingScore_Clipped", "CodingScore_Scaled"]].head())
