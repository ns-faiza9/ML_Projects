import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = r"C:\Users\nsfai\OneDrive\Desktop\WORK\STUDIES\SEMESTER - 4\ML\PROJECTS\EDA\placement_predict_50k Dataset.csv"

sns.set(style="white")

# Pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def show(title=""):
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File not found: {CSV_PATH}'.Update CSV_PATH at top of script.")
data = pd.read_csv(CSV_PATH)
print("=" * 80)
print("1. DATA LOADED")
print("=" * 80)
print("Shape:", data.shape)


# 2. BASIC INFO / STRUCTURE
print("\n" + "=" * 80)
print("2. BASIC INFO")
print("=" * 80)
print(data.info())
print("\nColumn dtypes: \n", data.dtypes)
print("\nDescribe (numeric): \n", data.describe())
print("\nDescribe (Categorical): \n", data.describe(include="str"))

# 3. MISSING VALUES
print("\n" + "=" * 80)
print("3. MISSING VALUES")
print("=" * 80)
missing = data.isnull().sum()
missing_pct = (missing / len(data)) * 100
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(by="missing_pct", ascending=False)
print(missing_df)


if not missing_df.empty:
    plt.figure(figsize=(10, 5), dpi=100)
    sns.barplot(x=missing_df.index, y=missing_df["missing_pct"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing %")
    plt.title("Missing values by column")
    show()


# 4. DUPLICATES
print("\n" + "=" * 80)
print("4. DUPLICATES ROWS")
print("=" * 80)
print("Duplicate rows:", data.duplicated().sum())

# 5. TARGET VARIABLE DISTRIBUTION (Placement Status)
print("\n" + "=" * 80)
print("5. TARGET VARIABLE - Placement Status ")
print("=" * 80)
print(data["PlacementStatus"].value_counts())

plt.figure(dpi=125)
sns.countplot(x="PlacementStatus", data=data)
plt.xlabel("Placement Status (0 = Not placed, 1 = Placed)")
plt.ylabel("Count")
plt.title("Count of Placement Status")
plt.show()

#6. NUMERIC FEATURE DISTRIBUTION
print("\n" + "=" * 80)
print("6. PLACEMENT DISTRIBUTION")
print("=" * 80)

hist_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore" ]
hist_cols = [c for c in hist_cols if c in data.columns]

data[hist_cols].hist(figsize=(14, 10), bins = 20)
show("Numeric Feature Distributions")

#Mean Line Example
plt.figure(dpi = 125)
sns.histplot(data["CGPA"], kde = True)
plt.axvline(x = np.mean(data["CGPA"]), color = "green", linestyle = "--", label = "CGPA" )
plt.legend()
plt.title("CGPA DISTRIBUTION")
show()

#7. OUTLIER DETECTION (80XPLOTS)
print("\n" + "=" * 80)


box_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore", "SalaryPackage" ]
box_cols = [c for c in box_cols if c in data.columns]

for col in box_cols:
    plt.figure(figsize = (10, 4))
    sns.boxplot( x = data[col], color = "pink")
    plt.title(f"Box Plot - {col}", fontsize = 20)
    show()

#8.CORRELATION HEATMAP
print("\n" + "=" * 80)
print("8. CORRELATION ANALYSIS")
print("=" * 80)

corr=data.select_dtypes(include=[np.number]).corr()
print(np.round(corr, decimals=2))

plt.figure(figsize = (16, 12), dpi=100)
sns.heatmap(np.round(corr, decimals=2), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
show()

#9. RELATIONSHIP PLOTS
print("\n" + "=" * 80)
print("9. RELATIONSHIP PLOTS")
print("=" * 80)

if "CGPA" in data.columns and "SalaryPackage" in data.columns:
    plt.figure(dpi=125)
    sns.regplot(x="CGPA", y="SalaryPackage", data=data, scatter_kws={"alpha":0.5})
    plt.title("CGPA vs Salary Package")
    show()

if "AptitudeTestScore" in data.columns and "CodingTestScore" in data.columns:
    plt.figure(dpi=125)
    sns.regplot(x="AptitudeTestScore", y="CodingTestScore", data=data, scatter_kws={"alpha":0.5})
    plt.title("Aptitude vs Coding Test Score")
    show()


#10. CATEGORICAL FEATURE COUNTS
print("=" * 80)
print("CATEGORICAL FEATURE COUNTS")
print("=" * 80)
cat_cols = ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CGPA_Tier"]
cat_cols = [c for c in cat_cols if c in data.columns]

for col in cat_cols:
    plt.figure(dpi=125, figsize=(10, 5))
    sns.countplot(x=col, data=data, order=data[col].value_counts().index)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45, ha="right")
    show()


#11. GENDER vs PLACEMENT STATUS
print("=" * 80)
print("11. GENDER vs PLACEMENT STATUS")
print("=" * 80)
if "Gender" in data.columns and "PlacementStatus" in data.columns:
    plt.figure(dpi=125)
    sns.countplot(x="Gender", hue="PlacementStatus", data=data)
    plt.title("Placement Status by Gender")
    show()


#12. COLLEGE TIER vs PLACEMENT STATUS
print("=" * 80)
print("12. COLLEGE TIER vs PLACEMENT STATUS ")
print("=" * 80)
for col in ["CollegeTier", "Stream"]:
    if col in data.columns and "PlacementStatus" in data.columns:
        plt.figure(dpi=125)
        sns.countplot(x=col, hue="PlacementStatus", data=data)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Placement Status by {col}")
        show()

# 13. SGPA TREND ACROSS SEMESTERS
print("=" * 80)
print("13. SGPA TREND ACROSS SEMESTERS ")
print("=" * 80)
sgpa_cols = [c for c in data.columns if c.startswith("SGPA_Sem")]
if sgpa_cols:
    mean_sgpa = data[sgpa_cols].mean()
    plt.figure(dpi=125)
    mean_sgpa.plot(marker="o")
    plt.title("Average SGPA Trend Across Semesters")
    plt.xlabel("Semester")
    plt.ylabel("Average SGPA")
    show()


# 14. SALARY PACKAGE ANALYSIS
print("=" * 80)
print("14. SALARY PACKAGE ANALYSIS ")
print("=" * 80)
if "SalaryPackage" in data.columns and "PlacementStatus" in data.columns:
    plt.figure(dpi=125)
    sns.histplot(data[data["PlacementStatus"]==1]["SalaryPackage"], kde=True)
    plt.title("Salary Distribution for Placed Students")
    show()

if "SalaryPackage" in data.columns and "CollegeTier" in data.columns:
    plt.figure(dpi=125)
    sns.boxplot(x="CollegeTier", y="SalaryPackage", data=data)
    plt.title("Salary Package by College Tier")
    show()


# 15. PAIRPLOT
print("=" * 80)
print("15. PAIRPLOT OF KEY NUMERIC FEATURES VS TARGET")
print("=" * 80)
pair_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
pair_cols = [c for c in pair_cols if c in data.columns]

if "PlacementStatus" in data.columns and pair_cols:
    sns.pairplot(data[pair_cols + ["PlacementStatus"]], hue="PlacementStatus", diag_kind="kde")
    plt.suptitle("Pairplot of Key Features by Placement Status", y=1.02)
    show()
