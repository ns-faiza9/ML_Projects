import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for the web app

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from load_data import load_data

BASE_DIR = os.path.dirname(__file__)
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")

sns.set(style="white")


def _save(fig, fname):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, fname)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return fname


def run_eda():
    """Runs the EDA, saves charts to static/charts, and returns a results dict."""
    data = load_data()

    charts = []

    # 1. Target variable distribution
    fig, ax = plt.subplots(dpi=100)
    sns.countplot(x="PlacementStatus", data=data, ax=ax)
    ax.set_xlabel("Placement Status (0 = Not placed, 1 = Placed)")
    ax.set_ylabel("Count")
    ax.set_title("Count of Placement Status")
    charts.append(_save(fig, "placement_status.png"))

    # 2. Numeric feature distributions
    hist_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore", "SoftSkillsRating",
                 "CodingTestScore", "MockInterviewScore"]
    hist_cols = [c for c in hist_cols if c in data.columns]
    fig = data[hist_cols].hist(figsize=(14, 10), bins=20)
    fig[0][0].figure.suptitle("Numeric Feature Distributions")
    charts.append(_save(fig[0][0].figure, "numeric_distributions.png"))

    # 3. Correlation heatmap
    corr = data.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
    sns.heatmap(np.round(corr, decimals=2), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    charts.append(_save(fig, "correlation_heatmap.png"))

    # 4. CGPA vs Salary Package
    if "CGPA" in data.columns and "SalaryPackage" in data.columns:
        fig, ax = plt.subplots(dpi=100)
        sns.regplot(x="CGPA", y="SalaryPackage", data=data, scatter_kws={"alpha": 0.5}, ax=ax)
        ax.set_title("CGPA vs Salary Package")
        charts.append(_save(fig, "cgpa_vs_salary.png"))

    # 5. Gender vs Placement Status
    if "Gender" in data.columns and "PlacementStatus" in data.columns:
        fig, ax = plt.subplots(dpi=100)
        sns.countplot(x="Gender", hue="PlacementStatus", data=data, ax=ax)
        ax.set_title("Placement Status by Gender")
        charts.append(_save(fig, "gender_vs_placement.png"))

    # 6. College Tier vs Placement Status
    if "CollegeTier" in data.columns and "PlacementStatus" in data.columns:
        fig, ax = plt.subplots(dpi=100)
        sns.countplot(x="CollegeTier", hue="PlacementStatus", data=data, ax=ax)
        ax.set_title("Placement Status by College Tier")
        charts.append(_save(fig, "college_tier_vs_placement.png"))

    # 7. SGPA trend across semesters
    sgpa_cols = [c for c in data.columns if c.startswith("SGPA_Sem")]
    if sgpa_cols:
        fig, ax = plt.subplots(dpi=100)
        data[sgpa_cols].mean().plot(marker="o", ax=ax)
        ax.set_title("Average SGPA Trend Across Semesters")
        ax.set_xlabel("Semester")
        ax.set_ylabel("Average SGPA")
        charts.append(_save(fig, "sgpa_trend.png"))

    # 8. Salary distribution for placed students
    if "SalaryPackage" in data.columns and "PlacementStatus" in data.columns:
        fig, ax = plt.subplots(dpi=100)
        sns.histplot(data[data["PlacementStatus"] == 1]["SalaryPackage"], kde=True, ax=ax)
        ax.set_title("Salary Distribution for Placed Students")
        charts.append(_save(fig, "salary_placed.png"))

    missing = data.isnull().sum()
    missing = missing[missing > 0].to_dict()

    return {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "duplicate_count": int(data.duplicated().sum()),
        "missing": missing,
        "target_counts": data["PlacementStatus"].value_counts().to_dict(),
        "charts": charts,
    }
