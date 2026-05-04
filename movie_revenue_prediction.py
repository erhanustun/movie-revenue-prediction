# ==============================================================
# MOVIE SUCCESS PREDICTION — Capstone Project
# Dataset: TMDB 5000 Movie Dataset (Kaggle)
# ==============================================================

# ------------------------------------------------------------------
# STEP 1: IMPORTING REQUIRED LIBRARIES
# ------------------------------------------------------------------
import ast
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
print("Libraries loaded successfully.")


# ------------------------------------------------------------------
# STEP 2: DATA LOADING
# The dataset is manually downloaded from Kaggle and placed in data/raw/.
# In a real-world pipeline, this data might be read from PostgreSQL:
#   pd.read_sql("SELECT * FROM mart.mart_movie_features", engine)
# ------------------------------------------------------------------
movies  = pd.read_csv("data/raw/tmdb_5000_movies.csv")
credits = pd.read_csv("data/raw/tmdb_5000_credits.csv")

print(f"\nMovies shape  : {movies.shape}")
print(f"Credits shape : {credits.shape}")
print("\n--- First 3 Rows of Movies ---")
print(movies.head(3))


# ------------------------------------------------------------------
# STEP 3: DATA MERGING
# Merging the two tables on the movie id.
# ------------------------------------------------------------------
credits = credits.rename(columns={"movie_id": "id"})
df = movies.merge(credits, on="id", how="inner")
print(f"\nMerged dataset shape: {df.shape}")


# ------------------------------------------------------------------
# STEP 4: DATA CLEANING
# ------------------------------------------------------------------
print("\n--- Missing Data (After Merging) ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Remove rows with zero budget or revenue 
# (these are effectively missing values in this context)
df = df[(df["budget"] > 0) & (df["revenue"] > 0)]

# Fill missing runtime values with the median
df["runtime"] = df["runtime"].fillna(df["runtime"].median())

# Drop rows with missing release dates
df = df.dropna(subset=["release_date"])

print(f"\nRow count after cleaning: {len(df)}")


# ------------------------------------------------------------------
# STEP 5: FEATURE ENGINEERING
# Deriving meaningful features from raw columns.
# ------------------------------------------------------------------
def parse_json_list(text):
    """Converts a JSON-formatted text list into a Python list."""
    try:
        return ast.literal_eval(text)
    except Exception:
        return []

def count_items(text):
    """Returns the number of elements in a JSON list."""
    return len(parse_json_list(text))

def get_names(text, key="name", limit=None):
    """Extracts values for a specific key from a JSON list."""
    items = parse_json_list(text)
    names = [item.get(key, "") for item in items]
    return names[:limit] if limit else names

# Date-based features
df["release_date"] = pd.to_datetime(df["release_date"])
df["release_year"]  = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month

# Numerical features from JSON columns
df["genre_count"]               = df["genres"].apply(count_items)
df["production_company_count"]  = df["production_companies"].apply(count_items)
df["spoken_language_count"]     = df["spoken_languages"].apply(count_items)
df["cast_count"]                = df["cast"].apply(count_items)
df["crew_count"]                = df["crew"].apply(count_items)

# Return on Investment (ROI)
df["roi"] = df["revenue"] / df["budget"]

# Target variable: Is the movie's revenue in the top 20%?
# This threshold is defined by business logic — "high success" definition.
revenue_threshold = df["revenue"].quantile(0.80)
df["is_high_revenue"] = (df["revenue"] >= revenue_threshold).astype(int)

print(f"\nRevenue threshold (Top 20%): ${revenue_threshold:,.0f}")
print(f"High-revenue movie count: {df['is_high_revenue'].sum()}")
print(f"Low-revenue movie count: {(df['is_high_revenue'] == 0).sum()}")


# ------------------------------------------------------------------
# STEP 6: EDA — EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------
print("\n--- Creating EDA Visualizations ---")

# Setup visual style
sns.set_style("white")
pastel = {0: "#E6B3B3", 1: "#74C6A9"}

# Visualization 1: Target variable distribution
plt.figure(figsize=(7, 4))
ax = sns.countplot(x="is_high_revenue", data=df,
                   palette=[pastel[0], pastel[1]], saturation=1)
plt.title("Distribution of High-Revenue Movies", fontsize=15, pad=15)
plt.xticks([0, 1], ["Low Revenue", "High Revenue"], fontsize=12)
plt.ylabel("Number of Movies")
sns.despine()
for c in ax.containers:
    ax.bar_label(c, fmt="%d", fontsize=11, color="gray", padding=4)
plt.tight_layout()
plt.savefig("output_1_target_distribution.png", dpi=150)
plt.show()

# Visualization 2: Budget vs Revenue relationship
plt.figure(figsize=(8, 5))
colors = df["is_high_revenue"].map(pastel)
plt.scatter(df["budget"] / 1e6, df["revenue"] / 1e6,
            c=colors, alpha=0.5, edgecolors="none", s=30)
plt.title("Budget vs Revenue", fontsize=15, pad=15)
plt.xlabel("Budget (Million $)")
plt.ylabel("Revenue (Million $)")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=pastel[1], label="High Revenue"),
                   Patch(facecolor=pastel[0], label="Low Revenue")]
plt.legend(handles=legend_elements, frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig("output_2_budget_vs_revenue.png", dpi=150)
plt.show()

# Visualization 3: Average revenue by genre (Top 8 genres)
all_genres = []
for _, row in df.iterrows():
    genres = get_names(row["genres"])
    for g in genres:
        all_genres.append({"genre": g, "revenue": row["revenue"]})

genre_df = pd.DataFrame(all_genres)
top_genres = (genre_df.groupby("genre")["revenue"]
              .mean()
              .sort_values(ascending=False)
              .head(8))

plt.figure(figsize=(9, 5))
ax = sns.barplot(x=top_genres.values / 1e6, y=top_genres.index,
                 palette="Pastel1", saturation=1)
plt.title("Average Revenue by Genre (Top 8)", fontsize=15, pad=15)
plt.xlabel("Average Revenue (Million $)")
sns.despine(left=True, bottom=True)
plt.xticks([])
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f M", fontsize=9, color="gray", padding=4)
plt.tight_layout()
plt.savefig("output_3_genre_revenue.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# STEP 7: MODEL PREPARATION
# ------------------------------------------------------------------
FEATURES = [
    "budget",
    "runtime",
    "release_year",
    "release_month",
    "genre_count",
    "production_company_count",
    "spoken_language_count",
    "cast_count",
    "crew_count",
    "popularity",
    "vote_average",
    "vote_count",
]

df_model = df[FEATURES + ["is_high_revenue"]].dropna()

X = df_model[FEATURES]
y = df_model["is_high_revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTraining set : {X_train.shape[0]} movies")
print(f"Test set     : {X_test.shape[0]} movies")


# ------------------------------------------------------------------
# STEP 8: MODEL TRAINING
# ------------------------------------------------------------------

# --- Model 1: Logistic Regression (Baseline) ---
# Simple, explainable, provides a good reference point.
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# --- Model 2: Random Forest ---
# Can capture non-linear relationships, provides feature importance.
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)


# ------------------------------------------------------------------
# STEP 9: MODEL EVALUATION
# ------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    return y_pred, acc, auc

print("\n--- MODEL RESULTS ---")
y_pred_lr, acc_lr, auc_lr = evaluate_model("Logistic Regression", lr, X_test_scaled, y_test)
y_pred_rf, acc_rf, auc_rf = evaluate_model("Random Forest",       rf, X_test_scaled, y_test)

# Confusion Matrix — Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            cbar=False, annot_kws={"size": 16}, linewidths=0)
plt.title("Confusion Matrix — Random Forest", fontsize=14, pad=15)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.xticks([0.5, 1.5], ["Low Revenue", "High Revenue"], fontsize=11)
plt.yticks([0.5, 1.5], ["Low Revenue", "High Revenue"], fontsize=11, rotation=0, va="center")
plt.tight_layout()
plt.savefig("output_4_confusion_matrix.png", dpi=150)
plt.show()


# ------------------------------------------------------------------
# STEP 10: FEATURE IMPORTANCE
# ------------------------------------------------------------------
importances = (pd.Series(rf.feature_importances_, index=FEATURES)
               .sort_values(ascending=False))

print("\n--- Most Important Features (Random Forest) ---")
print(importances)

plt.figure(figsize=(9, 5))
ax = sns.barplot(x=importances.values, y=importances.index,
                 palette="Pastel1", saturation=1)
plt.title("Key Factors Determining Box Office Success", fontsize=14, pad=15)
sns.despine(left=True, bottom=True)
plt.xlabel("")
plt.xticks([])
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=9, color="gray", padding=4)
plt.tight_layout()
plt.savefig("output_5_feature_importance.png", dpi=150)
plt.show()


# ------------------------------------------------------------------
# STEP 11: MODEL COMPARISON
# ------------------------------------------------------------------
print("\n--- MODEL COMPARISON ---")
comparison = pd.DataFrame({
    "Model"   : ["Logistic Regression", "Random Forest"],
    "Accuracy": [acc_lr, acc_rf],
    "ROC-AUC" : [auc_lr, auc_rf],
})
print(comparison.to_string(index=False))


# ------------------------------------------------------------------
# STEP 12: PREDICTION EXAMPLE (Demo)
# Making a prediction for a new movie based on hypothetical features.
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("  PREDICTION DEMO")
print("=" * 50)

new_movie = pd.DataFrame([{
    "budget"                    : 150_000_000,
    "runtime"                   : 130,
    "release_year"              : 2024,
    "release_month"             : 6,       # Summer release
    "genre_count"               : 3,
    "production_company_count"  : 2,
    "spoken_language_count"     : 1,
    "cast_count"                : 40,
    "crew_count"                : 120,
    "popularity"                : 50.0,
    "vote_average"              : 7.5,
    "vote_count"                : 2000,
}])

new_movie_scaled = scaler.transform(new_movie)
prediction  = rf.predict(new_movie_scaled)[0]
probability = rf.predict_proba(new_movie_scaled)[0][1]

if prediction == 1:
    print(f"Result : ✅ This will be a HIGH REVENUE movie.")
else:
    print(f"Result : ❌ This will be a low revenue movie.")
print(f"Probability (high revenue): %{probability*100:.1f}")

print("\n" + "=" * 50)
print("  Project completed.")
print("=" * 50)