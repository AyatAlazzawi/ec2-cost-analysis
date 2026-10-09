from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Save figures without opening interactive windows.
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

# Part 1: EC2 cost analysis
file_path = ROOT / "ec2dataset.csv"
data = pd.read_csv(file_path)

data.info()
print("Raw missing values:\n", data.isna().sum())
print(data.head())
cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace("$", "", regex=False)
                    .str.replace(" hourly", "", regex=False)
                    .str.replace(",", "", regex=False),
        errors="coerce"
    )

print(data[cost_columns].isnull().sum())


cost_summary = data[cost_columns].describe()

print("\nEC2 Cost Summary:")
print(cost_summary)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))

sns.boxplot(data=data[cost_columns], palette="Set2")

plt.title("Cost Comparison of Amazon EC2 Instances (Hourly)")
plt.ylabel("Hourly cost (USD/hour)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(OUTPUT / "overall_pricing.png", dpi=220, bbox_inches="tight")
plt.close()

def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]


outliers_on_demand = detect_outliers("On Demand")

print("\nOn-Demand Pricing Outliers:")
print(outliers_on_demand)
print("\nTotal outliers:", len(outliers_on_demand))

cost_comparison = data[
    ["Name", "On Demand", "Linux Reserved cost"]
].dropna().sort_values("On Demand")

print("\n10 Lowest-Cost EC2 Instances:")
print(cost_comparison.head(10))

def filter_instance_family(family):
    # Follow the handout: the T3 prefix also includes T3A instances.
    return data[data["Name"].str.startswith(family, na=False)]


t2_instances = filter_instance_family("T2")
t3_instances = filter_instance_family("T3")

t2_summary = t2_instances[cost_columns].describe()
t3_summary = t3_instances[cost_columns].describe()

print("\nT2 Instance Costs Summary:")
print(t2_summary)

print("\nT3 Instance Costs Summary:")
print(t3_summary)


plt.figure(figsize=(12, 6))
sns.boxplot(data=t2_instances[cost_columns], palette="Blues", showmeans=True)
plt.title("Cost Distribution for T2 Instances")
plt.ylabel("Hourly cost (USD/hour)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "t2_pricing.png", dpi=220, bbox_inches="tight")
plt.close()


plt.figure(figsize=(12, 6))
sns.boxplot(data=t3_instances[cost_columns], palette="Greens", showmeans=True)
plt.title("Cost Distribution for T3 Instances")
plt.ylabel("Hourly cost (USD/hour)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "t3_pricing.png", dpi=220, bbox_inches="tight")
plt.close()


comparison = pd.concat([
    t2_instances[["Name", "On Demand", "Linux Reserved cost"]],
    t3_instances[["Name", "On Demand", "Linux Reserved cost"]]
])

comparison_sorted = comparison.dropna().sort_values("On Demand")

print("\n10 Lowest-Cost T2 and T3 Instances:")
print(comparison_sorted.head(10))


# Part 2: Predict On-Demand cost using memory and vCPUs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data["Instance Memory"] = pd.to_numeric(
    data["Instance Memory"].str.extract(r"([\d.]+)")[0],
    errors="coerce"
)

data["vCPUs"] = pd.to_numeric(
    data["vCPUs"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

data_cleaned = data.dropna(
    subset=["On Demand", "Instance Memory", "vCPUs"]
)

print("\nMissing values after regression row filtering:\n", data_cleaned.isna().sum())

print("\nRegression Dataset:")
print(data_cleaned[["Instance Memory", "vCPUs", "On Demand"]].head())

print("\nTotal usable instances:", len(data_cleaned))

X = data_cleaned[["Instance Memory", "vCPUs"]]
y = data_cleaned["On Demand"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Intercept:", model.intercept_)
print("Model Coefficients:", model.coef_)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\nRegression Model Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7, color="b")

plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    linestyle="--"
)

plt.title("Actual vs Predicted On-Demand Costs")
plt.xlabel("Actual On-Demand cost (USD/hour)")
plt.ylabel("Predicted On-Demand cost (USD/hour)")

plt.tight_layout()
plt.savefig(OUTPUT / "actual_vs_predicted.png", dpi=220, bbox_inches="tight")
plt.close()

new_instance = pd.DataFrame(
    [[4, 2]],
    columns=["Instance Memory", "vCPUs"]
)

predicted_cost = model.predict(new_instance)

print(
    f"\nPredicted On-Demand Cost for 4 GiB, 2 vCPUs: "
    f"${predicted_cost[0]:.4f}"
)

# Report the required prediction without changing or clipping it.
print("A negative predicted price is a limitation of this linear model, not a valid hourly price.")
print("Saved the four assignment figures to", OUTPUT)

