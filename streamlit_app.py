"""Classroom dashboard for the existing EC2 Cost Analysis lab.

Keep the calculations aligned with analysis.py. Importing that script would run
the entire lab and overwrite its figures, so its analysis steps are reused here
without the printing or plotting side effects.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
COST_COLUMNS = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost",
]
COMPARISON_COLUMNS = ["Name", "On Demand", "Linux Reserved cost"]
FEATURES = ["Instance Memory", "vCPUs"]


@st.cache_data
def load_data():
    """Read the lab snapshot and apply the original price conversions."""
    raw = pd.read_csv(ROOT / "ec2dataset.csv")
    data = raw.copy()
    for column in COST_COLUMNS:
        data[column] = pd.to_numeric(
            data[column].str.replace("$", "", regex=False)
            .str.replace(" hourly", "", regex=False)
            .str.replace(",", "", regex=False),
            errors="coerce",
        )
    return raw, data


def detect_outliers(data, column):
    """Use the lab's 1.5 × interquartile range rule."""
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    return data[
        (data[column] < q1 - 1.5 * iqr) | (data[column] > q3 + 1.5 * iqr)
    ]


def filter_instance_family(data, family):
    # Match the handout and analysis.py: the T3 prefix includes T3A.
    return data[data["Name"].str.startswith(family, na=False)]


@st.cache_data
def run_regression(data):
    """Reproduce the lab model, including its unmodified 4 GiB prediction."""
    data = data.copy()
    data["Instance Memory"] = pd.to_numeric(
        data["Instance Memory"].str.extract(r"([\d.]+)")[0], errors="coerce"
    )
    data["vCPUs"] = pd.to_numeric(
        data["vCPUs"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    cleaned = data.dropna(subset=["On Demand", *FEATURES])
    x_train, x_test, y_train, y_test = train_test_split(
        cleaned[FEATURES], cleaned["On Demand"], test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "data": cleaned,
        "model": model,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mse,
        "rmse": mse ** 0.5,
        "intercept": model.intercept_,
        "coefficients": model.coef_,
    }


def show_chart(filename, caption):
    """Display a saved figure without regenerating or modifying it."""
    path = ROOT / "outputs" / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Saved chart missing: outputs/{filename}.")


def show_prices(frame):
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            column: st.column_config.NumberColumn(column, format="$%.4f")
            for column in COST_COLUMNS
            if column in frame.columns
        },
    )


def show_summary(frame):
    st.dataframe(frame[COST_COLUMNS].describe().style.format("{:.4f}"), width="stretch")


def main():
    st.set_page_config(page_title="EC2 Cost Analysis", page_icon="☁️", layout="wide")
    st.caption("AWS LAB · COMPUTE PRICING")
    st.title("EC2 Cost Analysis")
    st.write("Explore instance prices, compare T2 and T3, and evaluate a simple cost model.")
    st.caption("Source: ec2dataset.csv · Dataset snapshot, not live AWS prices · Costs in USD/hour")

    try:
        raw, data = load_data()
    except FileNotFoundError:
        st.error("Place ec2dataset.csv beside streamlit_app.py to open the dashboard.")
        st.stop()

    with st.container(border=True):
        instances, columns, prices = st.columns(3)
        instances.metric("Instances", f"{len(data):,}")
        columns.metric("Dataset columns", len(data.columns))
        prices.metric("Pricing options", len(COST_COLUMNS))

    overview, overall, families, regression = st.tabs(
        ["Dataset Overview", "Overall Pricing", "T2 vs T3 Analysis", "Linear Regression"]
    )

    with overview:
        st.subheader("Explore the dataset")
        view = st.radio("Dataset view", ["Original CSV", "Cleaned prices"], horizontal=True)
        if view == "Original CSV":
            st.dataframe(raw, hide_index=True, width="stretch", height=350)
        else:
            show_prices(data)
        st.caption(
            "Price cleaning removes dollar signs, commas, and 'hourly', then converts "
            "values to numbers. Unavailable prices become missing values."
        )

        st.subheader("Summary statistics")
        st.caption("Hourly prices across all instances. Count includes only available numeric prices.")
        show_summary(data)

        st.subheader("Missing values")
        st.dataframe(
            pd.DataFrame({
                "Original CSV": raw.isna().sum(),
                "After price cleaning": data.isna().sum(),
            }).rename_axis("Column"),
            width="stretch",
        )
        st.caption(
            "Text such as 'unavailable' is not counted as missing in the original CSV. "
            "The cleaned counts reveal these unavailable prices; they are not treated as zero."
        )

    with overall:
        st.subheader("Overall hourly pricing")
        low, median, high = st.columns(3)
        low.metric("Lowest On-Demand", f"${data['On Demand'].min():.4f}/hr")
        median.metric("Median On-Demand", f"${data['On Demand'].median():.4f}/hr")
        high.metric("Highest On-Demand", f"${data['On Demand'].max():.4f}/hr")

        show_chart("overall_pricing.png", "Cost distributions for the five pricing options.")
        st.caption(
            "Each box spans the middle 50% of prices; its line marks the median. "
            "Points beyond the whiskers are potential outliers."
        )

        st.subheader("10 lowest-cost EC2 instances")
        st.caption("Sorted by On-Demand price; rows must have both On-Demand and Linux Reserved prices.")
        cheapest = data[COMPARISON_COLUMNS].dropna().sort_values("On Demand")
        show_prices(cheapest.head(10))

        outliers = detect_outliers(data, "On Demand")
        with st.expander(f"On-Demand pricing outliers · {len(outliers)} instances"):
            st.write(
                "The lab flags prices below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR. "
                "These instances remain in the analysis and regression."
            )
            show_prices(outliers[["Name", *COST_COLUMNS]])

    with families:
        st.subheader("Compare T2 and T3 pricing")
        st.info("The lab matches instance name prefixes, so the T3 group also includes T3A instances.")
        t2 = filter_instance_family(data, "T2")
        t3 = filter_instance_family(data, "T3")
        for column, label, frame, chart in zip(
            st.columns(2),
            ["T2", "T3 (including T3A)"],
            [t2, t3],
            ["t2_pricing.png", "t3_pricing.png"],
        ):
            with column:
                st.metric(f"{label} instances", len(frame))
                st.caption(f"Median On-Demand: ${frame['On Demand'].median():.4f}/hr")
                show_chart(chart, f"{label} hourly price distributions; triangles mark means.")
                with st.expander(f"{label} summary statistics"):
                    show_summary(frame)
        st.caption(
            "The two saved charts use different vertical scales. Compare the numeric "
            "summaries as well as the shapes of the distributions."
        )

        st.subheader("10 lowest-cost T2 and T3 instances")
        st.caption("Sorted by On-Demand price; requires both On-Demand and Linux Reserved prices.")
        comparison = pd.concat([t2[COMPARISON_COLUMNS], t3[COMPARISON_COLUMNS]])
        show_prices(comparison.dropna().sort_values("On Demand").head(10))

    with regression:
        st.subheader("Predict On-Demand cost")
        st.write(
            "Linear regression uses memory (GiB) and vCPUs to predict hourly cost. "
            "Rows missing either feature or the target price are excluded. "
            "The lab uses an 80/20 train/test split with random_state=42."
        )
        result = run_regression(data)
        usable, train, test = st.columns(3)
        usable.metric("Usable instances", len(result["data"]))
        train.metric("Training instances", result["train_size"])
        test.metric("Testing instances", result["test_size"])

        st.subheader("Evaluation on the test set")
        mae, mse, rmse = st.columns(3)
        mae.metric("MAE · USD/hour", f"{result['mae']:.4f}", help="Mean absolute error.")
        mse.metric("MSE · (USD/hour)²", f"{result['mse']:.4f}", help="Mean squared error.")
        rmse.metric("RMSE · USD/hour", f"{result['rmse']:.4f}", help="Root mean squared error.")

        chart, prediction = st.columns([3, 2])
        with chart:
            show_chart("actual_vs_predicted.png", "The dashed line represents a perfect prediction.")
        with prediction:
            with st.container(border=True):
                st.subheader("Predict a new EC2 instance")
                with st.form("instance_cost_prediction"):
                    memory = st.number_input(
                        "Instance Memory (GiB)", min_value=0.0, value=4.0, step=0.5
                    )
                    vcpus = st.number_input(
                        "Number of vCPUs", min_value=0, value=2, step=1
                    )
                    predict_clicked = st.form_submit_button("Predict On-Demand Cost")
                if predict_clicked:
                    new_instance = pd.DataFrame([[memory, vcpus]], columns=FEATURES)
                    prediction_value = float(result["model"].predict(new_instance)[0])
                    st.success(f"Predicted On-Demand Cost: ${prediction_value:.4f} per hour")
                    if prediction_value < 0:
                        st.warning(
                            "This negative prediction is preserved exactly as the model produces it. "
                            "It is a limitation of linear regression, not a valid hourly price."
                        )
                st.write(
                    "Enter memory and vCPU values, then select Predict On-Demand Cost. "
                    "Linear regression does not constrain predictions to be nonnegative, "
                    "and memory and vCPUs alone do not describe every hardware difference."
                )

        with st.expander("Inspect the model and regression dataset"):
            st.write("Predicted cost = intercept + memory coefficient × GiB + vCPU coefficient × vCPUs")
            st.dataframe(
                pd.DataFrame({
                    "Term": ["Intercept", "Memory (per GiB)", "vCPUs (per vCPU)"],
                    "Coefficient": [result["intercept"], *result["coefficients"]],
                }).style.format({"Coefficient": "{:.6f}"}),
                hide_index=True,
                width="stretch",
            )
            show_prices(result["data"][["Name", *FEATURES, "On Demand"]])
            st.caption(f"Excluded rows: {len(data) - len(result['data'])}.")

    st.divider()
    st.caption("EC2 Cost Analysis · Classroom lab · Figures from the original analysis")


if __name__ == "__main__":
    main()
