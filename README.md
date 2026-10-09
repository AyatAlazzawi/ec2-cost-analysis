# EC2 Cost Analysis

Implements Parts 1 and 2 of **Activity: Analyzing Amazon EC2 Instance Costs**.

Start the interactive classroom dashboard from this folder:

```bash
source .venv/bin/activate
python -m streamlit run streamlit_app.py
```

The dashboard opens in your browser and has four tabs: **Dataset Overview**, **Overall Pricing**,
**T2 vs T3 Analysis**, and **Linear Regression**. It reuses the lab's cleaning,
comparisons, and regression settings and displays the four saved charts in
`outputs/`. In the Linear Regression tab, enter the instance's memory and vCPU
count, then click **Predict On-Demand Cost**. The initial values are 4 GiB and
2 vCPUs. That prediction remains negative, with an explanation of this limitation
of linear regression. Running the dashboard does not change the dataset or saved charts.

## Deploy on Streamlit Community Cloud

Put this folder in a GitHub repository. Commit `streamlit_app.py`,
`requirements.txt`, `ec2dataset.csv`, and all four PNG charts in `outputs/`.
The `.gitignore` excludes the local virtual environment and backup files.

For a new repository, first create an **empty** repository named
`ec2-cost-analysis` on GitHub (do not add a README or `.gitignore` there).
Then run these commands from this folder, replacing `YOUR_USERNAME` with your
GitHub username:

```bash
git init -b main
git add .gitignore README.md analysis.py streamlit_app.py requirements.txt ec2dataset.csv outputs/*.png
git commit -m "Add EC2 Cost Analysis Streamlit dashboard"
git remote add origin https://github.com/YOUR_USERNAME/ec2-cost-analysis.git
git push -u origin main
```

At [share.streamlit.io/deploy](https://share.streamlit.io/deploy), sign in with
GitHub, choose **Create app** → **Yup, I have an app**, and enter:

- **Repository:** the GitHub repository containing this folder
- **Branch:** `main` (or the branch you pushed)
- **Main file path:** `streamlit_app.py` if this folder is the repository root,
  or `ec2-cost-analysis/streamlit_app.py` if it is inside a larger repository

Click **Deploy**. The app uses no secrets. After it opens, check that all four
tabs and saved figures load and that the 4 GiB / 2 vCPU prediction is
`$-0.3993/hr`. If the app fails to build, open **Manage app** to read its logs.

Run from this folder using the existing virtual environment:

```bash
.venv/bin/python analysis.py
```

The script prints the assignment summaries, comparisons, outliers, regression metrics,
and required 4 GiB / 2 vCPU prediction. It saves four figures automatically:

- `outputs/overall_pricing.png`
- `outputs/t2_pricing.png`
- `outputs/t3_pricing.png`
- `outputs/actual_vs_predicted.png`

The only model is the required `LinearRegression`, using memory and vCPUs with an
80/20 split and `random_state=42`. The negative prediction is preserved and explained.
The family filter follows the handout's name-prefix code; therefore its T3 group
includes T3A instances.

Use `analysis.py`, `ec2dataset.csv`, and the four PNGs for your submission, subject
to the LMS instructions. `requirements.txt` lists the dependencies. `backups/`
contains preserved script versions; do not submit it or `.venv`.
