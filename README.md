# E-Commerce Customer Intelligence

A ready-to-run Intelligent Systems mini-project demonstrating two ML approaches in Streamlit:

- **K-Means (unsupervised):** discovers customer segments from purchase and engagement behavior.
- **Random Forest (supervised):** predicts repeat purchase (0/1) and reports accuracy, precision, recall, confusion matrix, and feature importance.

The app generates 1,200 synthetic customer records by default, so it runs without an external dataset or network access. You can also upload your own CSV.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended):

   **Windows PowerShell**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies and start Streamlit:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. Open the localhost URL printed by Streamlit (usually `http://localhost:8501`).

## Uploading your own data

Upload a CSV with these feature columns:

`Age, Gender, AnnualIncome, OrdersPerMonth, AvgOrderValue, DaysSinceLastPurchase, WebsiteVisitsPerMonth, DiscountUsageRate, SupportTickets`

`CustomerID` is optional. Add `RepeatPurchase` with values `0` or `1` if you want supervised model results for your own data. Without it, customer segmentation and exploration still work.

## Suggested demo walkthrough

1. Show the overview metrics and charts.
2. Change the number of segments in the sidebar and inspect the segment profiles.
3. Explain how K-Means uses scaled behavior features to group similar customers.
4. Show the Random Forest evaluation and feature-importance chart.
5. Download the segmented customer CSV.

## Academic note

The built-in data and its repeat-purchase labels are synthetic and intended only for a classroom demonstration. Model quality on synthetic data is not evidence of real-world performance. Explain that K-Means cluster numbers are arbitrary labels, and describe supervised evaluation on a held-out test split.
