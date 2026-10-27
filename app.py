import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="E-Commerce Customer Intelligence", page_icon="🛍️", layout="wide")

FEATURES = ["Age", "Gender", "AnnualIncome", "OrdersPerMonth", "AvgOrderValue", "DaysSinceLastPurchase", "WebsiteVisitsPerMonth", "DiscountUsageRate", "SupportTickets"]

@st.cache_data
def make_demo_data(n=1200, seed=42):
    """Generate plausible demo customers and a behavioral repeat-purchase target."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 71, n)
    income = np.clip(rng.normal(58_000, 22_000, n), 15_000, 160_000)
    visits = np.clip(rng.poisson(7, n), 0, 35)
    recency = np.clip(rng.gamma(2.1, 18, n), 0, 180)
    orders = np.clip(rng.poisson(2.7, n) + visits * .10 - recency / 65, 0, 18).round().astype(int)
    aov = np.clip(rng.lognormal(np.log(72), .48, n) * (income / 58_000) ** .12, 8, 550)
    disc = np.clip(rng.beta(2.0, 5.0, n) + (income < 35_000) * .12, 0, 1)
    tickets = np.clip(rng.poisson(.65, n), 0, 8)
    gender = rng.choice(["Female", "Male", "Other"], n, p=[.49, .48, .03])
    # Higher engagement, order frequency, and recency of purchase increase repeat likelihood.
    logit = 1.15 + .19 * orders + .085 * visits - .025 * recency - .34 * tickets - .75 * disc
    probability = 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"CustomerID": [f"C{i:05d}" for i in range(1, n + 1)], "Age": age, "Gender": gender,
        "AnnualIncome": income.round(0), "OrdersPerMonth": orders, "AvgOrderValue": aov.round(2),
        "DaysSinceLastPurchase": recency.round(0).astype(int), "WebsiteVisitsPerMonth": visits,
        "DiscountUsageRate": disc.round(2), "SupportTickets": tickets,
        "RepeatPurchase": rng.binomial(1, probability)})

def validate_data(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return f"Missing required columns: {', '.join(missing)}"
    return None

def prep_numeric():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])

st.title("🛍️ E-Commerce Customer Intelligence")
st.caption("An interactive mini-project combining **K-Means customer segmentation** (unsupervised learning) and **Random Forest repeat-purchase prediction** (supervised learning).")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload customer CSV (optional)", type="csv", help="Use the demo data or upload a CSV with the required customer feature columns.")
    st.markdown("**Required columns:** " + ", ".join(FEATURES))
    st.markdown("Optional target: `RepeatPurchase` (0/1). If absent, the app uses the demo behavior label only for demo data; with uploaded data it generates no target and hides prediction.")

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        error = validate_data(raw)
        if error:
            st.error(error)
            st.stop()
        df = raw.copy()
        if "CustomerID" not in df:
            df.insert(0, "CustomerID", [f"C{i:05d}" for i in range(1, len(df) + 1)])
        data_source = "Uploaded CSV"
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    df = make_demo_data()
    data_source = "Generated demo data"

if len(df) < 10:
    st.error("Please provide at least 10 customer rows for analysis.")
    st.stop()

# Normalize expected types while allowing uploaded data to contain missing values.
for col in ["Age", "AnnualIncome", "OrdersPerMonth", "AvgOrderValue", "DaysSinceLastPurchase", "WebsiteVisitsPerMonth", "DiscountUsageRate", "SupportTickets"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Gender"] = df["Gender"].fillna("Unknown").astype(str)

with st.sidebar:
    st.divider()
    st.header("Analysis settings")
    k = st.slider("Customer segments (K)", 2, 8, 4)
    test_size = st.slider("Prediction test split", 0.15, 0.40, 0.25, 0.05)
    trees = st.slider("Random Forest trees", 50, 400, 150, 50)

# K-Means segmentation
seg_cols = ["AnnualIncome", "OrdersPerMonth", "AvgOrderValue", "DaysSinceLastPurchase", "WebsiteVisitsPerMonth", "DiscountUsageRate", "SupportTickets"]
segment_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10))])
try:
    df["Segment"] = segment_pipe.fit_predict(df[seg_cols]) + 1
except Exception as exc:
    st.error(f"Could not create segments. Check that numeric columns have valid values. Details: {exc}")
    st.stop()

st.caption(f"Data source: **{data_source}** · {len(df):,} customers")
metric_cols = st.columns(4)
metric_cols[0].metric("Customers", f"{len(df):,}")
metric_cols[1].metric("Average order value", f"₹{df['AvgOrderValue'].mean():,.0f}" if pd.notna(df["AvgOrderValue"].mean()) else "—")
metric_cols[2].metric("Avg. monthly orders", f"{df['OrdersPerMonth'].mean():.1f}" if pd.notna(df["OrdersPerMonth"].mean()) else "—")
metric_cols[3].metric("Customer segments", k)

pages = st.tabs(["Overview", "Customer Segments", "Repeat Purchase Model", "Explore Data"])
with pages[0]:
    st.subheader("Customer overview")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="AvgOrderValue", nbins=35, title="Average order value distribution", color_discrete_sequence=["#5B6CFF"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        seg_counts = df["Segment"].value_counts().sort_index().rename_axis("Segment").reset_index(name="Customers")
        fig = px.pie(seg_counts, names="Segment", values="Customers", title="Customers by segment", hole=.45)
        st.plotly_chart(fig, use_container_width=True)
    st.info("Use the segment and prediction tabs to turn these patterns into customer groups and actionable retention insights.")

with pages[1]:
    st.subheader("K-Means customer segments")
    if len(df) > k:
        try:
            sil = silhouette_score(StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[seg_cols])), df["Segment"] - 1, sample_size=min(1000, len(df)), random_state=42)
            st.metric("Silhouette score", f"{sil:.3f}", help="Ranges from -1 to 1; higher values indicate more clearly separated clusters.")
        except Exception:
            pass
    profile = df.groupby("Segment").agg(Customers=("CustomerID", "count"), Age=("Age", "mean"), AnnualIncome=("AnnualIncome", "mean"), OrdersPerMonth=("OrdersPerMonth", "mean"), AvgOrderValue=("AvgOrderValue", "mean"), DaysSinceLastPurchase=("DaysSinceLastPurchase", "mean"), WebsiteVisitsPerMonth=("WebsiteVisitsPerMonth", "mean"), DiscountUsageRate=("DiscountUsageRate", "mean"), SupportTickets=("SupportTickets", "mean")).round(2).reset_index()
    st.dataframe(profile, use_container_width=True, hide_index=True)
    fig = px.scatter(df, x="AnnualIncome", y="AvgOrderValue", color=df["Segment"].astype(str), size="OrdersPerMonth", hover_data=["CustomerID", "DaysSinceLastPurchase"], title="Segments by income and order value", labels={"color": "Segment"})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**How to interpret:** segment numbers are labels, not rankings. Compare the profile table to identify groups such as high-value loyal customers, at-risk customers, or discount-dependent shoppers.")
    st.download_button("Download segmented customers CSV", df.to_csv(index=False).encode("utf-8"), "segmented_customers.csv", "text/csv")

with pages[2]:
    st.subheader("Repeat-purchase prediction · Random Forest")
    if "RepeatPurchase" not in df.columns:
        st.warning("This uploaded dataset has no `RepeatPurchase` target, so supervised model training is unavailable. Add a column with 0 = no repeat purchase and 1 = repeat purchase to enable it.")
    else:
        target = pd.to_numeric(df["RepeatPurchase"], errors="coerce")
        model_df = df.loc[target.isin([0, 1])].copy()
        model_df["RepeatPurchase"] = target[target.isin([0, 1])].astype(int)
        counts = model_df["RepeatPurchase"].value_counts()
        if len(counts) < 2 or counts.min() < 2:
            st.warning("The target needs at least two examples of both classes (0 and 1) for a stratified train/test split.")
        else:
            num_cols = [c for c in FEATURES if c != "Gender"]
            cat_cols = ["Gender"]
            prep = ColumnTransformer([("num", prep_numeric(), num_cols), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)])
            X = model_df[FEATURES]
            y = model_df["RepeatPurchase"]
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                model = Pipeline([("prep", prep), ("rf", RandomForestClassifier(n_estimators=trees, class_weight="balanced", random_state=42, max_depth=12))])
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                a, b, c = st.columns(3)
                a.metric("Accuracy", f"{accuracy_score(y_test, pred):.1%}")
                b.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.1%}")
                c.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.1%}")
                cm = confusion_matrix(y_test, pred, labels=[0, 1])
                st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted no", "Predicted yes"], y=["Actual no", "Actual yes"], color_continuous_scale="Blues", title="Confusion matrix"), use_container_width=True)
                names = model.named_steps["prep"].get_feature_names_out()
                importance = pd.DataFrame({"Feature": names, "Importance": model.named_steps["rf"].feature_importances_}).sort_values("Importance", ascending=False).head(12)
                st.plotly_chart(px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Most influential model features", color_discrete_sequence=["#5B6CFF"]), use_container_width=True)
                st.caption("Metrics are evaluated on a held-out test split. They describe this dataset and do not guarantee future customer behavior.")
            except Exception as exc:
                st.error(f"Model training failed: {exc}")

with pages[3]:
    st.subheader("Customer records")
    search = st.text_input("Filter by customer ID or gender")
    display_df = df
    if search:
        display_df = df[df["CustomerID"].astype(str).str.contains(search, case=False, na=False) | df["Gender"].str.contains(search, case=False, na=False)]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button("Download current data as CSV", df.to_csv(index=False).encode("utf-8"), "customer_analysis.csv", "text/csv")

with st.expander("Project methodology (for your presentation)"):
    st.markdown("""
    - **Problem:** understand customer groups and predict whether a customer will make a repeat purchase.
    - **Unsupervised algorithm:** K-Means groups customers using income, order behavior, purchase recency, visits, discount use, and support tickets. Numeric values are imputed and standardized first.
    - **Supervised algorithm:** Random Forest classifies `RepeatPurchase` (0/1), using a stratified train/test split. Numeric and categorical features are prepared in a scikit-learn pipeline.
    - **Evaluation:** held-out accuracy, precision, recall, and confusion matrix; feature importance helps explain predictions.
    - **Limitations:** generated data is synthetic for demonstration, and cluster IDs are arbitrary. Replace it with real, appropriately anonymized customer data for meaningful business conclusions.
    """)
