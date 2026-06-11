"""
AURA VECTOR — All-in-One Enterprise AI Dashboard
Run: streamlit run aura_vector_all_in_one.py

Features included:
- Secure login with hashed password + lockout after 3 wrong attempts (configurable)
- File upload (CSV / XLSX)
- Robust automatic data cleaning (numeric/categorical/date handling, outlier flagging)
- Numeric analysis + correlation heatmap (numeric-only)
- Attrition Radar (classification) with explainability (SHAP if available, fallback feature importances)
- Business Loss Radar (rule-based leakage estimator using common columns)
- Prescriptive Playbooks (one-click retention email template + create ticket)
- Time-series forecasting (Prophet if available, fallback rolling forecast)
- Anomaly detection (IsolationForest)
- Natural-language lightweight query (average/max/min/groupby)
- Alerts (Slack webhook optional) and threshold-based notifications
- Model governance & audit log (audit_log.csv)
- Role-based view (mask sensitive columns for non-admin)
- Auto executive report (text summary)
- All-in-one single-file app, with clear comments and graceful fallbacks
"""

import os
import time
import hashlib
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Optional libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------------------------
# Configuration (change as needed)
# ---------------------------
DEFAULT_PASSWORD = "admin123"            # change before demo for security
LOCKOUT_SECONDS = 30                     # base lockout after 3 wrong attempts
MAX_ATTEMPTS = 3
AUDIT_LOG = "audit_log.csv"
SLACK_WEBHOOK = os.environ.get("AURA_SLACK_WEBHOOK", "")  # optional: set env var for real alerts

# ---------------------------
# Utility functions
# ---------------------------
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def append_audit(entry: dict):
    entry["timestamp"] = now_ts()
    df_entry = pd.DataFrame([entry])
    if os.path.exists(AUDIT_LOG):
        df_entry.to_csv(AUDIT_LOG, mode="a", header=False, index=False)
    else:
        df_entry.to_csv(AUDIT_LOG, index=False)

def send_slack_alert(message: str):
    if not SLACK_WEBHOOK:
        return False, "No webhook configured"
    try:
        import requests
        resp = requests.post(SLACK_WEBHOOK, json={"text": message}, timeout=5)
        return resp.ok, resp.text
    except Exception as e:
        return False, str(e)

def safe_read(file):
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, encoding="utf-8", errors="replace")

# ---------------------------
# Streamlit App Start
# ---------------------------
st.set_page_config(page_title="AURA VECTOR — Enterprise AI", layout="wide", page_icon="🔮")
st.title("🔮 AURA VECTOR — Enterprise AI Dashboard (All-in-One)")

# ---------------------------
# Session state init
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "lockout_until" not in st.session_state:
    st.session_state.lockout_until = 0
if "user_role" not in st.session_state:
    st.session_state.user_role = "viewer"  # viewer / analyst / admin

# ---------------------------
# LOGIN / SIGNUP PANEL
# ---------------------------
with st.sidebar.expander("🔐 Login / Security", expanded=True):
    st.write("Secure access to the dashboard.")
    if not st.session_state.authenticated:
        if st.session_state.lockout_until > time.time():
            remaining = int(st.session_state.lockout_until - time.time())
            st.error(f"Too many wrong attempts. Try again in {remaining} seconds.")
        else:
            pw = st.text_input("Password", type="password", key="pw_input")
            role = st.selectbox("Role", ["viewer", "analyst", "admin"], index=0)
            if st.button("Login"):
                if hash_pw(pw) == hash_pw(DEFAULT_PASSWORD):
                    st.session_state.authenticated = True
                    st.session_state.user_role = role
                    st.success("Login successful")
                    append_audit({"action":"login_success","role":role})
                else:
                    st.session_state.login_attempts += 1
                    append_audit({"action":"login_failed","attempts":st.session_state.login_attempts})
                    st.error("Wrong password")
                    if st.session_state.login_attempts >= MAX_ATTEMPTS:
                        st.session_state.lockout_until = time.time() + LOCKOUT_SECONDS
                        st.session_state.login_attempts = 0
                        st.warning(f"Locked out for {LOCKOUT_SECONDS} seconds")
    else:
        st.success(f"Authenticated as {st.session_state.user_role}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_role = "viewer"
            st.experimental_rerun()

# If not authenticated, show limited demo info and stop
if not st.session_state.authenticated:
    st.info("Please login from the sidebar to access full features. Use the password provided for demo.")
    st.stop()

# ---------------------------
# Main UI: Data Upload
# ---------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV / Excel (HR or Sales sample)", type=["csv","xlsx","xls"])
use_sample = st.sidebar.checkbox("Use sample HR attrition dataset", value=True)

if use_sample and uploaded is None:
    # create sample dataframe (same as earlier)
    sample = pd.DataFrame([
        [101,29,"Sales","Sales Executive",35000,2,"Yes","Yes"],
        [102,41,"HR","HR Specialist",42000,10,"No","No"],
        [103,35,"IT","Software Engineer",60000,5,"Yes","No"],
        [104,28,"Finance","Accountant",38000,3,"No","Yes"],
        [105,45,"Sales","Sales Manager",75000,15,"No","No"],
        [106,32,"IT","Data Analyst",52000,4,"Yes","No"],
        [107,39,"Finance","Financial Analyst",58000,8,"Yes","No"],
        [108,26,"HR","Recruiter",31000,1,"Yes","Yes"],
        [109,30,"IT","Software Engineer",55000,3,"No","No"],
        [110,50,"Sales","Sales Director",90000,20,"No","No"],
        [111,27,"Finance","Accountant",36000,2,"Yes","Yes"],
        [112,34,"HR","HR Specialist",40000,6,"No","No"],
        [113,31,"IT","Data Scientist",65000,4,"Yes","No"],
        [114,29,"Sales","Sales Executive",37000,2,"Yes","Yes"],
        [115,42,"Finance","Financial Analyst",60000,12,"No","No"],
        [116,33,"IT","Software Engineer",57000,5,"Yes","No"],
        [117,38,"HR","Recruiter",45000,7,"No","No"],
        [118,25,"Sales","Sales Executive",34000,1,"Yes","Yes"],
        [119,36,"Finance","Accountant",41000,6,"No","No"],
        [120,40,"IT","Data Analyst",56000,9,"Yes","No"],
    ], columns=["EmployeeID","Age","Department","JobRole","MonthlyIncome","YearsAtCompany","OverTime","Attrition"])
    df = sample.copy()
else:
    if uploaded:
        df = safe_read(uploaded)
    else:
        st.warning("Upload a dataset or enable sample dataset to proceed.")
        st.stop()

st.subheader("Raw Data Preview")
st.dataframe(df.head(50))

# ---------------------------
# Data Cleaning & Feature Engineering
# ---------------------------
st.markdown("### 🧹 Automatic Data Cleaning & Feature Engineering")
df_clean = df.copy()

# Standardize column names
df_clean.columns = [str(c).strip() for c in df_clean.columns]

# Convert obvious numeric columns
for col in df_clean.columns:
    if df_clean[col].dtype == object:
        # try numeric conversion
        try:
            df_clean[col] = pd.to_numeric(df_clean[col].str.replace(",",""), errors="raise")
        except Exception:
            pass

# Date parsing: detect columns with date-like strings
for col in df_clean.select_dtypes(include=['object']).columns:
    sample_vals = df_clean[col].dropna().astype(str).head(10).tolist()
    if any("-" in v or "/" in v for v in sample_vals):
        try:
            parsed = pd.to_datetime(df_clean[col], errors="coerce")
            if parsed.notna().sum() > 0:
                df_clean[col] = parsed
                df_clean[f"{col}_Year"] = df_clean[col].dt.year
                df_clean[f"{col}_Month"] = df_clean[col].dt.month
                df_clean[f"{col}_DayOfWeek"] = df_clean[col].dt.dayofweek
        except Exception:
            pass

# Numeric imputation
num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    df_clean[c] = df_clean[c].fillna(df_clean[c].median())

# Categorical imputation
cat_cols = df_clean.select_dtypes(include=['object','category']).columns.tolist()
for c in cat_cols:
    if df_clean[c].isna().any():
        df_clean[c] = df_clean[c].fillna(df_clean[c].mode().iloc[0] if not df_clean[c].mode().empty else "Unknown")

# Flag outliers (IQR method) for numeric columns
outlier_flags = pd.DataFrame(index=df_clean.index)
for c in num_cols:
    q1 = df_clean[c].quantile(0.25)
    q3 = df_clean[c].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        outlier_flags[f"{c}_outlier"] = False
    else:
        outlier_flags[f"{c}_outlier"] = ((df_clean[c] < (q1 - 1.5*iqr)) | (df_clean[c] > (q3 + 1.5*iqr)))

if not outlier_flags.empty:
    df_clean = pd.concat([df_clean, outlier_flags], axis=1)

df_clean = df_clean.drop_duplicates().reset_index(drop=True)
st.success("Data cleaned: missing values handled, dates parsed, outliers flagged.")
st.write(df_clean.describe(include='all').transpose())

# ---------------------------
# Role-based column masking
# ---------------------------
st.sidebar.header("Access Controls")
role = st.session_state.user_role
if role != "admin":
    sensitive_cols = [c for c in df_clean.columns if "Salary" in c or "Income" in c or "MonthlyIncome" in c or "EmployeeID" in c]
    if sensitive_cols:
        st.sidebar.info(f"Sensitive columns hidden for role: {role}")
        display_df = df_clean.drop(columns=[c for c in sensitive_cols if c in df_clean.columns])
    else:
        display_df = df_clean.copy()
else:
    display_df = df_clean.copy()

st.subheader("Cleaned Data (role-based view)")
st.dataframe(display_df.head(100))

# ---------------------------
# Numeric Analysis & Correlation
# ---------------------------
st.markdown("### 📈 Numeric Analysis & Correlation")
numeric_df = df_clean.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 2:
    corr = numeric_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", title="Correlation Heatmap (numeric only)")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.warning("Not enough numeric columns for correlation heatmap.")

# ---------------------------
# Anomaly Detection
# ---------------------------
st.markdown("### ⚠️ Anomaly Detection")
if numeric_df.shape[0] >= 5 and numeric_df.shape[1] >= 1:
    iso = IsolationForest(contamination=0.05, random_state=42)
    try:
        iso_preds = iso.fit_predict(numeric_df.fillna(0))
        df_clean["_anomaly_flag"] = (iso_preds == -1)
        st.write("Anomalies flagged:", int(df_clean["_anomaly_flag"].sum()))
        st.dataframe(df_clean[df_clean["_anomaly_flag"]].head(20))
        append_audit({"action":"anomaly_detection","anomalies":int(df_clean["_anomaly_flag"].sum())})
    except Exception as e:
        st.error("Anomaly detection failed: " + str(e))
else:
    st.info("Not enough numeric data for anomaly detection.")

# ---------------------------
# Business Loss Radar (rule-based)
# ---------------------------
st.markdown("### 💸 Business Loss Radar (Rule-based leakage estimator)")
# Look for common leakage-related columns; if not present, create proxy signals
leak_score = pd.Series(0, index=df_clean.index, dtype=float)

# Common columns that indicate leakage if present
if "RefundAmount" in df_clean.columns:
    leak_score += df_clean["RefundAmount"].fillna(0) * 1.0
if "Returns" in df_clean.columns:
    leak_score += df_clean["Returns"].fillna(0) * 1000
if "DeliveryDelayDays" in df_clean.columns:
    leak_score += df_clean["DeliveryDelayDays"].fillna(0) * 500

# If none present, use proxy: high refund-like behavior from negative income or outlier flags
if leak_score.sum() == 0:
    proxies = [c for c in df_clean.columns if "outlier" in c or "Return" in c or "Refund" in c]
    if proxies:
        leak_score += df_clean[proxies].sum(axis=1).fillna(0) * 1000
    else:
        # fallback: use high variance in monthly income as proxy for suspicious records
        if "MonthlyIncome" in df_clean.columns:
            leak_score += (df_clean["MonthlyIncome"].std() - (df_clean["MonthlyIncome"] - df_clean["MonthlyIncome"].mean()).abs()).fillna(0).abs()

df_clean["_leak_score"] = leak_score
top_leaks = df_clean.sort_values("_leak_score", ascending=False).head(10)
st.write("Top potential leakage rows (by leak score):")
st.dataframe(top_leaks[["Department","JobRole","_leak_score"]].head(10))

estimated_monthly_leak = df_clean["_leak_score"].sum()
st.info(f"Estimated monthly leakage (proxy): {estimated_monthly_leak:.2f}")
append_audit({"action":"leakage_estimate","estimated_leak":float(estimated_monthly_leak)})

# ---------------------------
# Attrition Radar + Explainability
# ---------------------------
st.markdown("### 🧭 Attrition Radar with Explainability")
# If Attrition column exists or user selects a target
target_default = "Attrition" if "Attrition" in df_clean.columns else None
target_col = st.selectbox("Select target column for Attrition Radar (classification)", options=[None] + list(df_clean.columns), index=(1 if target_default else 0))
if target_col:
    y = df_clean[target_col].copy()
    # Only proceed if target is binary-like or categorical
    if y.nunique() <= 10:
        X = df_clean.drop(columns=[target_col])
        # Keep only reasonable features (drop datetime objects)
        X = X.select_dtypes(exclude=["datetime","timedelta"])
        # Encode categorical columns
        X_enc = pd.get_dummies(X, drop_first=True)
        # Align shapes
        X_enc = X_enc.fillna(0)
        # If target is object, encode
        if y.dtype == object or y.dtype.name == "category":
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            classes = list(le.classes_)
        else:
            # if numeric but small unique values, treat as classification if <=10 unique
            if y.nunique() <= 10:
                le = LabelEncoder()
                y_enc = le.fit_transform(y.astype(str))
                classes = list(le.classes_)
            else:
                st.warning("Target appears numeric with many unique values; Attrition Radar expects a categorical/binary target.")
                y_enc = None

        if y_enc is not None:
            try:
                X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42)

            # Choose model: XGBoost if available else RandomForest
            if XGB_AVAILABLE:
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=200, random_state=42)

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"Attrition model trained. Test accuracy: {acc:.2f}")
                append_audit({"action":"attrition_model_trained","accuracy":float(acc)})
            except Exception as e:
                st.error("Model training failed: " + str(e))
                model = None

            # Explainability
            if model is not None:
                st.markdown("#### Explainability (Top features)")
                try:
                    if SHAP_AVAILABLE:
                        explainer = shap.Explainer(model, X_train)
                        shap_values = explainer(X_test)
                        # compute mean absolute shap per feature
                        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                        feat_imp = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False).head(10)
                        st.bar_chart(feat_imp)
                    else:
                        # fallback to feature_importances_
                        if hasattr(model, "feature_importances_"):
                            fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
                            st.bar_chart(fi)
                        else:
                            st.info("Explainability library not available; feature importances not found.")
                except Exception as e:
                    st.error("Explainability failed: " + str(e))

                # Show top at-risk employees (predict on full dataset)
                try:
                    probs = model.predict_proba(X_enc)[:,1] if hasattr(model, "predict_proba") else model.predict(X_enc)
                    df_clean["_risk_score"] = probs
                    top_risk = df_clean.sort_values("_risk_score", ascending=False).head(20)
                    st.write("Top risk employees (by predicted risk):")
                    st.dataframe(top_risk[[c for c in df_clean.columns if c in ["EmployeeID","Department","JobRole","MonthlyIncome","YearsAtCompany","OverTime","_risk_score"]]].head(20))
                    append_audit({"action":"attrition_risk_scored","top_risk_count":int(top_risk.shape[0])})
                except Exception as e:
                    st.error("Risk scoring failed: " + str(e))
        else:
            st.warning("Could not encode target for classification.")
    else:
        st.warning("Target has too many unique values for classification.")

# ---------------------------
# Prescriptive Playbooks (one-click actions)
# ---------------------------
st.markdown("### 🛠️ Prescriptive Playbooks")
st.write("Select a flagged employee from the Attrition Radar to generate retention playbook and actions.")
if "_risk_score" in df_clean.columns:
    selected_idx = st.selectbox("Select employee index (from cleaned data)", options=df_clean.index.tolist())
    if selected_idx is not None:
        emp = df_clean.loc[selected_idx]
        st.write(emp[["EmployeeID","Department","JobRole","MonthlyIncome","YearsAtCompany","OverTime","_risk_score"]])
        # Generate playbook suggestions based on top drivers (simple rules)
        playbook = []
        if emp.get("OverTime","No") == "Yes" or emp.get("OverTime",False):
            playbook.append("Reduce overtime; discuss workload redistribution.")
        if emp.get("YearsAtCompany",0) <= 2:
            playbook.append("Offer mentorship and career path discussion.")
        if "MonthlyIncome" in emp and emp["MonthlyIncome"] < df_clean["MonthlyIncome"].median():
            playbook.append("Review compensation; consider retention bonus.")
        if emp.get("_leak_score",0) > 0:
            playbook.append("Investigate transaction anomalies linked to this employee.")
        if not playbook:
            playbook.append("Schedule 1:1 with manager to understand concerns.")

        st.markdown("**Suggested Playbook**")
        for p in playbook:
            st.write("- " + p)

        # One-click actions
        if st.button("Generate Retention Email Template"):
            manager = "manager@company.com"
            subject = f"Retention: Quick 1:1 for Employee {emp.get('EmployeeID','N/A')}"
            body = f"""Hi Manager,

I recommend scheduling a short 1:1 with {emp.get('EmployeeID','Employee')} (Role: {emp.get('JobRole','N/A')}) due to elevated attrition risk ({emp.get('_risk_score',0):.2f}).

Suggested actions:
- Discuss workload and overtime.
- Review compensation and career path.
- Offer mentorship/training.

Regards,
AURA VECTOR
"""
            st.code(f"Subject: {subject}\n\n{body}")
            append_audit({"action":"playbook_email_generated","employee":str(emp.get("EmployeeID"))})

        if st.button("Create Retention Ticket (Audit)"):
            ticket = {"action":"create_ticket","employee":str(emp.get("EmployeeID")),"reason":"High attrition risk","risk_score":float(emp.get("_risk_score",0))}
            append_audit(ticket)
            st.success("Ticket created and logged in audit.")

# ---------------------------
# Forecasting (time-series)
# ---------------------------
st.markdown("### 📅 Time-series Forecasting")
# Detect a date column and a numeric target for forecasting
date_cols = df_clean.select_dtypes(include=["datetime"]).columns.tolist()
if date_cols:
    date_col = st.selectbox("Select date column for forecasting", options=date_cols)
    numeric_targets = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_targets:
        ts_target = st.selectbox("Select numeric target to forecast", options=numeric_targets)
        if st.button("Run Forecast"):
            ts_df = df_clean[[date_col, ts_target]].dropna().rename(columns={date_col:"ds", ts_target:"y"})
            ts_df = ts_df.groupby("ds").sum().reset_index()
            if PROPHET_AVAILABLE:
                try:
                    m = Prophet()
                    m.fit(ts_df)
                    future = m.make_future_dataframe(periods=30, freq='D')
                    forecast = m.predict(future)
                    fig = px.line(forecast, x="ds", y="yhat", title="Prophet Forecast")
                    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper', line=dict(dash='dash'))
                    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower', line=dict(dash='dash'))
                    st.plotly_chart(fig, use_container_width=True)
                    append_audit({"action":"forecast_prophet","target":ts_target})
                except Exception as e:
                    st.error("Prophet forecasting failed: " + str(e))
            else:
                # fallback: simple rolling mean forecast
                ts_df = ts_df.set_index("ds").asfreq('D').fillna(method='ffill')
                ts_df["rolling_mean"] = ts_df["y"].rolling(window=7, min_periods=1).mean()
                future_idx = pd.date_range(ts_df.index.max()+pd.Timedelta(days=1), periods=30, freq='D')
                future_vals = [ts_df["rolling_mean"].iloc[-1]] * len(future_idx)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df["y"], name="Actual"))
                fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df["rolling_mean"], name="RollingMean"))
                fig.add_trace(go.Scatter(x=future_idx, y=future_vals, name="Forecast"))
                st.plotly_chart(fig, use_container_width=True)
                append_audit({"action":"forecast_rolling","target":ts_target})
    else:
        st.info("No numeric columns available for forecasting.")
else:
    st.info("No date columns detected for time-series forecasting. Add a date column to use forecasting.")

# ---------------------------
# Natural Language Query (lightweight)
# ---------------------------
st.markdown("### 💬 Ask Your Data (Natural-language lite)")
nlq = st.text_input("Ask (e.g., 'average MonthlyIncome by Department', 'max MonthlyIncome')", key="nlq_input")
if nlq:
    q = nlq.lower()
    try:
        if "average" in q or "avg" in q:
            # parse "average <col> by <group>"
            parts = q.replace(",", "").split()
            if "by" in parts:
                by_idx = parts.index("by")
                col = parts[1]
                group = parts[by_idx+1]
                if col in df_clean.columns and group in df_clean.columns:
                    res = df_clean.groupby(group)[col].mean().sort_values(ascending=False)
                    st.write(res)
                else:
                    st.warning("Columns not found for groupby average.")
            else:
                # average of a column
                for c in df_clean.columns:
                    if c.lower() in q:
                        st.write(f"Average {c}: {df_clean[c].mean()}")
                        break
                else:
                    st.warning("Could not detect column for average.")
        elif "max" in q:
            for c in df_clean.columns:
                if c.lower() in q:
                    st.write(f"Max {c}: {df_clean[c].max()}")
                    break
            else:
                st.warning("Could not detect column for max.")
        elif "min" in q:
            for c in df_clean.columns:
                if c.lower() in q:
                    st.write(f"Min {c}: {df_clean[c].min()}")
                    break
            else:
                st.warning("Could not detect column for min.")
        elif "show" in q and "top" in q:
            # show top N by column
            import re
            m = re.search(r"top\s+(\d+)\s+by\s+(\w+)", q)
            if m:
                n = int(m.group(1)); col = m.group(2)
                if col in df_clean.columns:
                    st.dataframe(df_clean.sort_values(col, ascending=False).head(n))
                else:
                    st.warning("Column not found.")
            else:
                st.warning("Try 'show top 5 by MonthlyIncome'")
        else:
            st.info("Query not recognized by lite parser. Try 'average', 'max', 'min', or 'show top N by <col>'.")
    except Exception as e:
        st.error("NLQ failed: " + str(e))

# ---------------------------
# Alerts: threshold-based
# ---------------------------
st.markdown("### 🔔 Alerts & Notifications")
alert_threshold = st.number_input("Alert threshold for top risk score (0-1)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
if "_risk_score" in df_clean.columns:
    top_risk_val = df_clean["_risk_score"].max()
    st.write(f"Top predicted risk: {top_risk_val:.2f}")
    if top_risk_val >= alert_threshold:
        st.warning("Top risk exceeds threshold. You can send an alert.")
        if st.button("Send Alert (Slack)"):
            msg = f"AURA VECTOR Alert: Top attrition risk {top_risk_val:.2f} exceeds threshold {alert_threshold}"
            ok, resp = send_slack_alert(msg)
            if ok:
                st.success("Alert sent to Slack.")
                append_audit({"action":"alert_sent","top_risk":float(top_risk_val)})
            else:
                st.error("Alert failed: " + resp)
    else:
        st.info("No alert: top risk below threshold.")
else:
    st.info("No risk scores available to evaluate alerts.")

# ---------------------------
# Auto Report Generator
# ---------------------------
st.markdown("### 📑 Executive Auto Report")
report_title = st.text_input("Report title", value="AURA VECTOR Executive Summary")
if st.button("Generate Report Summary"):
    summary_lines = []
    summary_lines.append(report_title)
    summary_lines.append(f"Generated: {now_ts()}")
    summary_lines.append(f"Dataset shape: {df_clean.shape}")
    if "_risk_score" in df_clean.columns:
        summary_lines.append(f"Top attrition risk: {df_clean['_risk_score'].max():.2f}")
        summary_lines.append(f"Avg attrition risk: {df_clean['_risk_score'].mean():.2f}")
    summary_lines.append(f"Estimated monthly leakage (proxy): {estimated_monthly_leak:.2f}")
    summary_lines.append("Top suggested actions:")
    summary_lines.append("- Review top at-risk employees and trigger playbooks.")
    summary_lines.append("- Investigate top leakage rows and reconcile transactions.")
    summary_lines.append("- Schedule forecasting review for next quarter.")
    report_text = "\n".join(summary_lines)
    st.text_area("Executive Summary", report_text, height=300)
    append_audit({"action":"report_generated","title":report_title})

# ---------------------------
# Model Governance & Audit Log Viewer
# ---------------------------
st.markdown("### 🧾 Model Governance & Audit Log")
if os.path.exists(AUDIT_LOG):
    try:
        audit_df = pd.read_csv(AUDIT_LOG)
        st.dataframe(audit_df.tail(200))
    except Exception as e:
        st.error("Failed to read audit log: " + str(e))
else:
    st.info("No audit log found yet. Actions will be logged to audit_log.csv in app folder.")

# ---------------------------
# Final notes and safe shutdown
# ---------------------------
st.markdown("---")
st.write("AURA VECTOR — demo-ready. Use the sidebar to login as admin for full access. Change DEFAULT_PASSWORD in the script for production. For production-grade deployment, move secrets to environment variables, enable HTTPS, and use a proper model registry and DB for audit logs.")
