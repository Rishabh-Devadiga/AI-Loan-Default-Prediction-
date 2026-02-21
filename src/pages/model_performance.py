import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import resolve_artifact_dir


DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic": "Logistic Regression",
    "mlp": "Neural Network",
}


def _metric_card(label: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _simulate_roc_curve(auc: float, points: int = 120):
    x = np.linspace(0, 1, points)
    a = max(1.01, 1 / max(0.001, 1 - min(0.995, auc)))
    y = 1 - (1 - x) ** a
    return x, np.clip(y, 0, 1)


def app():
    st.title("Model Performance")
    st.caption("Compare and analyze machine learning model performance metrics")

    metrics_path = resolve_artifact_dir() / "model_metrics_summary.csv"
    if not metrics_path.exists():
        st.error(f"Metrics file not found: {metrics_path}")
        return

    metrics = pd.read_csv(metrics_path, index_col=0)
    metrics = metrics.rename(index=DISPLAY_NAMES).sort_index()

    tabs = st.tabs(list(metrics.index))
    for tab, model_name in zip(tabs, metrics.index):
        with tab:
            row = metrics.loc[model_name]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                _metric_card("Accuracy", f"{100 * row['accuracy']:.1f}%", "Correctly classified")
            with c2:
                _metric_card("Precision", f"{100 * row['precision']:.1f}%", "Of predicted defaults, how many are real")
            with c3:
                _metric_card("Recall", f"{100 * row['recall']:.1f}%", "Of real defaults, how many caught")
            with c4:
                _metric_card("AUC-ROC", f"{row['roc_auc']:.3f}", "Overall ranking ability")

            left, right = st.columns([1.15, 1])

            with left:
                st.subheader("ROC Curve Comparison")
                fig = go.Figure()
                for idx, r in metrics.iterrows():
                    x, y = _simulate_roc_curve(float(r["roc_auc"]))
                    width = 3 if idx == model_name else 2
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{idx} ({r['roc_auc']:.3f})", line=dict(width=width)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8"), name="Random"))
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.subheader(f"Confusion Matrix - {model_name}")
                p = 1000
                n = 9000
                recall = float(row["recall"])
                precision = max(1e-6, float(row["precision"]))
                tp = int(round(p * recall))
                fn = p - tp
                fp = int(round(tp * (1 / precision - 1)))
                tn = max(0, n - fp)

                cm_df = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["Actual Negative", "Actual Positive"],
                    columns=["Pred Negative", "Pred Positive"],
                )
                st.dataframe(cm_df, use_container_width=True)
                st.caption("Matrix estimated from precision/recall for comparability.")

            fi_path = resolve_artifact_dir() / "feature_importances.csv"
            if fi_path.exists():
                fi = pd.read_csv(fi_path)
                key = {v: k for k, v in DISPLAY_NAMES.items()}.get(model_name, "")
                model_fi = fi[fi["model"] == key].sort_values("importance", ascending=False).head(12)
                if not model_fi.empty:
                    st.subheader("Top Feature Importance")
                    fig_fi = go.Figure(
                        go.Bar(
                            x=model_fi["importance"][::-1],
                            y=model_fi["feature"][::-1],
                            orientation="h",
                            marker=dict(color="#2563eb"),
                        )
                    )
                    fig_fi.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_fi, use_container_width=True)
