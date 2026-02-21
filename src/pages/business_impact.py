import numpy as np
import plotly.graph_objects as go
import streamlit as st


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


def app():
    st.title("Business Impact")
    st.caption("Simulate and quantify the financial impact of AI-powered risk detection")

    left, right = st.columns([1, 1.6])
    with left:
        st.subheader("Simulation Parameters")
        total_loans = st.number_input("Total Loans in Portfolio", min_value=100, value=10000, step=100)
        avg_loan_size = st.number_input("Average Loan Size ($)", min_value=1000, value=50000, step=1000)
        default_rate = st.slider("Expected Default Rate", min_value=1, max_value=40, value=14) / 100.0
        detection_rate = st.slider("AI Detection Rate", min_value=20, max_value=99, value=89) / 100.0
        lgd = st.slider("Loss Given Default", min_value=20, max_value=90, value=60) / 100.0
        model_cost = st.number_input("Annual AI Program Cost ($)", min_value=10000, value=250000, step=10000)

    total_defaults = total_loans * default_rate
    baseline_loss = total_defaults * avg_loan_size * lgd
    caught = total_defaults * detection_rate
    remaining_defaults = total_defaults - caught
    remaining_loss = remaining_defaults * avg_loan_size * lgd
    savings = baseline_loss - remaining_loss
    roi = savings / max(1.0, model_cost)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Estimated Annual Savings", f"${savings / 1e6:.1f}M", "From early default detection")
    with c2:
        _metric_card("Defaults Caught", f"{caught:,.0f}", f"{detection_rate * 100:.0f}% detection rate")
    with c3:
        _metric_card("Remaining Exposure", f"${remaining_loss / 1e6:.1f}M", f"{remaining_defaults:,.0f} undetected defaults")
    with c4:
        _metric_card("ROI Multiplier", f"{roi:.1f}x", "Return on model investment")

    with right:
        st.subheader("Loss Comparison: With vs Without AI")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Without AI", x=["Annual Loss"], y=[baseline_loss], marker=dict(color="#dc2626")))
        fig.add_trace(go.Bar(name="With AI", x=["Annual Loss"], y=[remaining_loss], marker=dict(color="#16a34a")))
        fig.update_layout(
            barmode="group",
            yaxis_tickprefix="$",
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("12-Month Impact Trajectory")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_without = np.array([baseline_loss / 12.0] * 12)
    monthly_with = np.array([remaining_loss / 12.0] * 12)
    monthly_savings = np.cumsum(monthly_without - monthly_with)

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=months, y=monthly_without, mode="lines", name="Without AI", line=dict(color="#dc2626")))
    fig_line.add_trace(go.Scatter(x=months, y=monthly_with, mode="lines", name="With AI", line=dict(color="#16a34a")))
    fig_line.add_trace(
        go.Scatter(
            x=months,
            y=monthly_savings,
            mode="lines",
            name="Cumulative Savings",
            yaxis="y2",
            line=dict(color="#1d4ed8", dash="dash"),
        )
    )
    fig_line.update_layout(
        height=360,
        yaxis=dict(title="Monthly Loss ($)", tickprefix="$"),
        yaxis2=dict(title="Cumulative Savings ($)", overlaying="y", side="right", tickprefix="$"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown(
        f"""
        **Executive Summary**  
        With a portfolio of **{total_loans:,}** loans and expected default rate of **{default_rate * 100:.1f}%**, deploying the AI model at
        **{detection_rate * 100:.0f}%** detection can reduce annual expected losses from **${baseline_loss/1e6:.1f}M** to
        **${remaining_loss/1e6:.1f}M**, generating **${savings/1e6:.1f}M** in annual savings and an estimated **{roi:.1f}x ROI**.
        """
    )
