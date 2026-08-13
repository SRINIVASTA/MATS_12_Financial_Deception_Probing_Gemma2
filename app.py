import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Deception Auditor", page_icon="📊", layout="wide")

st.title("📊 Gemma 2 Financial Deception Vector Auditor")
st.markdown("### **Research Framework: MATS Cohort 12.0 Application Task**")
st.markdown("""
**Lead Researcher:** Appala Srinivas Tanakala (FinTech Data Scientist / Quantitative Analyst)  
This portal maps internal neural layer activations inside **Gemma-2-2B-it**. By tracking the model's residual stream hidden states natively via PyTorch, we evaluate how the geometric vector space drifts when a standard factual financial disclosure is warped into a deceptive corporate spin statement designed to mislead investors.
""")

st.divider()

try:
    df_diff = pd.read_csv("layer_metrics.csv")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Net Geometric Vector Drift Across Transformer Blocks")
        fig = px.line(
            df_diff, x="Layer", y="Deviation Magnitude",
            labels={"Layer": "Model Neural Layer Block", "Deviation Magnitude": "Vector Shift Delta (L2 Norm)"},
            markers=True
        )
        fig.update_traces(line_color='#FF4B4B', line_width=3, marker_size=8)
        fig.update_layout(hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("📋 Raw Layer Logs")
        st.markdown("Isolating vector drift boundary changes down to 4 decimal points:")
        st.dataframe(df_diff.style.format({"Deviation Magnitude": "{:.4f}"}), use_container_width=True, height=420)
        
except FileNotFoundError:
    st.error("⚠️ Error: 'layer_metrics.csv' not found. Please upload the data file to the root of your GitHub repository.")

st.divider()
st.markdown("*Developed as an AI safety research asset exploring pragmatic model biology and corporate compliance underwriting.*")
