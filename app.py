import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration Setup
st.set_page_config(
    page_title="AI Deception Auditor", 
    page_icon="📊", 
    layout="wide"
)

# 2. Main Professional Headers
st.title("📊 Gemma 2 Financial Deception Vector Auditor")
st.markdown("### **Research Framework: MATS Cohort 12.0 Application Task**")
st.markdown("""
**Lead Researcher:** Appala Srinivas Tanakala (FinTech Data Scientist / Quantitative Analyst)  
This portal maps internal neural layer activations inside **Gemma-2-2B-it**. By tracking the model's residual stream hidden states natively via PyTorch, we evaluate how the geometric vector space drifts when a standard factual financial disclosure is warped into a deceptive corporate spin statement designed to mislead investors.
""")

st.divider()

# 3. Synchronized Data Loading & Rendering Pipe
try:
    # Read both metric tracking sheets exported from Colab
    df_compare = pd.read_csv("layer_metrics_compare.csv")
    df_diff = pd.read_csv("layer_metrics.csv") # <-- FIXED: Uses your exact filename
    
    col1, col2 = st.columns() 
    
    with col1:
        st.subheader("📈 Activation Energy Divergence Tracks")
        st.markdown("Comparing absolute hidden state vectors simultaneously:")
        
        fig_compare = px.line(
            df_compare, 
            x="Layer", 
            y="Activation Norm", 
            color="Prompt Type", 
            color_discrete_map={
                "Honest Audit Run": "#1F77B4",         
                "Deceptive Corporate Spin": "#FF4B4B"  
            },
            labels={"Layer": "Model Neural Layer Block", "Activation Norm": "Vector Magnitude (L2 Norm)"},
            markers=True
        )
        fig_compare.update_layout(
            hovermode="x unified", 
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
    with col2:
        st.subheader("📋 Net Geometric Vector Drift Logs")
        st.markdown("Absolute subtraction delta trace (L2 Norm error value per layer):")
        
        st.dataframe(
            df_diff.style.format({"Deviation Magnitude": "{:.4f}"}), 
            use_container_width=True, 
            height=420
        )
        
except FileNotFoundError as e:
    st.error("""
    ⚠️ **Configuration Deployment Alert:** Missing required metrics files in root directory! 
    Ensure both `layer_metrics_compare.csv` and `layer_metrics.csv` are pushed to your GitHub repository.
    """)

st.divider()
st.markdown("*Developed as an AI safety research asset exploring pragmatic model biology and corporate compliance underwriting.*")
