import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main header
st.title("📈 Stock Technical Analysis Hub")
st.markdown("**Choose your preferred analysis platform below**")

# Create two columns for the main options
col1, col2 = st.columns(2)

with col1:
    st.header("🆓 Yahoo Finance Version")
    st.markdown("""
    **Features:**
    - Free real-time data
    - Technical indicators (MA, MACD, RSI)
    - Support/resistance levels
    - Bulk analysis with Excel export
    - US and Indian markets
    """)
    
    if st.button("Open Yahoo Finance Version", key="yahoo", use_container_width=True):
        st.success("Yahoo Finance version selected!")
        st.info("**Access Yahoo Finance Version:**")
        st.code("Replace :3001 with :80 in your browser URL")
        st.markdown("Example: If you're at `yourproject.replit.dev:3001`, go to `yourproject.replit.dev:80`")

with col2:
    st.header("💼 GuruFocus Version")
    st.markdown("""
    **Features:**
    - Professional-grade data
    - Company fundamentals
    - Institutional analytics
    - Dividend analysis
    - Valuation metrics
    """)
    
    if st.button("Open GuruFocus Version", key="guru", use_container_width=True):
        st.success("GuruFocus version selected!")
        st.info("**Access GuruFocus Version:**")
        st.code("Replace :3001 with :3000 in your browser URL")
        st.markdown("Example: If you're at `yourproject.replit.dev:3001`, go to `yourproject.replit.dev:3000`")
        st.warning("Note: Requires GuruFocus API key")

# Feature comparison
st.markdown("---")
st.subheader("📊 Feature Comparison")

comparison_data = {
    "Feature": [
        "Data Source",
        "Cost",
        "Technical Analysis",
        "Fundamental Data",
        "Real-time Updates",
        "Bulk Analysis",
        "Excel Export",
        "Market Coverage"
    ],
    "Yahoo Finance": [
        "Yahoo Finance API",
        "Free",
        "✅ Advanced",
        "✅ Basic",
        "✅ Yes",
        "✅ Yes",
        "✅ Yes",
        "✅ Global"
    ],
    "GuruFocus": [
        "GuruFocus API",
        "Requires subscription",
        "✅ Basic",
        "✅ Professional",
        "✅ Yes",
        "✅ Yes",
        "✅ Yes",
        "✅ Global"
    ]
}

st.table(comparison_data)

# Quick start guide
st.markdown("---")
st.subheader("🚀 Quick Start Guide")

st.markdown("""
**For Yahoo Finance Version:**
1. Click "Open Yahoo Finance Version" above
2. Enter any stock symbol (e.g., AAPL, TSLA)
3. Analyze charts and technical indicators
4. Use bulk analysis for multiple stocks

**For GuruFocus Version:**
1. Get your API key from GuruFocus.com
2. Click "Open GuruFocus Version" above  
3. Enter your API key in the sidebar
4. Analyze professional fundamental data
""")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Choose the version that best fits your analysis needs")