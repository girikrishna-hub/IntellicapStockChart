import streamlit as st
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 3rem;
    }
    
    .version-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .free-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .pro-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .feature-list {
        text-align: left;
        margin: 1rem 0;
    }
    
    .comparison-table {
        margin: 2rem 0;
    }
    
    .launch-button {
        background: rgba(255,255,255,0.2);
        border: 2px solid white;
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        text-decoration: none;
        display: inline-block;
        margin: 1rem;
        transition: all 0.3s ease;
    }
    
    .launch-button:hover {
        background: white;
        color: #333;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .stat-box {
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin: 0.5rem;
        flex: 1;
        min-width: 150px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Stock Analysis Platform</h1>
        <h3>Professional Stock Research & Technical Analysis</h3>
        <p>Choose your preferred data source and analysis approach</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Technical Indicators", "8+", "Moving Averages, RSI, MACD")
    with col2:
        st.metric("Market Coverage", "Global", "US, Indian, International")
    with col3:
        st.metric("Analysis Modes", "2", "Single Stock & Bulk")
    with col4:
        st.metric("Export Formats", "Excel", "Professional Reports")
    
    st.markdown("---")
    
    # Version selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="version-card free-card">
            <h2>🆓 Yahoo Finance Version</h2>
            <h4>Free Technical Analysis</h4>
            
            <div class="feature-list">
                <h5>✅ Features:</h5>
                <ul>
                    <li>Real-time stock prices & volume</li>
                    <li>8 technical indicators (MA, MACD, RSI, CMF)</li>
                    <li>Support/resistance levels</li>
                    <li>52-week position analysis</li>
                    <li>Earnings data & dividend tracking</li>
                    <li>Auto-refresh every 10 minutes</li>
                    <li>US & Indian market support</li>
                    <li>Bulk analysis with Excel export</li>
                    <li>Saved stock lists</li>
                </ul>
                
                <h5>🎯 Perfect For:</h5>
                <ul>
                    <li>Individual investors & traders</li>
                    <li>Technical analysis enthusiasts</li>
                    <li>Real-time market monitoring</li>
                    <li>Learning stock analysis</li>
                </ul>
                
                <h5>💰 Cost: FREE</h5>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Yahoo Finance Version", key="yahoo", use_container_width=True):
            st.success("Opening Yahoo Finance version in new tab...")
            st.info("📍 Access directly at: http://localhost:5000")
            st.markdown("""
            <meta http-equiv="refresh" content="0; url=http://localhost:5000">
            <script>
                setTimeout(function(){
                    window.open('http://localhost:5000', '_blank');
                }, 100);
            </script>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="version-card pro-card">
            <h2>🏦 GuruFocus Version</h2>
            <h4>Professional Institutional Data</h4>
            
            <div class="feature-list">
                <h5>✅ Features:</h5>
                <ul>
                    <li>Institutional-grade financial data</li>
                    <li>30+ years of historical fundamentals</li>
                    <li>Comprehensive company profiles</li>
                    <li>Detailed dividend history</li>
                    <li>Valuation metrics & ratios</li>
                    <li>Revenue breakdowns by segment</li>
                    <li>Professional Excel reporting</li>
                    <li>Bulk analysis capabilities</li>
                    <li>API-driven data accuracy</li>
                </ul>
                
                <h5>🎯 Perfect For:</h5>
                <ul>
                    <li>Professional portfolio managers</li>
                    <li>Value investing research</li>
                    <li>Institutional analysis</li>
                    <li>Fundamental analysis deep-dives</li>
                </ul>
                
                <h5>💰 Cost: GuruFocus Subscription Required</h5>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch GuruFocus Version", key="guru", use_container_width=True):
            st.success("Opening GuruFocus version in new tab...")
            st.info("📍 Access directly at: http://localhost:5001")
            st.warning("Remember to enter your GuruFocus API key in the sidebar")
            st.markdown("""
            <script>
                setTimeout(function(){
                    window.open('http://localhost:5001', '_blank');
                }, 100);
            </script>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison table
    st.subheader("📊 Detailed Comparison")
    
    comparison_data = {
        "Feature": [
            "Cost",
            "Data Source", 
            "Real-time Prices",
            "Technical Indicators",
            "Historical Data",
            "Fundamentals",
            "Dividend History",
            "Market Coverage",
            "API Reliability",
            "Best For"
        ],
        "Yahoo Finance": [
            "Free",
            "Yahoo Finance API",
            "✅ Yes",
            "✅ 8 indicators",
            "✅ 5+ years",
            "⚠️ Basic",
            "✅ Good",
            "US, India",
            "Good",
            "Individual traders"
        ],
        "GuruFocus": [
            "Paid subscription",
            "GuruFocus API",
            "⚠️ Limited",
            "⚠️ Basic",
            "✅ 30+ years",
            "✅ Comprehensive",
            "✅ Complete",
            "Global",
            "Professional",
            "Institutional investors"
        ]
    }
    
    try:
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except ImportError:
        # Fallback table display without pandas
        st.markdown("| Feature | Yahoo Finance | GuruFocus |")
        st.markdown("|---------|---------------|-----------|")
        for i in range(len(comparison_data["Feature"])):
            feature = comparison_data["Feature"][i]
            yahoo = comparison_data["Yahoo Finance"][i]
            guru = comparison_data["GuruFocus"][i]
            st.markdown(f"| {feature} | {yahoo} | {guru} |")
    
    st.markdown("---")
    
    # Quick start guides
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Yahoo Finance Quick Start")
        st.markdown("""
        1. **Click "Launch Yahoo Finance Version"**
        2. **Enter stock symbol** (e.g., AAPL, MSFT, RELIANCE.NS)
        3. **View technical charts** with indicators
        4. **Enable auto-refresh** for live tracking
        5. **Try bulk analysis** for portfolio review
        """)
    
    with col2:
        st.subheader("🔑 GuruFocus Setup Guide")
        st.markdown("""
        1. **Sign up** for GuruFocus account
        2. **Subscribe** to data plan with API access
        3. **Get API key** from download page
        4. **Click "Launch GuruFocus Version"**
        5. **Enter API key** in sidebar
        6. **Start analyzing** with professional data
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <h4>📈 Happy Investing!</h4>
        <p>Both versions provide powerful stock analysis capabilities tailored to different investor needs.</p>
        <p><strong>Choose the version that best fits your investment strategy and budget.</strong></p>
        <br>
        <small>Last updated: {}</small>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()