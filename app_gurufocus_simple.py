import streamlit as st
import pandas as pd
import requests
import time

# Set page configuration
st.set_page_config(
    page_title="GuruFocus Stock Analysis App",
    page_icon="📊",
    layout="wide"
)

# GuruFocus API Configuration
GURUFOCUS_BASE_URL = "https://api.gurufocus.com/data"

def get_gurufocus_headers(api_key):
    """Get headers for GuruFocus API requests"""
    return {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }

def fetch_stock_profile(symbol, api_key):
    """Fetch stock profile data from GuruFocus API"""
    try:
        url = f"{GURUFOCUS_BASE_URL}/stocks/{symbol}/profile"
        headers = get_gurufocus_headers(api_key)
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.error("Invalid API key or insufficient permissions")
            return None
        elif response.status_code == 404:
            st.error(f"Stock symbol '{symbol}' not found")
            return None
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timeout - please try again")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error fetching profile data: {str(e)}")
        return None

def main():
    # Navigation bar
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("🏠 Main Menu", help="Return to landing page"):
            st.markdown('<meta http-equiv="refresh" content="0; url=/">', unsafe_allow_html=True)
    with col_nav2:
        st.markdown("**🏦 GuruFocus Version**")
    with col_nav3:
        if st.button("📈 Yahoo Finance Version", help="Switch to free technical analysis"):
            st.markdown('<meta http-equiv="refresh" content="0; url=http://localhost:5000">', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.title("📊 GuruFocus Stock Analysis Application")
    st.markdown("Advanced stock analysis using GuruFocus professional data")
    
    # API Key input
    st.sidebar.header("🔑 GuruFocus API Configuration")
    api_key = st.sidebar.text_input(
        "Enter your GuruFocus API Key:",
        type="password",
        help="Get your API key from GuruFocus download page"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your GuruFocus API key in the sidebar to get started")
        st.info("""
        **How to get GuruFocus API access:**
        1. Sign up for a GuruFocus account
        2. Subscribe to a data plan that includes API access
        3. Go to the GuruFocus download page
        4. Copy your API key and paste it in the sidebar
        """)
        return
    
    # Stock input
    st.header("🔍 Stock Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol:",
            placeholder="e.g., AAPL, MSFT, TSLA",
            help="Enter any stock ticker symbol"
        ).upper().strip()
    
    with col2:
        analyze_button = st.button("📊 Analyze Stock", type="primary")
    
    # Process the request when button is clicked
    if analyze_button and symbol:
        with st.spinner(f'Fetching GuruFocus data for {symbol}...'):
            profile_data = fetch_stock_profile(symbol, api_key)
            
            if profile_data:
                st.success(f"Successfully fetched data for {symbol}")
                
                # Display basic information
                if 'basic_information' in profile_data:
                    basic_info = profile_data['basic_information']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        company_name = basic_info.get('company', 'N/A')
                        st.metric("Company Name", company_name)
                    
                    with col2:
                        exchange = basic_info.get('exchange', 'N/A')
                        st.metric("Exchange", exchange)
                    
                    with col3:
                        sector = basic_info.get('sector', 'N/A')
                        st.metric("Sector", sector)
                
                # Display raw data for debugging
                st.subheader("📋 Raw Data (for debugging)")
                st.json(profile_data)
            else:
                st.error("Failed to fetch data. Please check your API key and symbol.")

if __name__ == "__main__":
    main()