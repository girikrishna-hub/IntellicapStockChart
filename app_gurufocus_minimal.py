import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="GuruFocus Stock Analysis",
    page_icon="📊",
    layout="wide"
)

# Simple test version
st.title("📊 GuruFocus Stock Analysis Application")
st.write("This is a minimal test version of the GuruFocus app.")

# Navigation
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠 Main Menu"):
        st.write("Would navigate to main menu")
with col2:
    st.write("**GuruFocus Version**")
with col3:
    if st.button("📈 Yahoo Finance"):
        st.write("Would navigate to Yahoo Finance")

# Simple API key input
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter GuruFocus API Key:", type="password")
    
    if api_key:
        st.success("API key entered!")
    else:
        st.warning("Enter your API key to continue")

# Simple stock input
st.header("Stock Analysis")
symbol = st.text_input("Enter Stock Symbol:", placeholder="e.g., AAPL")

if symbol:
    st.write(f"You entered: {symbol}")
    if st.button("Test Analysis"):
        st.write("This would perform analysis...")
        st.success("Test completed!")

st.write("---")
st.write("If you can see this page, the GuruFocus app is working properly.")