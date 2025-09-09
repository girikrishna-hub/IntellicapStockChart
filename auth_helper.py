import streamlit as st
import os

def check_authentication():
    """Simple authentication system for testing phase"""
    
    # Get credentials from environment
    TEST_USERNAME = os.environ.get("TEST_USERNAME", "testuser")
    TEST_PASSWORD = os.environ.get("TEST_PASSWORD", "testpass123")
    
    # Check if already authenticated
    if st.session_state.get("authenticated", False):
        return True
    
    # Show login form
    st.markdown("### 🔐 Access Required")
    st.markdown("Please enter the testing credentials to access the financial analysis platform.")
    
    with st.form("login_form"):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            
            submitted = st.form_submit_button("🚀 Access Platform")
        
        with col2:
            st.info("""
            **For Testing Access:**
            - Contact the platform owner for credentials
            - This is a controlled testing environment
            - Your API keys will be stored securely
            """)
    
    if submitted:
        if username == TEST_USERNAME and password == TEST_PASSWORD:
            st.session_state["authenticated"] = True
            st.success("✅ Access granted! Redirecting...")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please check your username and password.")
    
    return False

def logout():
    """Logout function"""
    if st.session_state.get("authenticated", False):
        st.session_state["authenticated"] = False
        st.rerun()

def show_logout_option():
    """Show logout option in sidebar"""
    if st.session_state.get("authenticated", False):
        with st.sidebar:
            st.markdown("---")
            if st.button("🚪 Logout", help="Logout from the platform"):
                logout()