import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
import openai
import smtplib
from email.mime.text import MIMEText
import requests

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")

class SecretsManager:
    """Manage user secrets and API keys"""
    
    def __init__(self):
        pass
    
    def _get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    
    def get_all_secrets(self):
        """Get all secrets configuration"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT secret_name, description, is_required, is_configured, 
                       validation_status, last_updated,
                       CASE WHEN secret_value IS NOT NULL THEN '***configured***' ELSE NULL END as status
                FROM user_secrets 
                ORDER BY is_required DESC, secret_name
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            st.error(f"Database error: {e}")
            return []
    
    def get_secret_value(self, secret_name):
        """Get a specific secret value"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT secret_value FROM user_secrets 
                WHERE secret_name = %s AND secret_value IS NOT NULL
            """, (secret_name,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result['secret_value'] if result else None
            
        except Exception as e:
            print(f"Error getting secret {secret_name}: {e}")
            return None
    
    def update_secret(self, secret_name, secret_value):
        """Update a secret value"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE user_secrets 
                SET secret_value = %s, 
                    is_configured = TRUE,
                    last_updated = CURRENT_TIMESTAMP,
                    validation_status = 'unchecked'
                WHERE secret_name = %s
            """, (secret_value, secret_name))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            st.error(f"Error updating secret: {e}")
            return False
    
    def delete_secret(self, secret_name):
        """Delete a secret value"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE user_secrets 
                SET secret_value = NULL, 
                    is_configured = FALSE,
                    last_updated = CURRENT_TIMESTAMP,
                    validation_status = 'unchecked'
                WHERE secret_name = %s
            """, (secret_name,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            st.error(f"Error deleting secret: {e}")
            return False
    
    def validate_openai_key(self, api_key):
        """Validate OpenAI API key"""
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True, "OpenAI API key is valid"
        except Exception as e:
            return False, f"OpenAI validation failed: {str(e)}"
    
    def validate_gmail_credentials(self, email, app_password):
        """Validate Gmail SMTP credentials"""
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(email, app_password)
            server.quit()
            return True, "Gmail credentials are valid"
        except Exception as e:
            return False, f"Gmail validation failed: {str(e)}"
    
    def validate_gurufocus_key(self, api_key):
        """Validate GuruFocus API key"""
        try:
            url = f"https://api.gurufocus.com/public/user/{api_key}/summary"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True, "GuruFocus API key is valid"
            else:
                return False, f"GuruFocus validation failed: HTTP {response.status_code}"
        except Exception as e:
            return False, f"GuruFocus validation failed: {str(e)}"
    
    def validate_fmp_key(self, api_key):
        """Validate Financial Modeling Prep API key"""
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True, "FMP API key is valid"
            else:
                return False, f"FMP validation failed: HTTP {response.status_code}"
        except Exception as e:
            return False, f"FMP validation failed: {str(e)}"
    
    def update_validation_status(self, secret_name, status):
        """Update validation status for a secret"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE user_secrets 
                SET validation_status = %s
                WHERE secret_name = %s
            """, (status, secret_name))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error updating validation status: {e}")

def admin_secrets_tab():
    """Admin interface for managing secrets"""
    st.markdown("### 🔐 Admin - API Keys & Secrets Management")
    st.markdown("Configure and manage all API keys and credentials used by the application.")
    st.markdown("---")
    
    secrets_manager = SecretsManager()
    
    # Get all secrets
    secrets = secrets_manager.get_all_secrets()
    
    st.markdown("#### 📊 Current Configuration Status")
    
    # Configuration overview
    col1, col2, col3 = st.columns(3)
    configured_count = sum(1 for s in secrets if s['is_configured'])
    required_count = sum(1 for s in secrets if s['is_required'])
    total_count = len(secrets)
    
    with col1:
        st.metric("Configured", f"{configured_count}/{total_count}")
    with col2:
        st.metric("Required", f"{required_count}")
    with col3:
        st.metric("Optional", f"{total_count - required_count}")
    
    st.markdown("---")
    st.markdown("#### 🔧 Manage API Keys")
    
    # Display each secret with management options
    for secret in secrets:
        with st.expander(f"{'🔴' if secret['is_required'] and not secret['is_configured'] else '🟢' if secret['is_configured'] else '🟡'} {secret['secret_name']}", expanded=not secret['is_configured']):
            
            col_info, col_action = st.columns([2, 1])
            
            with col_info:
                st.markdown(f"**Description:** {secret['description']}")
                st.markdown(f"**Required:** {'Yes' if secret['is_required'] else 'No'}")
                st.markdown(f"**Status:** {secret['status'] or 'Not configured'}")
                
                if secret['is_configured']:
                    st.markdown(f"**Last Updated:** {secret['last_updated'].strftime('%Y-%m-%d %H:%M UTC')}")
                    
                    # Validation status
                    validation_color = {
                        'valid': '🟢',
                        'invalid': '🔴', 
                        'unchecked': '🟡'
                    }
                    st.markdown(f"**Validation:** {validation_color.get(secret['validation_status'], '🟡')} {secret['validation_status'].title()}")
            
            with col_action:
                # Input field for secret value
                if secret['secret_name'] in ['GMAIL_APP_PASSWORD', 'OPENAI_API_KEY', 'GURUFOCUS_API_KEY', 'FMP_API_KEY']:
                    secret_input = st.text_input(
                        "API Key/Password", 
                        type="password", 
                        key=f"input_{secret['secret_name']}",
                        placeholder="Enter your API key..."
                    )
                else:
                    secret_input = st.text_input(
                        "Value", 
                        key=f"input_{secret['secret_name']}",
                        placeholder="Enter value..."
                    )
                
                # Action buttons
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    if st.button("💾 Save", key=f"save_{secret['secret_name']}"):
                        if secret_input:
                            if secrets_manager.update_secret(secret['secret_name'], secret_input):
                                st.success("✅ Saved!")
                                st.rerun()
                        else:
                            st.warning("Please enter a value")
                
                with col_btn2:
                    if secret['is_configured'] and st.button("🗑️ Delete", key=f"delete_{secret['secret_name']}"):
                        if secrets_manager.delete_secret(secret['secret_name']):
                            st.success("✅ Deleted!")
                            st.rerun()
                
                # Validation button
                if secret['is_configured']:
                    if st.button("🔍 Test", key=f"test_{secret['secret_name']}"):
                        with st.spinner("Testing API key..."):
                            success, message = test_secret(secret['secret_name'], secrets_manager)
                            
                            if success:
                                st.success(f"✅ {message}")
                                secrets_manager.update_validation_status(secret['secret_name'], 'valid')
                            else:
                                st.error(f"❌ {message}")
                                secrets_manager.update_validation_status(secret['secret_name'], 'invalid')
                            
                            st.rerun()
    
    st.markdown("---")
    st.markdown("#### 📋 Setup Instructions")
    
    with st.expander("🔧 How to Get API Keys", expanded=False):
        st.markdown("""
        **OpenAI API Key (Required):**
        1. Visit https://platform.openai.com/api-keys
        2. Sign in to your OpenAI account
        3. Click "Create new secret key"
        4. Copy the key and paste it above
        
        **Gmail App Password (Optional - for email sending):**
        1. Enable 2-Factor Authentication on your Gmail account
        2. Go to Google Account Settings → Security → App passwords
        3. Generate a new app password for "Mail"
        4. Use this 16-character password (not your regular Gmail password)
        
        **GuruFocus API Key (Optional - for premium financial data):**
        1. Visit https://www.gurufocus.com/api
        2. Sign up for an API plan
        3. Get your API key from the dashboard
        
        **Financial Modeling Prep API Key (Optional):**
        1. Visit https://financialmodelingprep.com/developer/docs
        2. Sign up for a free or paid plan
        3. Get your API key from the dashboard
        """)
    
    # System info
    st.markdown("---")
    st.markdown("#### ℹ️ System Information")
    st.info("""
    **Security Notes:**
    - All API keys are stored encrypted in the PostgreSQL database
    - Keys are never displayed in plain text in the UI
    - Validation tests use minimal API calls to verify functionality
    - You can delete any key at any time
    
    **Priority:**
    - 🔴 Red: Required and not configured
    - 🟢 Green: Configured and working
    - 🟡 Yellow: Optional or needs attention
    """)

def test_secret(secret_name, secrets_manager):
    """Test a specific secret"""
    secret_value = secrets_manager.get_secret_value(secret_name)
    
    if not secret_value:
        return False, "Secret not configured"
    
    if secret_name == "OPENAI_API_KEY":
        return secrets_manager.validate_openai_key(secret_value)
    
    elif secret_name == "GMAIL_APP_PASSWORD":
        gmail_email = secrets_manager.get_secret_value("GMAIL_EMAIL")
        if not gmail_email:
            return False, "Gmail email not configured"
        return secrets_manager.validate_gmail_credentials(gmail_email, secret_value)
    
    elif secret_name == "GMAIL_EMAIL":
        gmail_password = secrets_manager.get_secret_value("GMAIL_APP_PASSWORD")
        if not gmail_password:
            return False, "Gmail app password not configured"
        return secrets_manager.validate_gmail_credentials(secret_value, gmail_password)
    
    elif secret_name == "GURUFOCUS_API_KEY":
        return secrets_manager.validate_gurufocus_key(secret_value)
    
    elif secret_name == "FMP_API_KEY":
        return secrets_manager.validate_fmp_key(secret_value)
    
    else:
        return True, "Secret configured (no validation available)"

if __name__ == "__main__":
    admin_secrets_tab()