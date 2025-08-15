import streamlit as st
import requests
import feedparser
from datetime import datetime, timedelta
import json
import os
from openai import OpenAI
import urllib.parse

def get_openai_client():
    """
    Initialize OpenAI client with API key validation
    
    Returns:
        OpenAI: Configured OpenAI client
    
    Raises:
        ValueError: If API key is not found
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key)

def create_social_sharing_section(symbol, sentiment_data):
    """
    Create enhanced social sharing section with working links
    
    Args:
        symbol (str): Stock symbol
        sentiment_data (dict): Sentiment analysis results
    """
    st.markdown("---")
    st.markdown("### 📤 Share Your Sentiment Analysis")
    st.markdown("Share your AI-powered sentiment insights with customizable privacy settings")
    
    col_privacy, col_generate = st.columns([1, 1])
    
    with col_privacy:
        privacy_level = st.selectbox(
            "Privacy Level:",
            ["public", "anonymized", "private"],
            format_func=lambda x: {
                "public": "🌐 Public - Full Details",
                "anonymized": "🔒 Anonymized - No Stock Name", 
                "private": "🔐 Private - Limited Info"
            }.get(x, x),
            help="Choose how much information to include when sharing",
            key=f"sentiment_privacy_{symbol}"
        )
    
    with col_generate:
        if st.button("🚀 Generate Shareable Insight", type="primary", key=f"sentiment_share_{symbol}"):
            st.session_state[f'sentiment_sharing_{symbol}'] = True
    
    # Display sharing options if button clicked
    if st.session_state.get(f'sentiment_sharing_{symbol}', False):
        # Create formatted text based on privacy level
        formatted_text = create_sharing_text(symbol, sentiment_data, privacy_level)
        
        st.success("✅ Shareable insight generated!")
        
        # Display preview
        st.markdown("**📋 Sharing Preview:**")
        st.info(formatted_text)
        
        # Create sharing URLs
        whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(formatted_text)}"
        linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url=https://example.com&text={urllib.parse.quote(formatted_text)}"
        email_subject = "Stock Sentiment Analysis Results"
        email_url = f"mailto:?subject={urllib.parse.quote(email_subject)}&body={urllib.parse.quote(formatted_text)}"
        
        st.markdown("**🔗 Share Your Analysis:**")
        
        # Method 1: Direct clickable links
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center;">
                <a href="{whatsapp_url}" target="_blank" style="
                    background-color: #25D366; 
                    color: white; 
                    padding: 10px 20px; 
                    text-decoration: none; 
                    border-radius: 5px;
                    display: inline-block;
                    font-weight: bold;
                ">📱 WhatsApp</a>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center;">
                <a href="{linkedin_url}" target="_blank" style="
                    background-color: #0077B5; 
                    color: white; 
                    padding: 10px 20px; 
                    text-decoration: none; 
                    border-radius: 5px;
                    display: inline-block;
                    font-weight: bold;
                ">💼 LinkedIn</a>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: center;">
                <a href="{email_url}" style="
                    background-color: #D44638; 
                    color: white; 
                    padding: 10px 20px; 
                    text-decoration: none; 
                    border-radius: 5px;
                    display: inline-block;
                    font-weight: bold;
                ">📧 Email</a>
            </div>
            """, unsafe_allow_html=True)
        
        # Method 2: Copy to clipboard functionality
        st.markdown("---")
        st.markdown("**Alternative: Copy Links Manually**")
        
        with st.expander("📋 Copy URLs if buttons don't work"):
            st.text_area("WhatsApp URL:", value=whatsapp_url, height=80, key=f"wa_url_{symbol}")
            st.text_area("LinkedIn URL:", value=linkedin_url, height=80, key=f"li_url_{symbol}")
            st.text_area("Email URL:", value=email_url, height=100, key=f"email_url_{symbol}")
        
        # Method 3: Simple text copy buttons
        st.markdown("**Or copy the text to share manually:**")
        col_copy1, col_copy2, col_copy3 = st.columns(3)
        
        with col_copy1:
            if st.button("📋 Copy for WhatsApp", key=f"copy_wa_{symbol}"):
                st.code(formatted_text, language=None)
                st.success("Text ready to copy!")
        
        with col_copy2:
            if st.button("📋 Copy for LinkedIn", key=f"copy_li_{symbol}"):
                st.code(formatted_text, language=None)  
                st.success("Text ready to copy!")
        
        with col_copy3:
            if st.button("📋 Copy for Email", key=f"copy_email_{symbol}"):
                st.code(formatted_text, language=None)
                st.success("Text ready to copy!")

def create_sharing_text(symbol, sentiment_data, privacy_level):
    """
    Create formatted text for sharing based on privacy level
    
    Args:
        symbol (str): Stock symbol
        sentiment_data (dict): Sentiment analysis results
        privacy_level (str): Privacy level for sharing
    
    Returns:
        str: Formatted text for sharing
    """
    sentiment_score = sentiment_data.get('sentiment_score', 0)
    overall_impact = sentiment_data.get('overall_impact', 'Unknown')
    avg_confidence = sentiment_data.get('avg_confidence', 0)
    total_articles = sentiment_data.get('total_articles', 0)
    
    # Determine sentiment description
    if sentiment_score > 0.3:
        sentiment_desc = "Very Positive"
    elif sentiment_score > 0.1:
        sentiment_desc = "Positive"
    elif sentiment_score > -0.1:
        sentiment_desc = "Neutral"
    elif sentiment_score > -0.3:
        sentiment_desc = "Negative"
    else:
        sentiment_desc = "Very Negative"
    
    if privacy_level == "public":
        return f"""📰 News Sentiment Analysis for {symbol.upper()}:
• Overall Sentiment: {sentiment_desc} ({sentiment_score:+.2f})
• Investment Outlook: {overall_impact}
• Confidence Level: {avg_confidence:.1%}
• Articles Analyzed: {total_articles}

#StockAnalysis #SentimentAnalysis #Investing"""
        
    elif privacy_level == "anonymized":
        return f"""📰 Stock News Sentiment Analysis:
• Overall Sentiment: {sentiment_desc}
• Investment Outlook: {overall_impact}  
• Confidence Level: {avg_confidence:.1%}
• Articles Analyzed: {total_articles}

#StockAnalysis #SentimentAnalysis"""
        
    else:  # private
        return f"""📰 News Sentiment Analysis Complete:
• Analysis shows {sentiment_desc.lower()} sentiment
• {total_articles} articles analyzed with {avg_confidence:.1%} confidence  
• Investment outlook: {overall_impact}"""

# Test function for the social sharing
def test_social_sharing():
    """Test function to demonstrate social sharing"""
    st.title("🧪 Social Sharing Test")
    
    # Mock sentiment data
    test_data = {
        'sentiment_score': 0.25,
        'overall_impact': 'Bullish',
        'avg_confidence': 0.85,
        'total_articles': 5
    }
    
    symbol = st.text_input("Enter stock symbol:", value="AAPL")
    
    if symbol:
        create_social_sharing_section(symbol, test_data)

if __name__ == "__main__":
    test_social_sharing()