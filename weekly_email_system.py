import streamlit as st
import openai
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import json

# the newest OpenAI model is "gpt-4o" which was released August 7, 2025.
# do not change this unless explicitly requested by the user

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")

# Database connection helper
import psycopg2
from psycopg2.extras import RealDictCursor

# Secrets manager
class SecretsHelper:
    """Helper to get user-configured secrets from database"""
    
    @staticmethod
    def get_secret(secret_name):
        """Get a secret value from user configuration"""
        try:
            conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
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
    
    @staticmethod
    def get_openai_client():
        """Get configured OpenAI client"""
        api_key = SecretsHelper.get_secret('OPENAI_API_KEY')
        if api_key:
            return openai.OpenAI(api_key=api_key)
        return None
    
    @staticmethod
    def get_gmail_credentials():
        """Get Gmail credentials"""
        email = SecretsHelper.get_secret('GMAIL_EMAIL')
        password = SecretsHelper.get_secret('GMAIL_APP_PASSWORD')
        return email, password

class WeeklyEmailGenerator:
    """Generate comprehensive weekly market analysis emails"""
    
    def __init__(self, market="US"):
        self.market = market
        self.week_start = self._get_week_start()
        self.week_end = self.week_start + timedelta(days=6)
        self.week_year = f"{self.week_start.year}-W{self.week_start.isocalendar()[1]:02d}"
    
    def _get_week_start(self):
        """Get the Monday of the current week"""
        today = datetime.now().date()
        days_since_monday = today.weekday()
        week_start = today - timedelta(days=days_since_monday)
        return datetime.combine(week_start, datetime.min.time())
    
    def _get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    
    def check_existing_email(self):
        """Check if email already exists for this week and market"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, email_html, email_subject, generated_at 
                FROM weekly_emails 
                WHERE week_year = %s AND market_type = %s
            """, (self.week_year, self.market))
            
            result = cursor.fetchone()
            conn.close()
            
            return dict(result) if result else None
            
        except Exception as e:
            print(f"Database error checking existing email: {e}")
            return None
    
    def save_email_to_db(self, email_html, email_subject):
        """Save generated email to database"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Mark all previous emails as not current
            cursor.execute("""
                UPDATE weekly_emails 
                SET is_current_week = FALSE 
                WHERE market_type = %s
            """, (self.market,))
            
            # Insert new email
            cursor.execute("""
                INSERT INTO weekly_emails 
                (week_start, week_end, market_type, email_html, email_subject, week_year, is_current_week)
                VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                ON CONFLICT (week_year, market_type) 
                DO UPDATE SET 
                    email_html = EXCLUDED.email_html,
                    email_subject = EXCLUDED.email_subject,
                    generated_at = CURRENT_TIMESTAMP,
                    is_current_week = TRUE
                RETURNING id
            """, (
                self.week_start.date(), 
                self.week_end.date(), 
                self.market, 
                email_html, 
                email_subject, 
                self.week_year
            ))
            
            email_id = cursor.fetchone()['id']
            conn.commit()
            conn.close()
            
            return email_id
            
        except Exception as e:
            print(f"Database error saving email: {e}")
            return None
    
    def get_recent_emails(self, limit=5):
        """Get recent emails from database"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, week_year, market_type, email_subject, generated_at
                FROM weekly_emails 
                WHERE market_type = %s
                ORDER BY generated_at DESC 
                LIMIT %s
            """, (self.market, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"Database error getting recent emails: {e}")
            return []
        
    def generate_week_ahead_section(self):
        """Generate Week Ahead section with earnings and economic events"""
        try:
            # Get upcoming earnings (next 7 days)
            earnings_data = self._get_upcoming_earnings()
            
            # Get economic events (simplified - can be enhanced)
            economic_events = self._get_economic_events()
            
            week_ahead_html = f"""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h2 style="color: #2c3e50; margin-bottom: 15px;">📅 Week Ahead</h2>
                
                <h3 style="color: #34495e;">🏢 Major Earnings This Week</h3>
                <ul style="color: #2c3e50;">
                    {earnings_data}
                </ul>
                
                <h3 style="color: #34495e;">📊 Key Economic Events</h3>
                <ul style="color: #2c3e50;">
                    {economic_events}
                </ul>
            </div>
            """
            
            return week_ahead_html
            
        except Exception as e:
            return f"<p>Error generating Week Ahead section: {str(e)}</p>"
    
    def generate_ai_market_insights(self):
        """Generate AI-powered market insights using OpenAI"""
        try:
            # Get OpenAI client from user configuration
            openai_client = SecretsHelper.get_openai_client()
            if not openai_client:
                return self._get_fallback_insights()
            
            # Get recent market data for context
            market_context = self._get_market_context()
            
            if self.market == "Indian":
                prompt = f"""
                As a financial market analyst specializing in Indian markets, provide a comprehensive weekly analysis based on this data:
                
                {market_context}
                
                Please provide insights specifically for Indian markets on:
                1. NSE/BSE sentiment and Nifty/Sensex trends
                2. Key Indian sectors to watch (IT, Banking, Pharma, Auto, etc.)
                3. Risk factors including global cues, FII flows, and domestic policy
                4. Technical analysis of major Indian indices
                5. Investment themes relevant to Indian investors
                6. Impact of monsoon, inflation, and RBI policy on markets
                
                Keep the analysis professional but accessible, around 200-300 words.
                """
            else:
                prompt = f"""
                As a financial market analyst, provide a comprehensive weekly market analysis based on this data:
                
                {market_context}
                
                Please provide insights on:
                1. Overall market sentiment and trends
                2. Key sectors to watch
                3. Risk factors for the coming week
                4. Technical analysis observations
                5. Investment themes and opportunities
                
                Keep the analysis professional but accessible, around 200-300 words.
                """
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst providing weekly market insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            ai_insights = response.choices[0].message.content
            
            ai_insights_html = f"""
            <div style="background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h2 style="color: #2c3e50; margin-bottom: 15px;">🧠 AI Market Insights</h2>
                <div style="color: #2c3e50; line-height: 1.6;">
                    {ai_insights.replace(chr(10), '<br>')}
                </div>
                <p style="font-size: 12px; color: #7f8c8d; margin-top: 15px;">
                    <em>Analysis generated by OpenAI based on current market data</em>
                </p>
            </div>
            """
            
            return ai_insights_html
            
        except Exception as e:
            # Fallback content when AI fails
            fallback_insights = """
            <strong>Market Analysis (This Week):</strong><br>
            • Markets continue to show resilience amid economic uncertainty<br>
            • Technology sector remains a key focus for investors<br>
            • Federal Reserve policy decisions continue to influence market sentiment<br>
            • Earnings season provides important guidance for sector allocation<br>
            • Volatility presents both opportunities and risks for active traders<br><br>
            
            <strong>Key Themes:</strong><br>
            • Monitor interest rate trends and Fed communications<br>
            • Watch for earnings surprises in major technology stocks<br>
            • Consider defensive positioning amid global uncertainties<br>
            • Focus on companies with strong fundamentals and cash flow
            """
            
            return f"""
            <div style="background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h2 style="color: #2c3e50; margin-bottom: 15px;">🧠 AI Market Insights</h2>
                <div style="color: #2c3e50; line-height: 1.6;">
                    {fallback_insights}
                </div>
                <p style="font-size: 12px; color: #7f8c8d; margin-top: 15px;">
                    <em>Fallback analysis - AI service temporarily unavailable (Error: {str(e)})</em>
                </p>
            </div>
            """
    
    def generate_action_items(self):
        """Generate actionable investment recommendations"""
        try:
            # Get stock screening data for recommendations
            action_items_data = self._get_action_items_data()
            
            # Use AI to generate personalized action items
            prompt = f"""
            Based on current market conditions and this stock data:
            {action_items_data}
            
            Provide 4-5 specific, actionable investment recommendations for this week:
            1. Stocks to research or consider
            2. Sectors showing strength/weakness
            3. Technical levels to watch
            4. Risk management suggestions
            5. Timing considerations
            
            Keep each item concise and specific with clear rationale.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial advisor providing specific, actionable investment recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            action_items = response.choices[0].message.content
            
            action_items_html = f"""
            <div style="background: #fff3cd; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h2 style="color: #2c3e50; margin-bottom: 15px;">🎯 Action Items</h2>
                <div style="color: #2c3e50; line-height: 1.6;">
                    {action_items.replace(chr(10), '<br>')}
                </div>
                <p style="font-size: 12px; color: #7f8c8d; margin-top: 15px;">
                    <em>Recommendations are for informational purposes only and not financial advice</em>
                </p>
            </div>
            """
            
            return action_items_html
            
        except Exception as e:
            # Fallback action items when AI fails
            fallback_actions = """
            <strong>This Week's Action Items:</strong><br>
            1. <strong>Review Earnings Calendar:</strong> Check upcoming earnings for your portfolio holdings<br>
            2. <strong>Monitor Fed Communications:</strong> Watch for policy signals affecting interest rates<br>
            3. <strong>Sector Rotation Check:</strong> Evaluate if any sector allocations need adjustment<br>
            4. <strong>Risk Assessment:</strong> Review position sizes and consider taking profits on winners<br>
            5. <strong>Research Opportunities:</strong> Look for oversold quality stocks in current market conditions
            """
            
            return f"""
            <div style="background: #fff3cd; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h2 style="color: #2c3e50; margin-bottom: 15px;">🎯 Action Items</h2>
                <div style="color: #2c3e50; line-height: 1.6;">
                    {fallback_actions}
                </div>
                <p style="font-size: 12px; color: #7f8c8d; margin-top: 15px;">
                    <em>Fallback recommendations - AI service temporarily unavailable (Error: {str(e)})</em>
                </p>
            </div>
            """
    
    def _get_upcoming_earnings(self):
        """Get upcoming earnings for major stocks"""
        try:
            if self.market == "Indian":
                # Major Indian stocks
                major_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 'ITC.NS']
            else:
                # US stocks
                major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
            
            earnings_list = []
            
            for symbol in major_stocks[:5]:  # Limit to prevent API overload
                try:
                    ticker = yf.Ticker(symbol)
                    calendar = ticker.calendar
                    if calendar is not None and not calendar.empty:
                        # Check if earnings are in the next 7 days
                        earnings_date = calendar.index[0] if len(calendar.index) > 0 else None
                        if earnings_date:
                            date_str = earnings_date.strftime("%A, %B %d")
                            display_symbol = symbol.replace('.NS', '') if self.market == "Indian" else symbol
                            earnings_list.append(f"<li><strong>{display_symbol}</strong> - {date_str}</li>")
                except:
                    continue
            
            if not earnings_list:
                if self.market == "Indian":
                    earnings_list = [
                        "<li>No major Indian earnings scheduled this week</li>",
                        "<li>Check NSE/BSE calendars for specific dates</li>"
                    ]
                else:
                    earnings_list = [
                        "<li>No major earnings scheduled this week</li>",
                        "<li>Check individual stock calendars for specific dates</li>"
                    ]
            
            return "".join(earnings_list)
            
        except Exception as e:
            return f"<li>Error loading earnings data: {str(e)}</li>"
    
    def _get_economic_events(self):
        """Get key economic events for the week"""
        if self.market == "Indian":
            events = [
                "<li><strong>Reserve Bank of India (RBI)</strong> - Monitor for policy updates and rates</li>",
                "<li><strong>GDP Growth Data</strong> - Quarterly GDP and industrial production</li>",
                "<li><strong>Inflation Data</strong> - CPI and WPI releases from MOSPI</li>",
                "<li><strong>FII/DII Flows</strong> - Foreign and domestic institutional investments</li>",
                "<li><strong>Monsoon Updates</strong> - Weather patterns affecting agriculture sector</li>",
                "<li><strong>GST Collections</strong> - Monthly tax collection data</li>"
            ]
        else:
            events = [
                "<li><strong>Federal Reserve</strong> - Monitor for policy updates</li>",
                "<li><strong>Employment Data</strong> - Weekly jobless claims (Thursday)</li>",
                "<li><strong>Inflation Indicators</strong> - CPI/PPI releases if scheduled</li>",
                "<li><strong>International Markets</strong> - Central bank meetings and data</li>",
                "<li><strong>Sector Rotation</strong> - Watch for institutional flows</li>"
            ]
        return "".join(events)
    
    def _get_market_context(self):
        """Get current market data for AI analysis"""
        try:
            if self.market == "Indian":
                # Indian market indices
                indices = ['^NSEI', '^BSESN', '^NSEBANK']  # Nifty 50, Sensex, Bank Nifty
                index_names = {'^NSEI': 'Nifty 50', '^BSESN': 'Sensex', '^NSEBANK': 'Bank Nifty'}
            else:
                # US market indices
                indices = ['^GSPC', '^IXIC', '^DJI']  # S&P 500, NASDAQ, Dow
                index_names = {'^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones'}
            
            market_data = {}
            
            for index in indices:
                ticker = yf.Ticker(index)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    market_data[index] = {
                        'price': current_price,
                        'change_pct': change_pct
                    }
            
            if self.market == "Indian":
                context = f"""
                Indian Market Data Summary:
                Nifty 50: {market_data.get('^NSEI', {}).get('change_pct', 0):.2f}% change
                Sensex: {market_data.get('^BSESN', {}).get('change_pct', 0):.2f}% change
                Bank Nifty: {market_data.get('^NSEBANK', {}).get('change_pct', 0):.2f}% change
                Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
                Market: Indian Stock Exchanges (NSE/BSE)
                """
            else:
                context = f"""
                Market Data Summary:
                S&P 500: {market_data.get('^GSPC', {}).get('change_pct', 0):.2f}% change
                NASDAQ: {market_data.get('^IXIC', {}).get('change_pct', 0):.2f}% change
                Dow Jones: {market_data.get('^DJI', {}).get('change_pct', 0):.2f}% change
                Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
                """
            
            return context
            
        except Exception as e:
            return f"Error getting market context: {str(e)}"
    
    def _get_action_items_data(self):
        """Get data for generating action items"""
        try:
            if self.market == "Indian":
                # Major Indian stocks
                sample_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
            else:
                # US stocks
                sample_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            
            stock_data = []
            
            for symbol in sample_stocks:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1mo")
                    
                    if not hist.empty and info:
                        current_price = hist['Close'].iloc[-1]
                        month_ago_price = hist['Close'].iloc[0]
                        month_change = ((current_price - month_ago_price) / month_ago_price) * 100
                        
                        display_symbol = symbol.replace('.NS', '') if self.market == "Indian" else symbol
                        stock_data.append(f"{display_symbol}: {month_change:.1f}% monthly change")
                except:
                    continue
            
            market_label = "Indian Stock Performance Summary" if self.market == "Indian" else "Stock Performance Summary"
            return f"{market_label}: " + ", ".join(stock_data)
            
        except Exception as e:
            return f"Error getting action items data: {str(e)}"
    
    def send_email(self, recipient_email, email_html):
        """Send email using Gmail SMTP"""
        try:
            # Get Gmail credentials from user configuration
            gmail_email, gmail_password = SecretsHelper.get_gmail_credentials()
            
            if not gmail_email or not gmail_password:
                return False, "Gmail credentials not configured. Please set them in Admin → API Keys."
            
            # Create message
            msg = MIMEMultipart('alternative')
            market_label = f" - {self.market} Markets" if hasattr(self, 'market') else ""
            msg['Subject'] = f"Weekly Market Analysis{market_label} - {datetime.now().strftime('%B %d, %Y')}"
            msg['From'] = gmail_email
            msg['To'] = recipient_email
            
            # Add HTML content
            html_part = MIMEText(email_html, 'html')
            msg.attach(html_part)
            
            # Connect to Gmail SMTP server
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(gmail_email, gmail_password)
            
            # Send email
            text = msg.as_string()
            server.sendmail(gmail_email, recipient_email, text)
            server.quit()
            
            return True, "Email sent successfully!"
            
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"
    
    def generate_complete_email(self, force_regenerate=False):
        """Generate or retrieve the complete weekly email HTML"""
        try:
            # Check if email already exists for this week
            if not force_regenerate:
                existing_email = self.check_existing_email()
                if existing_email:
                    return existing_email['email_html'], True  # Return HTML and existing flag
            
            # Generate new email content
            week_ahead = self.generate_week_ahead_section()
            ai_insights = self.generate_ai_market_insights()
            action_items = self.generate_action_items()
            
            market_title = f"{self.market} " if self.market == "Indian" else ""
            
            # Create complete email template
            email_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Weekly {market_title}Market Analysis</title>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; border-radius: 8px; }}
                    .footer {{ background: #ecf0f1; padding: 15px; text-align: center; margin-top: 30px; border-radius: 8px; }}
                    .week-badge {{ background: #3498db; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>📊 Weekly {market_title}Market Analysis</h1>
                    <p>Week of {self.week_start.strftime('%B %d, %Y')}</p>
                    <span class="week-badge">{self.week_year}</span>
                </div>
                
                {week_ahead}
                {ai_insights}
                {action_items}
                
                <div class="footer">
                    <p>Generated by Intellicap Finance Analysis Platform</p>
                    <p style="font-size: 12px; color: #7f8c8d;">
                        This analysis is for informational purposes only and does not constitute financial advice.
                    </p>
                    <p style="font-size: 10px; color: #95a5a6;">
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
                    </p>
                </div>
            </body>
            </html>
            """
            
            # Save to database
            market_label = f" - {self.market} Markets" if self.market else ""
            email_subject = f"Weekly Market Analysis{market_label} - {self.week_start.strftime('%B %d, %Y')}"
            
            email_id = self.save_email_to_db(email_html, email_subject)
            
            return email_html, False  # Return HTML and new flag
            
        except Exception as e:
            error_html = f"<html><body><h1>Error generating email: {str(e)}</h1></body></html>"
            return error_html, False

def weekly_email_tab():
    """Weekly Email tab interface"""
    st.markdown("### 📧 Weekly Market Analysis Email")
    st.markdown("Generate and preview your weekly market analysis email with AI insights and actionable recommendations.")
    st.markdown("---")
    
    # Market selection
    col_market, col_spacer = st.columns([2, 3])
    with col_market:
        market_choice = st.radio(
            "🌍 Choose Market:",
            ["🇺🇸 US Markets", "🇮🇳 Indian Markets"],
            help="Select which market to focus the weekly analysis on"
        )
    
    selected_market = "Indian" if market_choice == "🇮🇳 Indian Markets" else "US"
    
    # Email generator interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 Email Content Preview")
        
        # Email generation controls
        col_gen1, col_gen2 = st.columns([2, 1])
        with col_gen1:
            generate_btn = st.button("🚀 Generate Weekly Email", type="primary")
        with col_gen2:
            force_regenerate = st.checkbox("🔄 Force Regenerate", help="Generate new content even if this week's email already exists")
        
        if generate_btn:
            email_generator = WeeklyEmailGenerator(market=selected_market)
            
            # Check current week info
            st.info(f"📅 Current week: {email_generator.week_year} ({email_generator.week_start.strftime('%B %d')} - {email_generator.week_end.strftime('%B %d, %Y')})")
            
            with st.spinner(f"Preparing your weekly {selected_market.lower()} market analysis..."):
                # Generate email content
                email_html, was_existing = email_generator.generate_complete_email(force_regenerate=force_regenerate)
                
                # Display status
                if was_existing and not force_regenerate:
                    st.success(f"✅ Retrieved existing weekly {selected_market} market email for {email_generator.week_year}")
                    st.info("💡 This email was previously generated this week. Use 'Force Regenerate' to create new content.")
                else:
                    st.success(f"✅ Weekly {selected_market} market email generated and saved to database!")
                
                # Display preview
                st.markdown("#### 📧 Email Preview")
                st.components.v1.html(email_html, height=800, scrolling=True)
                
                # Store in session state for sending
                st.session_state.generated_email = email_html
                st.session_state.email_market = selected_market
                st.session_state.email_week = email_generator.week_year
    
    with col2:
        st.markdown("#### ⚙️ Email Settings")
        
        # Email configuration
        user_email = st.text_input("📧 Your Email", placeholder="your.email@example.com")
        
        st.markdown("**📅 Content Sections:**")
        st.info("""
        ✅ **Week Ahead**
        - Upcoming earnings
        - Economic events
        
        ✅ **AI Market Insights**
        - Market sentiment analysis
        - Sector recommendations
        
        ✅ **Action Items**
        - Specific stock recommendations
        - Risk management tips
        """)
        
        # Send email functionality
        if st.button("📤 Send Email"):
            if user_email and 'generated_email' in st.session_state:
                market_type = st.session_state.get('email_market', 'US')
                
                with st.spinner("Sending email..."):
                    # Create email generator instance to access send_email method
                    email_generator = WeeklyEmailGenerator(market=market_type)
                    success, message = email_generator.send_email(user_email, st.session_state.generated_email)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.markdown(f"📧 **Email sent to:** {user_email}")
                        st.markdown(f"🌍 **Market Focus:** {market_type} Markets")
                        st.balloons()
                    else:
                        st.error(f"❌ {message}")
                        st.info("💡 **Troubleshooting Tips:**")
                        st.markdown("""
                        - Verify your Gmail credentials are correct
                        - Ensure 2-factor authentication is enabled on Gmail
                        - Check that you're using an App Password (not your regular password)
                        - Make sure Gmail allows less secure app access
                        """)
            else:
                st.warning("⚠️ Please generate email content first and enter your email address")
        
        st.markdown("---")
        st.markdown("#### 📚 Email History")
        
        # Show recent emails for selected market
        email_generator = WeeklyEmailGenerator(market=selected_market)
        recent_emails = email_generator.get_recent_emails()
        
        if recent_emails:
            st.markdown("**Recent Weekly Emails:**")
            for email in recent_emails:
                with st.expander(f"📧 {email['email_subject']} ({email['generated_at'].strftime('%b %d, %Y')})"):
                    col_email1, col_email2 = st.columns([2, 1])
                    with col_email1:
                        st.markdown(f"**Week:** {email['week_year']}")
                        st.markdown(f"**Market:** {email['market_type']}")
                        st.markdown(f"**Generated:** {email['generated_at'].strftime('%Y-%m-%d %H:%M UTC')}")
                    with col_email2:
                        if st.button(f"📄 View", key=f"view_{email['id']}"):
                            # Load this email for viewing
                            conn = email_generator._get_db_connection()
                            cursor = conn.cursor()
                            cursor.execute("SELECT email_html FROM weekly_emails WHERE id = %s", (email['id'],))
                            result = cursor.fetchone()
                            conn.close()
                            
                            if result:
                                st.session_state.generated_email = result['email_html']
                                st.session_state.email_market = email['market_type']
                                st.rerun()
        else:
            st.info("No previous emails found for this market.")
        
        st.markdown("---")
        st.markdown("#### 🔄 Database-Driven System")
        st.info("""
        **Current Features:**
        - ✅ Weekly emails stored in PostgreSQL database
        - ✅ Prevents duplicate generation for same week
        - ✅ Automatic Monday-Sunday week calculation
        - ✅ Separate storage for US and Indian markets
        - ✅ Email history and retrieval
        
        **Scheduling Logic:**
        - Emails generated per calendar week (2025-W37, etc.)
        - Force regenerate available for content updates
        - Database ensures consistency across sessions
        """)

if __name__ == "__main__":
    weekly_email_tab()