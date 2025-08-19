import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
import io
import requests
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill

def apply_gurufocus_view_mode_css():
    """Apply CSS styling based on the selected view mode for GuruFocus"""
    view_mode = st.session_state.get('gurufocus_view_mode', 'Standard')
    
    if view_mode == 'Compact':
        st.markdown("""
        <style>
        /* Compact Mode - Reduce metric font sizes */
        .stMetric > div > div > div {
            font-size: 0.65rem !important;
            line-height: 1.0 !important;
        }
        .stMetric > div > div > div > div {
            font-size: 0.75rem !important;
            margin-bottom: 0.1rem !important;
        }
        .stMetric [data-testid="metric-container"] {
            padding: 0.2rem 0 !important;
        }
        
        /* Target metric values specifically */
        .stMetric [data-testid="metric-container"] > div:first-child {
            font-size: 0.75rem !important;
        }
        .stMetric [data-testid="metric-container"] > div:last-child {
            font-size: 0.65rem !important;
        }
        
        /* Reduce metric value numbers */
        .stMetric .metric-value {
            font-size: 0.85rem !important;
        }
        
        /* More specific metric targeting */
        div[data-testid="metric-container"] div {
            font-size: 0.75rem !important;
        }
        div[data-testid="metric-container"] > div > div > div {
            font-size: 0.75rem !important;
        }
        
        /* Target the actual number values in metrics */
        .stMetric > div:first-child > div:first-child > div:first-child {
            font-size: 0.85rem !important;
        }
        
        /* Reduce header sizes */
        h1 {
            font-size: 1.5rem !important;
            margin: 0.3rem 0 0.2rem 0 !important;
        }
        h2 {
            font-size: 1.2rem !important;
            margin: 0.3rem 0 0.1rem 0 !important;
        }
        h3 {
            font-size: 1.0rem !important;
            margin: 0.2rem 0 0.1rem 0 !important;
        }
        .stSubheader {
            font-size: 0.95rem !important;
            margin: 0.2rem 0 0.1rem 0 !important;
        }
        
        /* Reduce DataFrame font sizes */
        .stDataFrame {
            font-size: 0.75rem !important;
        }
        .stDataFrame table {
            font-size: 0.75rem !important;
        }
        
        /* Reduce button and input sizes */
        .stButton button {
            font-size: 0.8rem !important;
            padding: 0.25rem 0.75rem !important;
        }
        
        /* Reduce column spacing */
        .stColumn {
            padding: 0.25rem !important;
        }
        
        /* Reduce expander font size */
        .streamlit-expanderHeader {
            font-size: 0.9rem !important;
        }
        
        /* Reduce markdown text size */
        .stMarkdown p {
            font-size: 0.85rem !important;
            margin-bottom: 0.3rem !important;
        }
        
        /* Reduce divider spacing */
        hr {
            margin: 0.5rem 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:  # Standard mode
        st.markdown("""
        <style>
        /* Standard Mode - Normal sizes */
        .stMetric > div > div > div {
            font-size: 0.875rem !important;
            line-height: 1.2 !important;
        }
        .stMetric > div > div > div > div {
            font-size: 1.25rem !important;
            margin-bottom: 0.25rem !important;
        }
        .stMetric [data-testid="metric-container"] {
            padding: 0.5rem 0 !important;
        }
        
        /* Standard header sizes */
        h1 {
            font-size: 2.25rem !important;
            margin: 1rem 0 0.5rem 0 !important;
        }
        h2 {
            font-size: 1.875rem !important;
            margin: 0.75rem 0 0.5rem 0 !important;
        }
        h3 {
            font-size: 1.5rem !important;
            margin: 0.5rem 0 0.25rem 0 !important;
        }
        .stSubheader {
            font-size: 1.25rem !important;
            margin: 0.5rem 0 0.25rem 0 !important;
        }
        
        /* Standard DataFrame sizes */
        .stDataFrame {
            font-size: 0.875rem !important;
        }
        
        /* Standard button sizes */
        .stButton button {
            font-size: 0.875rem !important;
            padding: 0.5rem 1rem !important;
        }
        
        /* Standard column spacing */
        .stColumn {
            padding: 0.5rem !important;
        }
        
        /* Standard markdown text size */
        .stMarkdown p {
            font-size: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Standard divider spacing */
        hr {
            margin: 1rem 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

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
    """
    Fetch stock profile data from GuruFocus API
    
    Args:
        symbol (str): Stock ticker symbol
        api_key (str): GuruFocus API key
    
    Returns:
        dict: Profile data or None if error
    """
    try:
        url = f"{GURUFOCUS_BASE_URL}/stocks/{symbol}/profile"
        headers = get_gurufocus_headers(api_key)
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.error("❌ Invalid API key or insufficient permissions")
            return None
        elif response.status_code == 404:
            st.error(f"❌ Stock symbol '{symbol}' not found")
            return None
        else:
            st.error(f"❌ API Error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout - please try again")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error fetching profile data: {str(e)}")
        return None

def fetch_stock_fundamentals(symbol, api_key):
    """
    Fetch stock fundamentals data from GuruFocus API
    
    Args:
        symbol (str): Stock ticker symbol
        api_key (str): GuruFocus API key
    
    Returns:
        dict: Fundamentals data or None if error
    """
    try:
        url = f"{GURUFOCUS_BASE_URL}/stocks/{symbol}/fundamentals"
        headers = get_gurufocus_headers(api_key)
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Could not fetch fundamentals: {response.status_code}")
            return None
            
    except Exception as e:
        st.warning(f"Error fetching fundamentals: {str(e)}")
        return None

def fetch_stock_valuations(symbol, api_key):
    """
    Fetch stock valuation metrics from GuruFocus API
    
    Args:
        symbol (str): Stock ticker symbol
        api_key (str): GuruFocus API key
    
    Returns:
        dict: Valuation data or None if error
    """
    try:
        url = f"{GURUFOCUS_BASE_URL}/stocks/{symbol}/valuations"
        headers = get_gurufocus_headers(api_key)
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Could not fetch valuations: {response.status_code}")
            return None
            
    except Exception as e:
        st.warning(f"Error fetching valuations: {str(e)}")
        return None

def fetch_stock_dividends(symbol, api_key):
    """
    Fetch stock dividend data from GuruFocus API
    
    Args:
        symbol (str): Stock ticker symbol
        api_key (str): GuruFocus API key
    
    Returns:
        dict: Dividend data or None if error
    """
    try:
        url = f"{GURUFOCUS_BASE_URL}/stocks/{symbol}/dividends"
        headers = get_gurufocus_headers(api_key)
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Could not fetch dividends: {response.status_code}")
            return None
            
    except Exception as e:
        st.warning(f"Error fetching dividends: {str(e)}")
        return None

def create_price_chart_from_fundamentals(fundamentals_data, symbol):
    """
    Create price chart using historical data from fundamentals
    
    Args:
        fundamentals_data (dict): Fundamentals data from GuruFocus
        symbol (str): Stock symbol
    
    Returns:
        plotly.graph_objects.Figure: Chart figure
    """
    try:
        # Extract historical price data if available in fundamentals
        if not fundamentals_data or 'historical_data' not in fundamentals_data:
            # Create a simple placeholder chart
            fig = go.Figure()
            fig.add_annotation(
                text=f"Historical price data not available for {symbol}<br>Please check GuruFocus data package access",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=f"{symbol} - Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600
            )
            return fig
        
        # Process historical data if available
        historical_data = fundamentals_data['historical_data']
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(historical_data)
        
        # Create the chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Stock Price Chart (GuruFocus Data)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        # Return simple placeholder chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart for {symbol}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def display_profile_metrics(profile_data, symbol):
    """
    Display key profile metrics from GuruFocus data
    
    Args:
        profile_data (dict): Profile data from GuruFocus
        symbol (str): Stock symbol
    """
    if not profile_data:
        st.warning("No profile data available")
        return
    
    try:
        # Extract basic information
        basic_info = profile_data.get('basic_information', {})
        company_name = basic_info.get('company', symbol)
        exchange = basic_info.get('exchange', 'N/A')
        
        st.subheader(f"📊 Company Profile: {company_name}")
        
        # Display basic company information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Company", company_name)
            st.metric("Exchange", exchange)
            
        with col2:
            st.metric("Symbol", symbol)
            company_id = basic_info.get('company_id', 'N/A')
            st.metric("GuruFocus ID", company_id)
            
        with col3:
            stock_id = basic_info.get('stockid', 'N/A')
            st.metric("Stock ID", stock_id)
        
        # Display additional profile metrics if available
        if 'metrics' in profile_data:
            metrics = profile_data['metrics']
            st.subheader("📈 Key Metrics")
            
            metric_cols = st.columns(4)
            
            metric_items = [
                ('Market Cap', metrics.get('market_cap')),
                ('P/E Ratio', metrics.get('pe_ratio')),
                ('Revenue', metrics.get('revenue')),
                ('Profit Margin', metrics.get('profit_margin'))
            ]
            
            for i, (label, value) in enumerate(metric_items):
                with metric_cols[i % 4]:
                    if value is not None:
                        if label == 'Market Cap' and isinstance(value, (int, float)):
                            st.metric(label, f"${value:,.0f}")
                        elif label == 'Revenue' and isinstance(value, (int, float)):
                            st.metric(label, f"${value:,.0f}")
                        elif label == 'Profit Margin' and isinstance(value, (int, float)):
                            st.metric(label, f"{value:.2%}")
                        else:
                            st.metric(label, str(value))
                    else:
                        st.metric(label, "N/A")
        
    except Exception as e:
        st.error(f"Error displaying profile metrics: {str(e)}")

def display_fundamentals_summary(fundamentals_data, symbol):
    """
    Display fundamentals summary from GuruFocus data
    
    Args:
        fundamentals_data (dict): Fundamentals data from GuruFocus
        symbol (str): Stock symbol
    """
    if not fundamentals_data:
        st.warning("No fundamentals data available")
        return
    
    try:
        st.subheader(f"📋 Financial Fundamentals: {symbol}")
        
        # Display latest financial metrics if available
        if 'latest_financials' in fundamentals_data:
            latest = fundamentals_data['latest_financials']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                revenue = latest.get('revenue')
                if revenue:
                    st.metric("Revenue", f"${revenue:,.0f}" if isinstance(revenue, (int, float)) else str(revenue))
                
                net_income = latest.get('net_income')
                if net_income:
                    st.metric("Net Income", f"${net_income:,.0f}" if isinstance(net_income, (int, float)) else str(net_income))
            
            with col2:
                total_assets = latest.get('total_assets')
                if total_assets:
                    st.metric("Total Assets", f"${total_assets:,.0f}" if isinstance(total_assets, (int, float)) else str(total_assets))
                
                total_debt = latest.get('total_debt')
                if total_debt:
                    st.metric("Total Debt", f"${total_debt:,.0f}" if isinstance(total_debt, (int, float)) else str(total_debt))
            
            with col3:
                book_value = latest.get('book_value')
                if book_value:
                    st.metric("Book Value", f"${book_value:,.0f}" if isinstance(book_value, (int, float)) else str(book_value))
                
                cash = latest.get('cash_and_equivalents')
                if cash:
                    st.metric("Cash & Equivalents", f"${cash:,.0f}" if isinstance(cash, (int, float)) else str(cash))
            
            with col4:
                eps = latest.get('earnings_per_share')
                if eps:
                    st.metric("EPS", f"${eps:.2f}" if isinstance(eps, (int, float)) else str(eps))
                
                shares = latest.get('shares_outstanding')
                if shares:
                    st.metric("Shares Outstanding", f"{shares:,.0f}" if isinstance(shares, (int, float)) else str(shares))
        
    except Exception as e:
        st.error(f"Error displaying fundamentals: {str(e)}")

def display_dividend_info(dividend_data, symbol):
    """
    Display dividend information from GuruFocus data
    
    Args:
        dividend_data (dict): Dividend data from GuruFocus
        symbol (str): Stock symbol
    """
    if not dividend_data:
        st.info("No dividend data available")
        return
    
    try:
        st.subheader(f"💰 Dividend Information: {symbol}")
        
        # Get basic information and dividends
        basic_info = dividend_data.get('basic_information', {})
        dividends = dividend_data.get('dividends', [])
        
        if dividends:
            # Get most recent dividend
            recent_dividend = dividends[-1] if dividends else {}
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cash_amount = recent_dividend.get('cash_amount', 0)
                st.metric("Latest Dividend", f"${cash_amount:.4f}" if cash_amount else "N/A")
                
                pay_date = recent_dividend.get('pay_date', 'N/A')
                st.metric("Last Pay Date", pay_date)
            
            with col2:
                announce_date = recent_dividend.get('announce_date', 'N/A')
                st.metric("Last Announce Date", announce_date)
                
                frequency = recent_dividend.get('Frequency', 'N/A')
                st.metric("Frequency", f"{frequency} times/year" if frequency != 'N/A' else 'N/A')
            
            with col3:
                dividend_type = recent_dividend.get('dividend_type', 'N/A')
                st.metric("Dividend Type", dividend_type)
                
                currency = recent_dividend.get('CurrencyId', 'USD')
                st.metric("Currency", currency)
            
            # Show dividend history table
            if len(dividends) > 1:
                st.subheader("📜 Recent Dividend History")
                
                # Get last 10 dividends
                recent_dividends = dividends[-10:] if len(dividends) > 10 else dividends
                
                df_dividends = pd.DataFrame(recent_dividends)
                display_columns = ['announce_date', 'pay_date', 'cash_amount', 'dividend_type']
                
                # Filter to available columns
                available_columns = [col for col in display_columns if col in df_dividends.columns]
                
                if available_columns:
                    df_display = df_dividends[available_columns].copy()
                    
                    # Format cash amount
                    if 'cash_amount' in df_display.columns:
                        df_display = df_display.copy()
                        df_display['cash_amount'] = df_display['cash_amount'].astype(str).apply(
                            lambda x: f"${float(x):.4f}" if x != 'N/A' and pd.notnull(x) and str(x).replace('.','',1).isdigit() else "N/A"
                        )
                    
                    # Rename columns for display
                    column_names = {
                        'announce_date': 'Announce Date',
                        'pay_date': 'Pay Date',
                        'cash_amount': 'Amount',
                        'dividend_type': 'Type'
                    }
                    
                    df_display = df_display.rename(columns=column_names)
                    if 'Announce Date' in df_display.columns:
                        df_display = df_display.sort_values('Announce Date', ascending=False)
                    
                    st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No dividend history available for this stock")
            
    except Exception as e:
        st.error(f"Error displaying dividend information: {str(e)}")

def bulk_analysis_gurufocus(symbols, api_key):
    """
    Perform bulk analysis using GuruFocus API
    
    Args:
        symbols (list): List of stock symbols
        api_key (str): GuruFocus API key
    
    Returns:
        pandas.DataFrame: Results dataframe
    """
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        try:
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
            
            # Fetch profile data
            profile_data = fetch_stock_profile(symbol, api_key)
            
            # Fetch fundamentals data
            fundamentals_data = fetch_stock_fundamentals(symbol, api_key)
            
            # Fetch dividend data
            dividend_data = fetch_stock_dividends(symbol, api_key)
            
            # Extract key metrics
            result = {
                'Symbol': symbol,
                'Company': 'N/A',
                'Exchange': 'N/A',
                'Market Cap': 'N/A',
                'Revenue': 'N/A',
                'Net Income': 'N/A',
                'P/E Ratio': 'N/A',
                'Book Value': 'N/A',
                'Latest Dividend': 'N/A',
                'Dividend Frequency': 'N/A',
                'Status': 'Success'
            }
            
            # Extract profile information
            if profile_data and 'basic_information' in profile_data:
                basic_info = profile_data['basic_information']
                result['Company'] = basic_info.get('company', 'N/A')
                result['Exchange'] = basic_info.get('exchange', 'N/A')
                
                if 'metrics' in profile_data:
                    metrics = profile_data['metrics']
                    result['Market Cap'] = metrics.get('market_cap', 'N/A')
                    result['P/E Ratio'] = metrics.get('pe_ratio', 'N/A')
            
            # Extract fundamentals information
            if fundamentals_data and 'latest_financials' in fundamentals_data:
                latest = fundamentals_data['latest_financials']
                result['Revenue'] = latest.get('revenue', 'N/A')
                result['Net Income'] = latest.get('net_income', 'N/A')
                result['Book Value'] = latest.get('book_value', 'N/A')
            
            # Extract dividend information
            if dividend_data and 'dividends' in dividend_data:
                dividends = dividend_data['dividends']
                if dividends:
                    recent_dividend = dividends[-1]
                    result['Latest Dividend'] = recent_dividend.get('cash_amount', 'N/A')
                    result['Dividend Frequency'] = recent_dividend.get('Frequency', 'N/A')
            
            results.append(result)
            
        except Exception as e:
            # Add error result
            results.append({
                'Symbol': symbol,
                'Company': 'Error',
                'Exchange': 'Error',
                'Market Cap': 'Error',
                'Revenue': 'Error', 
                'Net Income': 'Error',
                'P/E Ratio': 'Error',
                'Book Value': 'Error',
                'Latest Dividend': 'Error',
                'Dividend Frequency': 'Error',
                'Status': f'Error: {str(e)[:50]}'
            })
        
        # Update progress
        progress_bar.progress((i + 1) / len(symbols))
        time.sleep(0.1)  # Small delay to avoid rate limiting
    
    status_text.text("Analysis complete!")
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def create_excel_report_gurufocus(df, filename="gurufocus_stock_analysis.xlsx"):
    """
    Create Excel report for GuruFocus bulk analysis
    
    Args:
        df (pandas.DataFrame): Analysis results
        filename (str): Output filename
    
    Returns:
        BytesIO: Excel file buffer
    """
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Stock Analysis', index=False)
            
            # Get workbook and worksheet for formatting
            workbook = writer.book
            worksheet = writer.sheets['Stock Analysis']
            
            # Header formatting
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_alignment = Alignment(horizontal="center")
            
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Create summary sheet
            summary_data = {
                'Metric': ['Total Stocks Analyzed', 'Successful Analyses', 'Failed Analyses', 'Success Rate'],
                'Value': [
                    len(df),
                    len(df[df['Status'] == 'Success']),
                    len(df[df['Status'] != 'Success']),
                    f"{(len(df[df['Status'] == 'Success']) / len(df) * 100):.1f}%" if len(df) > 0 else "0%"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Format summary sheet
            summary_ws = writer.sheets['Summary']
            for cell in summary_ws[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
            
            # Auto-adjust summary column widths
            for column in summary_ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 30)
                summary_ws.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        return output
        
    except Exception as e:
        st.error(f"Error creating Excel report: {str(e)}")
        return None

# Main application
def main():
    # Navigation bar
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("🏠 Main Menu", help="Return to landing page"):
            st.markdown('<meta http-equiv="refresh" content="0; url=http://localhost:3000">', unsafe_allow_html=True)
    with col_nav2:
        st.markdown("**🏦 GuruFocus Version**")
    with col_nav3:
        if st.button("📈 Yahoo Finance Version", help="Switch to free technical analysis"):
            st.markdown('<meta http-equiv="refresh" content="0; url=http://localhost:5000">', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.title("📊 GuruFocus Stock Analysis Application")
    st.markdown("Advanced stock analysis using GuruFocus professional data")
    
    # View Mode Selector
    st.markdown("---")
    col_view1, col_view2, col_view3 = st.columns([1, 2, 1])
    
    with col_view2:
        view_mode = st.radio(
            "Display View:",
            ["Standard", "Compact"],
            horizontal=True,
            help="Standard: Normal font sizes and spacing. Compact: Reduced font sizes and minimal spacing for less scrolling.",
            key="gurufocus_view_mode_selector"
        )
        
        # Store view mode in session state
        st.session_state['gurufocus_view_mode'] = view_mode
    
    st.markdown("---")
    
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
    
    # Analysis mode selection
    st.sidebar.header("📈 Analysis Mode")
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["Single Stock Analysis", "Bulk Stock Analysis"]
    )
    
    if analysis_mode == "Single Stock Analysis":
        single_stock_analysis(api_key)
    else:
        bulk_stock_analysis(api_key)

def single_stock_analysis(api_key):
    """Single stock analysis interface"""
    
    # Apply view mode styling
    apply_gurufocus_view_mode_css()
    
    st.header("🔍 Single Stock Analysis")
    
    # Stock input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol:",
            placeholder="e.g., AAPL, MSFT, TSLA",
            help="Enter any stock ticker symbol"
        ).upper().strip()
    
    with col2:
        analyze_button = st.button("📊 Analyze Stock", type="primary")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (10 min)", help="Automatically refresh data every 10 minutes")
    
    # Auto-refresh functionality
    if auto_refresh and symbol:
        # Initialize refresh state
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = time.time()
            st.session_state.refresh_count = 0
        
        current_time = time.time()
        time_since_refresh = int(current_time - st.session_state.last_refresh_time)
        
        # Check if 10 minutes have passed
        if time_since_refresh >= 600:  # 600 seconds = 10 minutes
            st.session_state.last_refresh_time = current_time
            st.session_state.refresh_count += 1
            st.rerun()
        
        # Display refresh status with countdown
        minutes_since = time_since_refresh // 60
        seconds_since = time_since_refresh % 60
        next_refresh_in = 600 - time_since_refresh
        next_refresh_min = next_refresh_in // 60
        next_refresh_sec = next_refresh_in % 60
        
        refresh_info = f"🔄 Auto-refresh active (#{st.session_state.refresh_count}) | Last update: {minutes_since}m {seconds_since}s ago | Next refresh: {next_refresh_min}m {next_refresh_sec}s"
        st.info(refresh_info)
        
        # Add auto-refresh timer with meta refresh
        st.markdown(f'<meta http-equiv="refresh" content="{next_refresh_in}">', unsafe_allow_html=True)
    
    # Process the request when button is clicked or symbol is entered
    if analyze_button or symbol:
        if symbol:
            # Show loading spinner
            with st.spinner(f'Fetching GuruFocus data for {symbol}...'):
                # Fetch all data types
                profile_data = fetch_stock_profile(symbol, api_key)
                fundamentals_data = fetch_stock_fundamentals(symbol, api_key)
                valuations_data = fetch_stock_valuations(symbol, api_key)
                dividend_data = fetch_stock_dividends(symbol, api_key)
            
            if profile_data or fundamentals_data or dividend_data:
                # Display auto-refresh status and timestamp
                if auto_refresh:
                    col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
                    with col_status1:
                        st.success(f"✅ Live tracking {symbol} - Updates every 10 minutes")
                    with col_status2:
                        st.metric("Refresh #", st.session_state.get('refresh_count', 0))
                    with col_status3:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        st.metric("Last Updated", current_time)
                
                # Display profile metrics
                if profile_data:
                    display_profile_metrics(profile_data, symbol)
                
                # Display fundamentals summary
                if fundamentals_data:
                    display_fundamentals_summary(fundamentals_data, symbol)
                
                # Create and display chart
                st.subheader(f"📈 Price Chart: {symbol}")
                chart = create_price_chart_from_fundamentals(fundamentals_data, symbol)
                st.plotly_chart(chart, use_container_width=True)
                
                # Display dividend information
                if dividend_data:
                    display_dividend_info(dividend_data, symbol)
                
                # Display valuation metrics if available
                if valuations_data:
                    st.subheader(f"💹 Valuation Metrics: {symbol}")
                    st.json(valuations_data)  # Display raw valuation data for now
                
            else:
                st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol and your API access.")
        else:
            st.warning("⚠️ Please enter a stock symbol")

def bulk_stock_analysis(api_key):
    """Bulk stock analysis interface"""
    
    # Apply view mode styling
    apply_gurufocus_view_mode_css()
    
    st.header("📊 Bulk Stock Analysis")
    
    # Stock list input
    st.subheader("📝 Enter Stock Symbols")
    
    # Text area for multiple symbols
    symbols_input = st.text_area(
        "Enter stock symbols (one per line or comma-separated):",
        placeholder="AAPL\nMSFT\nTSLA\nGOOGL\nAMZN",
        height=100
    )
    
    # Saved lists functionality
    st.subheader("💾 Saved Stock Lists")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Save current list
        save_name = st.text_input("List name:", placeholder="My Portfolio")
        if st.button("💾 Save List") and save_name and symbols_input:
            if 'saved_lists' not in st.session_state:
                st.session_state.saved_lists = {}
            
            # Process symbols
            symbols = []
            if ',' in symbols_input:
                symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            else:
                symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
            
            st.session_state.saved_lists[save_name] = symbols
            st.success(f"✅ Saved '{save_name}' with {len(symbols)} symbols")
    
    with col2:
        # Load saved list
        if 'saved_lists' in st.session_state and st.session_state.saved_lists:
            selected_list = st.selectbox("Select saved list:", [""] + list(st.session_state.saved_lists.keys()))
            if st.button("📂 Load List") and selected_list:
                loaded_symbols = st.session_state.saved_lists[selected_list]
                st.session_state.symbols_input = '\n'.join(loaded_symbols)
                st.success(f"✅ Loaded '{selected_list}' with {len(loaded_symbols)} symbols")
                st.rerun()
    
    with col3:
        # Delete saved list
        if 'saved_lists' in st.session_state and st.session_state.saved_lists:
            delete_list = st.selectbox("Delete list:", [""] + list(st.session_state.saved_lists.keys()))
            if st.button("🗑️ Delete List") and delete_list:
                del st.session_state.saved_lists[delete_list]
                st.success(f"✅ Deleted list '{delete_list}'")
                st.rerun()
    
    # Update symbols input if loaded from session state
    if 'symbols_input' in st.session_state:
        symbols_input = st.session_state.symbols_input
        del st.session_state.symbols_input
    
    # Analyze button
    if st.button("🚀 Start Bulk Analysis", type="primary"):
        if symbols_input:
            # Process symbols
            symbols = []
            if ',' in symbols_input:
                symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            else:
                symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
            
            if symbols:
                st.info(f"🔍 Analyzing {len(symbols)} stocks using GuruFocus API...")
                
                # Perform bulk analysis
                results_df = bulk_analysis_gurufocus(symbols, api_key)
                
                # Display results
                st.subheader("📊 Analysis Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Success/failure summary
                successful = len(results_df[results_df['Status'] == 'Success'])
                total = len(results_df)
                st.metric("Success Rate", f"{successful}/{total} ({successful/total*100:.1f}%)")
                
                # Excel download
                excel_buffer = create_excel_report_gurufocus(results_df)
                if excel_buffer:
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"gurufocus_analysis_{current_time}.xlsx"
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("⚠️ Please enter at least one stock symbol")
        else:
            st.warning("⚠️ Please enter stock symbols to analyze")

if __name__ == "__main__":
    main()