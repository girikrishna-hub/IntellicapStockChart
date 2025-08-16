import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
import io
import json
import hashlib
import urllib.parse
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from news_sentiment_analyzer import run_sentiment_analysis, get_sentiment_summary_for_sharing

# Set page configuration
st.set_page_config(
    page_title="Stock 50-Day Moving Average Chart",
    page_icon="📈",
    layout="wide"
)

def fetch_stock_data(symbol, period="1y", market="US"):
    """
    Fetch stock data for the given symbol and period
    
    Args:
        symbol (str): Stock ticker symbol
        period (str): Time period for data retrieval
        market (str): Market type ("US" or "India")
    
    Returns:
        tuple: (historical_data, ticker_info, ticker_object) or (None, None, None) if error
    """
    try:
        # Format symbol for Indian stocks
        if market == "India":
            # Add .NS (NSE) or .BO (BSE) suffix if not present
            if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
                # Default to NSE (.NS) for Indian stocks
                symbol = f"{symbol}.NS"
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch historical data
        data = ticker.history(period=period)
        
        if data.empty:
            return None, None, None
        
        # Fetch additional company info
        try:
            info = ticker.info
        except:
            info = {}
            
        return data, info, ticker
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None, None

def get_beta_value(ticker_info):
    """
    Get the beta value for a stock
    
    Args:
        ticker_info (dict): Stock information from yfinance
    
    Returns:
        str: Beta value formatted as string
    """
    try:
        beta = ticker_info.get('beta', None)
        if beta is not None and not pd.isna(beta):
            return f"{beta:.2f}"
        else:
            return "N/A"
    except:
        return "N/A"

def export_chart_as_png(fig, filename_prefix="chart"):
    """
    Export a Plotly figure as PNG and return download data
    
    Args:
        fig: Plotly figure object
        filename_prefix (str): Prefix for the filename
        
    Returns:
        tuple: (bytes_data, filename)
    """
    try:
        # Try to convert figure to PNG bytes using Kaleido
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2, engine="kaleido")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        
        return img_bytes, filename
    except Exception as e:
        # If Kaleido fails, try alternative method using HTML export
        try:
            # Convert to HTML and then use a different approach
            html_str = fig.to_html(include_plotlyjs='cdn')
            
            # Create a simple PNG export message
            st.warning("PNG export is not available in this environment. Please use PDF export instead, or try the 'Download plot as PNG' option from the chart's toolbar (camera icon in the top-right corner of the chart).")
            return None, None
            
        except Exception as e2:
            st.error(f"Error exporting PNG: {str(e)}. Please try using the chart's built-in download feature (camera icon) or PDF export.")
            return None, None

def export_chart_as_pdf(fig, filename_prefix="chart", title="Stock Analysis Chart"):
    """
    Export a Plotly figure as PDF and return download data
    
    Args:
        fig: Plotly figure object
        filename_prefix (str): Prefix for the filename
        title (str): Title for the PDF document
        
    Returns:
        tuple: (bytes_data, filename)
    """
    try:
        # Try to convert figure to PNG first (for embedding in PDF)
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2, engine="kaleido")
        except:
            # If Kaleido fails, create a text-based PDF without the chart image
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.black,
                spaceAfter=20,
                alignment=1  # Center alignment
            )
            
            # Create story
            story = []
            
            # Add title
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 20))
            
            # Add note about chart
            note_text = "Chart image could not be embedded. Please use the chart's built-in download feature (camera icon in the top-right corner) to save the chart as an image."
            story.append(Paragraph(note_text, styles['Normal']))
            
            # Add timestamp
            timestamp_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Spacer(1, 20))
            story.append(Paragraph(timestamp_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            # Get PDF bytes
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.pdf"
            
            return pdf_bytes, filename
        
        # If we got the image, create PDF with chart
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.black,
            spaceAfter=20,
            alignment=1  # Center alignment
        )
        
        # Create story
        story = []
        
        # Add title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))
        
        # Add chart image
        img_stream = io.BytesIO(img_bytes)
        img = ReportLabImage(img_stream, width=7*inch, height=4.5*inch)
        story.append(img)
        
        # Add timestamp
        timestamp_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Spacer(1, 20))
        story.append(Paragraph(timestamp_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pdf"
        
        return pdf_bytes, filename
    except Exception as e:
        st.error(f"Error exporting PDF: {str(e)}")
        return None, None

def create_export_buttons(fig, chart_name, symbol=""):
    """
    Create export buttons for PNG and PDF download
    
    Args:
        fig: Plotly figure object
        chart_name (str): Name of the chart for filename
        symbol (str): Stock symbol for filename
    """
    # Prepare filename prefix
    filename_prefix = f"{symbol}_{chart_name}".replace(" ", "_").replace("/", "_") if symbol else chart_name.replace(" ", "_")
    
    # Show built-in chart download option first
    st.info("💡 **Tip:** You can also download charts directly using the camera icon (📷) in the chart's toolbar (top-right corner when you hover over the chart).")
    
    col_png, col_pdf = st.columns(2)
    
    with col_png:
        if st.button(f"📷 Export PNG", key=f"png_{chart_name}_{symbol}", help="Download chart as PNG image (may require browser setup)"):
            png_data, png_filename = export_chart_as_png(fig, filename_prefix)
            if png_data:
                st.download_button(
                    label="⬇️ Download PNG",
                    data=png_data,
                    file_name=png_filename,
                    mime="image/png",
                    key=f"download_png_{chart_name}_{symbol}"
                )
    
    with col_pdf:
        if st.button(f"📄 Export PDF", key=f"pdf_{chart_name}_{symbol}", help="Download chart as PDF document"):
            pdf_title = f"{symbol.upper()} - {chart_name}" if symbol else chart_name
            pdf_data, pdf_filename = export_chart_as_pdf(fig, filename_prefix, pdf_title)
            if pdf_data:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_data,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    key=f"download_pdf_{chart_name}_{symbol}"
                )

def create_shareable_insight(symbol, data, metrics, privacy_level="public", fibonacci_data=None):
    """
    Create a shareable investment insight with customizable privacy
    
    Args:
        symbol (str): Stock ticker symbol
        data (pd.DataFrame): Stock price data
        metrics (dict): Key stock metrics
        privacy_level (str): Privacy level ("public", "private", "anonymized")
        fibonacci_data (dict): Fibonacci analysis data
    
    Returns:
        dict: Shareable insight data
    """
    try:
        latest_price = data['Close'].iloc[-1]
        price_change = ((latest_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
        
        # Base insight structure
        insight = {
            "id": hashlib.md5(f"{symbol}_{datetime.now()}".encode()).hexdigest()[:8],
            "timestamp": datetime.now().isoformat(),
            "privacy_level": privacy_level,
            "analysis_type": "technical_analysis"
        }
        
        # Process Fibonacci data if available
        fib_summary = None
        if fibonacci_data:
            trend_dir = fibonacci_data.get('trend_direction', 'unknown')
            swing_high = fibonacci_data.get('swing_high', 0)
            swing_low = fibonacci_data.get('swing_low', 0)
            price_range = swing_high - swing_low
            
            # Find closest Fibonacci level
            uptrend_levels = fibonacci_data.get('uptrend_levels', {})
            closest_fib_level = None
            closest_distance = float('inf')
            
            for level_name, level_price in uptrend_levels.items():
                if level_name != 'Current Price':
                    distance = abs(latest_price - level_price)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_fib_level = level_name
            
            fib_summary = {
                "trend_direction": trend_dir,
                "swing_range": f"${price_range:.2f}",
                "closest_fib_level": closest_fib_level,
                "near_support_resistance": closest_fib_level is not None and closest_distance < (price_range * 0.02)
            }
        
        if privacy_level == "public":
            # Full public sharing
            public_data = {
                "symbol": symbol,
                "company_name": metrics.get('longName', symbol),
                "current_price": f"${latest_price:.2f}",
                "daily_change": f"{price_change:+.2f}%",
                "key_metrics": {
                    "rsi": metrics.get('RSI', 'N/A'),
                    "ma_50_signal": metrics.get('Price vs 50-Day MA (%)', 'N/A'),
                    "ma_200_signal": metrics.get('Price vs 200-Day MA (%)', 'N/A'),
                    "volume": metrics.get('Volume', 'N/A'),
                    "beta": metrics.get('Beta', 'N/A')
                },
                "recommendation": generate_advanced_recommendation(metrics, fib_summary),
                "risk_level": assess_risk_level(metrics)
            }
            
            if fib_summary:
                public_data["fibonacci_analysis"] = {
                    "trend": fib_summary["trend_direction"].title(),
                    "swing_range": fib_summary["swing_range"],
                    "key_level": fib_summary["closest_fib_level"] or "No nearby levels",
                    "near_key_level": fib_summary["near_support_resistance"]
                }
            
            insight.update(public_data)
            
        elif privacy_level == "anonymized":
            # Anonymized - no symbol or company name
            anon_data = {
                "symbol": "***",
                "company_name": "Anonymized Stock",
                "current_price": f"${latest_price:.2f}",
                "daily_change": f"{price_change:+.2f}%",
                "key_metrics": {
                    "rsi": metrics.get('RSI', 'N/A'),
                    "ma_50_signal": metrics.get('Price vs 50-Day MA (%)', 'N/A'),
                    "ma_200_signal": metrics.get('Price vs 200-Day MA (%)', 'N/A')
                },
                "recommendation": generate_advanced_recommendation(metrics, fib_summary),
                "note": "Stock identity hidden for privacy"
            }
            
            if fib_summary:
                anon_data["fibonacci_analysis"] = {
                    "trend": fib_summary["trend_direction"].title(),
                    "key_level": fib_summary["closest_fib_level"] or "No nearby levels"
                }
            
            insight.update(anon_data)
            
        else:  # private
            # Private sharing - limited info
            insight.update({
                "symbol": symbol,
                "analysis_summary": "Private technical analysis with Fibonacci levels completed",
                "recommendation": generate_advanced_recommendation(metrics, fib_summary),
                "shared_with": "Private analysis - limited details"
            })
        
        return insight
        
    except Exception as e:
        return {"error": f"Failed to create insight: {str(e)}"}

def generate_simple_recommendation(metrics):
    """Generate a simple investment recommendation based on key metrics"""
    try:
        rsi_text = metrics.get('RSI', '50')
        if rsi_text != 'N/A':
            rsi = float(rsi_text)
        else:
            rsi = 50
            
        ma_50_text = metrics.get('Price vs 50-Day MA (%)', '0')
        if ma_50_text != 'N/A':
            ma_50_signal = float(ma_50_text.replace('%', ''))
        else:
            ma_50_signal = 0
            
        # Simple recommendation logic
        if rsi > 70:
            return "⚠️ Potentially Overbought"
        elif rsi < 30:
            return "📈 Potentially Oversold - Watch for Entry"
        elif ma_50_signal > 5:
            return "📈 Above 50-Day MA - Bullish Trend"
        elif ma_50_signal < -5:
            return "📉 Below 50-Day MA - Bearish Trend"
        else:
            return "➡️ Neutral - Monitor for Clear Signals"
            
    except:
        return "➡️ Analysis Available - Review Details"

def generate_advanced_recommendation(metrics, fib_summary=None):
    """Generate advanced investment recommendation including Fibonacci analysis"""
    try:
        rsi_text = metrics.get('RSI', '50')
        if rsi_text != 'N/A':
            rsi = float(rsi_text)
        else:
            rsi = 50
            
        ma_50_text = metrics.get('Price vs 50-Day MA (%)', '0')
        if ma_50_text != 'N/A':
            ma_50_signal = float(ma_50_text.replace('%', ''))
        else:
            ma_50_signal = 0
            
        # Base recommendation from technical indicators
        base_rec = ""
        if rsi > 70:
            base_rec = "⚠️ Overbought"
        elif rsi < 30:
            base_rec = "📈 Oversold Entry Zone"
        elif ma_50_signal > 5:
            base_rec = "📈 Bullish Trend"
        elif ma_50_signal < -5:
            base_rec = "📉 Bearish Trend"
        else:
            base_rec = "➡️ Neutral"
            
        # Enhance with Fibonacci analysis
        if fib_summary:
            trend_dir = fib_summary.get("trend_direction", "unknown")
            near_key_level = fib_summary.get("near_support_resistance", False)
            
            if near_key_level:
                if trend_dir == "uptrend":
                    base_rec += " + Near Fib Support"
                elif trend_dir == "downtrend":
                    base_rec += " + Near Fib Resistance"
                else:
                    base_rec += " + At Key Fib Level"
            else:
                base_rec += f" ({trend_dir.title()} Pattern)"
        
        return base_rec
            
    except:
        return "➡️ Advanced Analysis Available"

def assess_risk_level(metrics):
    """Assess risk level based on key metrics"""
    try:
        rsi_text = metrics.get('RSI', '50')
        if rsi_text != 'N/A':
            rsi = float(rsi_text)
        else:
            rsi = 50
            
        beta_text = metrics.get('Beta', '1.0')
        if beta_text != 'N/A':
            beta = float(beta_text)
        else:
            beta = 1.0
            
        # Risk assessment
        if beta > 1.5 or rsi > 75 or rsi < 25:
            return "🔴 High Risk"
        elif beta > 1.2 or rsi > 65 or rsi < 35:
            return "🟡 Medium Risk"
        else:
            return "🟢 Lower Risk"
            
    except:
        return "🟡 Risk Assessment Available"

def create_share_urls(insight):
    """Create shareable URLs for different platforms"""
    
    # Create a summary text
    symbol = insight.get('symbol', 'Stock')
    recommendation = insight.get('recommendation', 'Analysis available')
    
    if insight['privacy_level'] == 'public':
        fib_info = ""
        if 'fibonacci_analysis' in insight:
            fib_data = insight['fibonacci_analysis']
            fib_info = f" Trend: {fib_data['trend']}. Key Level: {fib_data['key_level']}."
        share_text = f"📊 {symbol} Analysis: {recommendation}. Current: {insight.get('current_price', 'N/A')} ({insight.get('daily_change', 'N/A')}). Risk: {insight.get('risk_level', 'N/A')}.{fib_info}"
    elif insight['privacy_level'] == 'anonymized':
        fib_info = ""
        if 'fibonacci_analysis' in insight:
            fib_data = insight['fibonacci_analysis']
            fib_info = f" Trend: {fib_data['trend']}."
        share_text = f"📊 Stock Analysis: {recommendation}. Daily change: {insight.get('daily_change', 'N/A')}.{fib_info} Technical indicators suggest monitoring for opportunities."
    else:
        share_text = f"📊 Completed advanced technical analysis with Fibonacci levels: {recommendation}"
    
    hashtags = "#StockAnalysis #TechnicalAnalysis #Investing"
    
    urls = {
        "twitter": f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}&hashtags={urllib.parse.quote('StockAnalysis,TechnicalAnalysis')}",
        "linkedin": f"https://www.linkedin.com/sharing/share-offsite/?url={urllib.parse.quote('https://example.com')}&summary={urllib.parse.quote(share_text)}",
        "copy_text": f"{share_text} {hashtags}",
        "email_subject": f"Stock Analysis Insight - {symbol}",
        "email_body": f"Hi,\n\nI wanted to share this investment insight:\n\n{share_text}\n\nGenerated using technical analysis tools.\n\nBest regards"
    }
    
    return urls

def calculate_ctp_levels(current_price):
    """
    Calculate Safe Level High and Low reference levels
    
    Args:
        current_price (float): Current stock price
    
    Returns:
        dict: Dictionary with upper and lower safe levels
    """
    try:
        upper_ctp = current_price * 1.125  # +12.5%
        lower_ctp = current_price * 0.875  # -12.5%
        
        return {
            'upper_ctp': upper_ctp,
            'lower_ctp': lower_ctp,
            'upper_percentage': '+12.5%',
            'lower_percentage': '-12.5%'
        }
    except:
        return {
            'upper_ctp': None,
            'lower_ctp': None,
            'upper_percentage': 'N/A',
            'lower_percentage': 'N/A'
        }

def get_after_market_data(symbol, market="US"):
    """
    Get after-market trading data for a stock
    
    Args:
        symbol (str): Stock ticker symbol
        market (str): Market type ("US" or "India")
    
    Returns:
        dict: After-market trading information
    """
    try:
        # Format symbol for Indian stocks
        if market == "India":
            if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
                symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        
        # Get today's data including pre/post market
        today_data = ticker.history(period="1d", interval="1m")
        
        if today_data.empty:
            return {
                'pre_market_change': 'N/A',
                'pre_market_change_percent': 'N/A',
                'post_market_change': 'N/A', 
                'post_market_change_percent': 'N/A',
                'regular_session_close': 'N/A'
            }
        
        # Get regular session close (4 PM ET for US markets)
        regular_close = today_data['Close'].iloc[-1]
        
        # Try to get real-time quote data
        try:
            info = ticker.info
            current_price = info.get('currentPrice', regular_close)
            pre_market_price = info.get('preMarketPrice', None)
            post_market_price = info.get('postMarketPrice', None)
            
            # Calculate pre-market movement
            pre_market_change = 'N/A'
            pre_market_change_percent = 'N/A'
            if pre_market_price and pre_market_price != regular_close:
                prev_close = info.get('previousClose', regular_close)
                if prev_close:
                    pre_change = pre_market_price - prev_close
                    pre_change_percent = (pre_change / prev_close) * 100
                    pre_market_change = f"${pre_change:+.2f}" if market == "US" else f"₹{pre_change:+.2f}"
                    pre_market_change_percent = f"{pre_change_percent:+.2f}%"
            
            # Calculate post-market movement
            post_market_change = 'N/A'
            post_market_change_percent = 'N/A'
            if post_market_price and post_market_price != regular_close:
                post_change = post_market_price - regular_close
                post_change_percent = (post_change / regular_close) * 100
                post_market_change = f"${post_change:+.2f}" if market == "US" else f"₹{post_change:+.2f}"
                post_market_change_percent = f"{post_change_percent:+.2f}%"
            
            return {
                'pre_market_change': pre_market_change,
                'pre_market_change_percent': pre_market_change_percent,
                'post_market_change': post_market_change,
                'post_market_change_percent': post_market_change_percent,
                'regular_session_close': f"${regular_close:.2f}" if market == "US" else f"₹{regular_close:.2f}",
                'current_price': f"${current_price:.2f}" if market == "US" else f"₹{current_price:.2f}"
            }
            
        except Exception as e:
            # Fallback to basic calculation
            return {
                'pre_market_change': 'N/A',
                'pre_market_change_percent': 'N/A', 
                'post_market_change': 'N/A',
                'post_market_change_percent': 'N/A',
                'regular_session_close': f"${regular_close:.2f}" if market == "US" else f"₹{regular_close:.2f}",
                'current_price': f"${regular_close:.2f}" if market == "US" else f"₹{regular_close:.2f}"
            }
            
    except Exception as e:
        return {
            'pre_market_change': 'N/A',
            'pre_market_change_percent': 'N/A',
            'post_market_change': 'N/A', 
            'post_market_change_percent': 'N/A',
            'regular_session_close': 'N/A',
            'current_price': 'N/A'
        }

def calculate_moving_average(data, window=50):
    """
    Calculate moving average for the given data
    
    Args:
        data (pd.DataFrame): Stock price data
        window (int): Moving average window size
    
    Returns:
        pd.Series: Moving average values
    """
    return data['Close'].rolling(window=window).mean()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator
    
    Args:
        data (pd.DataFrame): Stock price data
        fast_period (int): Fast EMA period (default 12)
        slow_period (int): Slow EMA period (default 26)
        signal_period (int): Signal EMA period (default 9)
    
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    # Calculate exponential moving averages
    ema_fast = data['Close'].ewm(span=fast_period).mean()
    ema_slow = data['Close'].ewm(span=slow_period).mean()
    
    # MACD line = Fast EMA - Slow EMA
    macd_line = ema_fast - ema_slow
    
    # Signal line = EMA of MACD line
    signal_line = macd_line.ewm(span=signal_period).mean()
    
    # Histogram = MACD line - Signal line
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_rsi(data, period=14):
    """
    Calculate RSI (Relative Strength Index) indicator
    
    Args:
        data (pd.DataFrame): Stock price data
        period (int): RSI period (default 14)
    
    Returns:
        pd.Series: RSI values
    """
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Calculate relative strength
    rs = gain / loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_chaikin_money_flow(data, period=20):
    """
    Calculate Chaikin Money Flow (CMF) indicator
    
    Args:
        data (pd.DataFrame): Stock price data with OHLCV
        period (int): CMF period (default 20)
    
    Returns:
        pd.Series: CMF values
    """
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    
    # Handle division by zero (when High = Low)
    mf_multiplier = mf_multiplier.fillna(0)
    
    # Money Flow Volume = Money Flow Multiplier * Volume
    mf_volume = mf_multiplier * data['Volume']
    
    # Chaikin Money Flow = Sum of Money Flow Volume over period / Sum of Volume over period
    cmf = mf_volume.rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()
    
    return cmf

def calculate_support_resistance(data, window=20):
    """
    Calculate potential support and resistance levels
    
    Args:
        data (pd.DataFrame): Stock price data
        window (int): Rolling window for calculations
    
    Returns:
        tuple: (support_level, resistance_level)
    """
    # Calculate rolling lows and highs
    rolling_lows = data['Low'].rolling(window=window).min()
    rolling_highs = data['High'].rolling(window=window).max()
    
    # Get recent support (lowest low) and resistance (highest high)
    recent_support = rolling_lows.iloc[-window:].min()
    recent_resistance = rolling_highs.iloc[-window:].max()
    
    return recent_support, recent_resistance

def calculate_fibonacci_retracement(data, lookback_period=50):
    """
    Calculate Fibonacci retracement levels
    
    Args:
        data (pd.DataFrame): Stock price data
        lookback_period (int): Period to look back for swing high/low
    
    Returns:
        list: List of tuples (level, price, label) for Fibonacci levels
    """
    try:
        # Get recent period data
        recent_data = data.tail(lookback_period)
        
        # Find swing high and low
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        
        # Calculate the range
        price_range = swing_high - swing_low
        
        # Fibonacci levels (as decimal percentages)
        fib_levels = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
        fib_labels = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]
        
        # Calculate retracement levels
        fibonacci_data = []
        
        for level, label in zip(fib_levels, fib_labels):
            # For uptrend: Start from high and retrace down
            # For downtrend: Start from low and extend up
            current_price = data['Close'].iloc[-1]
            
            if current_price > (swing_high + swing_low) / 2:
                # Likely uptrend - calculate retracement from high
                fib_price = swing_high - (price_range * level)
            else:
                # Likely downtrend - calculate extension from low
                fib_price = swing_low + (price_range * level)
            
            fibonacci_data.append((level, fib_price, f"Fib {label}"))
        
        return fibonacci_data
        
    except Exception as e:
        print(f"Error calculating Fibonacci retracement: {str(e)}")
        return None



def get_earnings_info(ticker_obj, ticker_info):
    """
    Extract earnings information from ticker object and info with improved accuracy and multiple fallback sources
    
    Args:
        ticker_obj: yfinance Ticker object
        ticker_info (dict): Company information from yfinance
    
    Returns:
        dict: Earnings information
    """
    earnings_info = {
        'last_earnings': None,
        'next_earnings': None,
        'last_earnings_formatted': 'N/A',
        'next_earnings_formatted': 'N/A'
    }
    
    current_date = pd.Timestamp.now()
    print(f"Gathering earnings info...")
    
    try:
        # Method 1: Try earnings_dates (most comprehensive source)
        try:
            earnings_dates = ticker_obj.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                print(f"Found {len(earnings_dates)} earnings dates from earnings_dates")
                
                # Convert current_date to match earnings_dates timezone
                if hasattr(earnings_dates.index[0], 'tz') and earnings_dates.index[0].tz:
                    current_date_tz = current_date.tz_localize(earnings_dates.index[0].tz)
                else:
                    current_date_tz = current_date
                
                # Look for the most recent past earnings (prioritize 2025 data)
                # First try to find earnings from current year (2025)
                current_year = current_date_tz.year
                current_year_earnings = earnings_dates[
                    (earnings_dates.index <= current_date_tz) & 
                    (earnings_dates.index.year == current_year)
                ]
                
                if not current_year_earnings.empty:
                    # Sort by date to ensure we get the most recent (last in chronological order)
                    current_year_earnings_sorted = current_year_earnings.sort_index()
                    last_earnings = current_year_earnings_sorted.index[-1]
                    earnings_info['last_earnings'] = last_earnings
                    earnings_info['last_earnings_formatted'] = last_earnings.strftime('%Y-%m-%d')
                    print(f"Found last earnings (current year): {earnings_info['last_earnings_formatted']} from {len(current_year_earnings)} 2025 earnings dates")
                else:
                    # Fallback to most recent within 12 months (more aggressive cutoff)
                    cutoff_date_tz = current_date_tz - pd.Timedelta(days=365)  # 12 months
                    recent_past_earnings = earnings_dates[
                        (earnings_dates.index <= current_date_tz) & 
                        (earnings_dates.index >= cutoff_date_tz)
                    ]
                    if not recent_past_earnings.empty:
                        last_earnings = recent_past_earnings.index[-1]
                        earnings_info['last_earnings'] = last_earnings
                        earnings_info['last_earnings_formatted'] = last_earnings.strftime('%Y-%m-%d')
                        print(f"Found last earnings (within 12 months): {earnings_info['last_earnings_formatted']}")
                    else:
                        # Check if we have any past earnings and show as old
                        all_past_earnings = earnings_dates[earnings_dates.index <= current_date_tz]
                        if not all_past_earnings.empty:
                            old_earnings = all_past_earnings.index[-1]
                            print(f"Found old earnings (skipped): {old_earnings.strftime('%Y-%m-%d')}")
                
                # Get next upcoming earnings
                future_earnings = earnings_dates[earnings_dates.index > current_date_tz]
                if not future_earnings.empty:
                    next_earnings = future_earnings.index[0]
                    earnings_info['next_earnings'] = next_earnings
                    earnings_info['next_earnings_formatted'] = next_earnings.strftime('%Y-%m-%d')
                    print(f"Found next earnings: {earnings_info['next_earnings_formatted']}")
        except Exception as e:
            print(f"Earnings dates error: {e}")
        
        # Method 2: Try earnings calendar for next earnings
        if earnings_info['next_earnings'] is None:
            try:
                calendar = ticker_obj.calendar
                if calendar is not None:
                    # Handle both DataFrame and dict calendar formats
                    if hasattr(calendar, 'empty') and not calendar.empty:
                        next_earnings = calendar.index[0]
                        earnings_info['next_earnings'] = next_earnings
                        earnings_info['next_earnings_formatted'] = next_earnings.strftime('%Y-%m-%d')
                        print(f"Found next earnings from calendar: {earnings_info['next_earnings_formatted']}")
                    elif isinstance(calendar, dict) and calendar:
                        # Handle dict format calendar
                        for key, dates in calendar.items():
                            if dates:
                                # Handle single date or list of dates
                                if hasattr(dates, '__len__') and not isinstance(dates, str):
                                    # It's a list or array
                                    if len(dates) > 0:
                                        next_earnings = pd.to_datetime(dates[0])
                                        earnings_info['next_earnings'] = next_earnings
                                        earnings_info['next_earnings_formatted'] = next_earnings.strftime('%Y-%m-%d')
                                        print(f"Found next earnings from calendar dict list: {earnings_info['next_earnings_formatted']}")
                                        break
                                else:
                                    # It's a single date object
                                    next_earnings = pd.to_datetime(dates)
                                    earnings_info['next_earnings'] = next_earnings
                                    earnings_info['next_earnings_formatted'] = next_earnings.strftime('%Y-%m-%d')
                                    print(f"Found next earnings from calendar dict single: {earnings_info['next_earnings_formatted']}")
                                    break
            except Exception as e:
                print(f"Calendar error: {e}")
        
        # Method 3: Try earnings history for past earnings (with date validation)
        if earnings_info['last_earnings'] is None:
            try:
                earnings_history = ticker_obj.earnings
                if earnings_history is not None and not earnings_history.empty:
                    # Convert earnings history index to datetime and filter for recent dates
                    last_earnings_date = pd.to_datetime(earnings_history.index[-1])
                    
                    # Only use if it's within the last 2 years (more reasonable for earnings)
                    cutoff_date = current_date - pd.Timedelta(days=730)  # 2 years
                    if last_earnings_date >= cutoff_date:
                        earnings_info['last_earnings'] = last_earnings_date
                        earnings_info['last_earnings_formatted'] = last_earnings_date.strftime('%Y-%m-%d')
                        print(f"Found last earnings from history: {earnings_info['last_earnings_formatted']}")
                    else:
                        print(f"Earnings history date too old: {last_earnings_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"Earnings history error: {e}")
        
        # Method 4: Fallback to ticker_info fields
        if earnings_info['next_earnings'] is None:
            try:
                if 'earningsDate' in ticker_info and ticker_info['earningsDate']:
                    earnings_date = ticker_info['earningsDate']
                    if isinstance(earnings_date, list) and earnings_date:
                        next_earnings = pd.to_datetime(earnings_date[0], unit='s')
                    else:
                        next_earnings = pd.to_datetime(earnings_date, unit='s')
                    
                    earnings_info['next_earnings'] = next_earnings
                    earnings_info['next_earnings_formatted'] = next_earnings.strftime('%Y-%m-%d')
                    print(f"Found next earnings from ticker info: {earnings_info['next_earnings_formatted']}")
            except Exception as e:
                print(f"Ticker info earnings date error: {e}")
        
        # Method 5: Try additional ticker_info fields for last earnings (with date validation)
        if earnings_info['last_earnings'] is None:
            try:
                # Try multiple possible fields but validate dates
                for field in ['lastFiscalYearEnd', 'mostRecentQuarter', 'lastQuarterEnd']:
                    if field in ticker_info and ticker_info[field]:
                        try:
                            last_earnings = pd.to_datetime(ticker_info[field], unit='s')
                            
                            # Only use if it's within the last 18 months (more reasonable for recent earnings)
                            cutoff_date = current_date - pd.Timedelta(days=540)  # 18 months
                            if last_earnings >= cutoff_date:
                                earnings_info['last_earnings'] = last_earnings
                                earnings_info['last_earnings_formatted'] = last_earnings.strftime('%Y-%m-%d')
                                print(f"Found last earnings from {field}: {earnings_info['last_earnings_formatted']}")
                                break
                            else:
                                print(f"Field {field} date too old: {last_earnings.strftime('%Y-%m-%d')}")
                        except:
                            continue
            except Exception as e:
                print(f"Ticker info fallback error: {e}")
        
        # Method 6: Estimate next earnings based on last earnings (quarterly companies)
        if earnings_info['next_earnings'] is None and earnings_info['last_earnings'] is not None:
            try:
                last_earnings_date = earnings_info['last_earnings']
                print(f"Attempting to estimate next earnings from last: {last_earnings_date}")
                
                # Try different estimation approaches
                for days_offset in [90, 91, 92, 89, 88]:  # Try variations around 90 days (quarterly)
                    try:
                        if hasattr(last_earnings_date, 'tz') and last_earnings_date.tz:
                            estimated_next = last_earnings_date + pd.DateOffset(days=days_offset)
                        else:
                            estimated_next = pd.to_datetime(last_earnings_date) + pd.DateOffset(days=days_offset)
                        
                        # Check if estimated date is in the future
                        if estimated_next > pd.Timestamp.now():
                            earnings_info['next_earnings'] = estimated_next
                            earnings_info['next_earnings_formatted'] = f"~{estimated_next.strftime('%Y-%m-%d')}"
                            print(f"Estimated next earnings (+{days_offset} days): {earnings_info['next_earnings_formatted']}")
                            break
                    except Exception as inner_e:
                        continue
                
                # If quarterly estimation doesn't work, try yearly estimation
                if earnings_info['next_earnings'] is None:
                    try:
                        estimated_next = pd.to_datetime(last_earnings_date) + pd.DateOffset(years=1)
                        if estimated_next > pd.Timestamp.now():
                            earnings_info['next_earnings'] = estimated_next
                            earnings_info['next_earnings_formatted'] = f"~{estimated_next.strftime('%Y-%m-%d')}"
                            print(f"Estimated next earnings (yearly): {earnings_info['next_earnings_formatted']}")
                    except:
                        pass
                        
            except Exception as e:
                print(f"Estimation error: {e}")
                
    except Exception as e:
        print(f"General earnings info error: {e}")
    
    print(f"Final earnings info - Last: {earnings_info['last_earnings_formatted']}, Next: {earnings_info['next_earnings_formatted']}")
    return earnings_info

def get_earnings_performance_analysis(ticker_obj, data, market="US"):
    """
    Analyze stock performance after earnings for available quarters (up to 4)
    
    Args:
        ticker_obj: yfinance Ticker object
        data (pd.DataFrame): Stock price data
        market (str): Market type (US/IN)
    
    Returns:
        tuple: (pd.DataFrame: Earnings performance analysis table, int: number of quarters found)
    """
    try:
        # Get earnings history - try multiple methods
        earnings = None
        earnings_count = 0
        
        # Method 1: Try earnings_dates
        try:
            earnings = ticker_obj.earnings_dates
            if earnings is not None and not earnings.empty:
                earnings_count = len(earnings)
        except Exception as e:
            print(f"Earnings dates error: {e}")
            pass
        
        # Method 2: Try calendar if earnings_dates failed  
        if earnings is None or earnings.empty:
            try:
                calendar = ticker_obj.calendar
                if calendar is not None:
                    # Calendar might be a dict, not DataFrame
                    if hasattr(calendar, 'empty') and not calendar.empty:
                        earnings = calendar
                        earnings_count = len(calendar)
                    elif isinstance(calendar, dict) and calendar:
                        # Convert dict to DataFrame if needed
                        earnings = pd.DataFrame(calendar)
                        earnings_count = len(earnings)
            except Exception as e:
                print(f"Calendar error: {e}")
                pass
        
        # Method 3: Try earnings history
        if earnings is None or earnings.empty:
            try:
                earnings_history = ticker_obj.earnings
                if earnings_history is not None and not earnings_history.empty:
                    # Convert earnings history to datetime index
                    earnings = earnings_history
                    earnings_count = len(earnings_history)
            except:
                pass
        
        if earnings is None or earnings.empty:
            return None, 0
        
        # Get available earnings dates and filter for unique quarters (up to 4)
        earnings_dates = earnings.index.tolist()
        earnings_dates.sort(reverse=True)  # Most recent first
        
        # Filter for unique quarters to avoid duplicate earnings in same quarter
        unique_quarters = []
        seen_quarters = set()
        
        for date in earnings_dates:
            quarter_key = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
            if quarter_key not in seen_quarters:
                unique_quarters.append(date)
                seen_quarters.add(quarter_key)
                if len(unique_quarters) >= 4:
                    break
        
        available_earnings = unique_quarters
        print(f"Filtered to {len(available_earnings)} unique quarters from {len(earnings_dates)} total earnings dates")
        
        analysis_data = []
        successful_analyses = 0
        
        for earnings_date in available_earnings:
            try:
                print(f"Processing earnings date: {earnings_date}")
                
                # Ensure both earnings_date and data.index have the same timezone handling
                # Convert earnings_date to match data.index timezone
                if hasattr(data.index[0], 'tz') and data.index[0].tz:
                    # Data has timezone, make sure earnings_date has it too
                    if hasattr(earnings_date, 'tz') and earnings_date.tz:
                        earnings_date_for_comparison = earnings_date
                    else:
                        # Convert naive datetime to data's timezone
                        earnings_date_for_comparison = earnings_date.tz_localize(data.index[0].tz)
                else:
                    # Data is timezone-naive, make earnings_date naive too
                    if hasattr(earnings_date, 'tz') and earnings_date.tz:
                        earnings_date_for_comparison = earnings_date.tz_localize(None)
                    else:
                        earnings_date_for_comparison = earnings_date
                
                # Find the trading day before earnings (baseline price)
                pre_earnings_mask = data.index < earnings_date_for_comparison
                if not pre_earnings_mask.any():
                    print(f"No pre-earnings data found for {earnings_date}")
                    continue
                    
                pre_earnings_price = data[pre_earnings_mask]['Close'].iloc[-1]
                pre_earnings_date = data[pre_earnings_mask].index[-1]
                
                # Find the first trading day after earnings
                post_earnings_mask = data.index > earnings_date_for_comparison
                if not post_earnings_mask.any():
                    print(f"No post-earnings data found for {earnings_date}")
                    continue
                    
                # Opening price the day after earnings
                post_earnings_open = data[post_earnings_mask]['Open'].iloc[0]
                post_earnings_date = data[post_earnings_mask].index[0]
                
                # Calculate overnight change (from previous close to next open)
                overnight_change = ((post_earnings_open - pre_earnings_price) / pre_earnings_price) * 100
                
                # Find end of week price (5 trading days after earnings, or last available)
                week_end_mask = data.index >= post_earnings_date
                week_data = data[week_end_mask].head(5)  # Get up to 5 trading days
                
                if not week_data.empty:
                    week_end_price = week_data['Close'].iloc[-1]
                    week_end_date = week_data.index[-1]
                    
                    # Calculate week performance (from pre-earnings close to end of week)
                    week_performance = ((week_end_price - pre_earnings_price) / pre_earnings_price) * 100
                else:
                    week_end_price = post_earnings_open
                    week_end_date = post_earnings_date
                    week_performance = overnight_change
                
                # Determine quarter
                quarter = f"Q{((earnings_date.month - 1) // 3) + 1} {earnings_date.year}"
                
                print(f"Analysis successful for {earnings_date}: {overnight_change:+.2f}% overnight, {week_performance:+.2f}% week")
                
                analysis_data.append({
                    'Quarter': quarter,
                    'Earnings Date': earnings_date.strftime('%Y-%m-%d'),
                    'Pre-Earnings Close': format_currency(pre_earnings_price, market),
                    'Next Day Open': format_currency(post_earnings_open, market),
                    'Overnight Change (%)': f"{overnight_change:+.2f}%",
                    'End of Week Close': format_currency(week_end_price, market),
                    'Week Performance (%)': f"{week_performance:+.2f}%",
                    'Direction': '📈 Up' if week_performance > 0 else '📉 Down' if week_performance < 0 else '➡️ Flat'
                })
                successful_analyses += 1
                
            except Exception as e:
                print(f"Error processing {earnings_date}: {e}")
                continue
        
        if analysis_data:
            df = pd.DataFrame(analysis_data)
            return df, successful_analyses
        else:
            return None, 0
            
    except Exception as e:
        return None, 0

def get_detailed_earnings_performance_analysis(ticker_obj, data, market="US", max_quarters=8):
    """
    Enhanced earnings performance analysis with extended historical data
    
    Args:
        ticker_obj: yfinance Ticker object
        data (pd.DataFrame): Stock price data
        market (str): Market type (US/IN)
        max_quarters (int): Maximum number of quarters to analyze
    
    Returns:
        tuple: (pd.DataFrame: Detailed earnings performance analysis, int: number of quarters found)
    """
    try:
        # Get comprehensive earnings data
        earnings = None
        
        # Primary method: earnings_dates
        try:
            earnings = ticker_obj.earnings_dates
            if earnings is not None and not earnings.empty:
                print(f"Found {len(earnings)} earnings records from earnings_dates")
        except Exception as e:
            print(f"Earnings dates error: {e}")
        
        if earnings is None or earnings.empty:
            print("No earnings data found")
            return None, 0
        
        # Get available earnings dates and filter for unique quarters
        earnings_dates = earnings.index.tolist()
        earnings_dates.sort(reverse=True)  # Most recent first
        
        # Filter for unique quarters to avoid duplicate earnings in same quarter
        unique_quarters = []
        seen_quarters = set()
        
        for date in earnings_dates:
            quarter_key = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
            if quarter_key not in seen_quarters:
                unique_quarters.append(date)
                seen_quarters.add(quarter_key)
                if len(unique_quarters) >= max_quarters:
                    break
        
        available_earnings = unique_quarters
        print(f"Filtered to {len(available_earnings)} unique quarters from {len(earnings_dates)} total earnings dates")
        
        analysis_data = []
        successful_analyses = 0
        
        print(f"Analyzing {len(available_earnings)} unique quarterly earnings dates...")
        
        # Debug: Show which quarters we're analyzing
        for date in available_earnings:
            quarter_key = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
            print(f"Selected {quarter_key}: {date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        for earnings_date in available_earnings:
            try:
                print(f"Processing earnings date: {earnings_date}")
                
                # Handle timezone compatibility
                if hasattr(data.index[0], 'tz') and data.index[0].tz:
                    if hasattr(earnings_date, 'tz') and earnings_date.tz:
                        earnings_date_for_comparison = earnings_date
                    else:
                        earnings_date_for_comparison = earnings_date.tz_localize(data.index[0].tz)
                else:
                    if hasattr(earnings_date, 'tz') and earnings_date.tz:
                        earnings_date_for_comparison = earnings_date.tz_localize(None)
                    else:
                        earnings_date_for_comparison = earnings_date
                
                # Find pre-earnings data
                pre_earnings_mask = data.index < earnings_date_for_comparison
                if not pre_earnings_mask.any():
                    print(f"No pre-earnings data for {earnings_date}")
                    continue
                    
                pre_earnings_price = data[pre_earnings_mask]['Close'].iloc[-1]
                pre_earnings_date = data[pre_earnings_mask].index[-1]
                
                # Find post-earnings data
                post_earnings_mask = data.index > earnings_date_for_comparison
                if not post_earnings_mask.any():
                    print(f"No post-earnings data for {earnings_date}")
                    continue
                    
                post_earnings_open = data[post_earnings_mask]['Open'].iloc[0]
                post_earnings_date = data[post_earnings_mask].index[0]
                
                # Calculate overnight change
                overnight_change = ((post_earnings_open - pre_earnings_price) / pre_earnings_price) * 100
                
                # Calculate week performance (up to 5 trading days)
                week_end_mask = data.index >= post_earnings_date
                week_data = data[week_end_mask].head(5)
                
                if not week_data.empty:
                    week_end_price = week_data['Close'].iloc[-1]
                    week_performance = ((week_end_price - pre_earnings_price) / pre_earnings_price) * 100
                else:
                    week_end_price = post_earnings_open
                    week_performance = overnight_change
                
                # Get EPS data if available
                eps_estimate = None
                eps_actual = None
                surprise_pct = None
                
                if earnings_date in earnings.index:
                    row = earnings.loc[earnings_date]
                    if hasattr(row, 'get'):
                        eps_estimate = row.get('EPS Estimate', None)
                        eps_actual = row.get('Reported EPS', None) 
                        surprise_pct = row.get('Surprise(%)', None)
                
                # Determine quarter
                quarter_num = (earnings_date.month - 1) // 3 + 1
                quarter = f"Q{quarter_num} {earnings_date.year}"
                
                # Create detailed analysis row
                analysis_row = {
                    'Quarter': quarter,
                    'Earnings Date': earnings_date.strftime('%Y-%m-%d'),
                    'Pre-Earnings Close': format_currency(pre_earnings_price, market),
                    'Next Day Open': format_currency(post_earnings_open, market),
                    'Overnight Change (%)': f"{overnight_change:+.2f}%",
                    'End of Week Close': format_currency(week_end_price, market),
                    'Week Performance (%)': f"{week_performance:+.2f}%",
                    'Direction': '📈 Up' if week_performance > 0 else '📉 Down' if week_performance < 0 else '➡️ Flat'
                }
                
                # Add EPS information if available
                if eps_estimate and not pd.isna(eps_estimate):
                    analysis_row['EPS Est'] = f"${eps_estimate:.2f}"
                if eps_actual and not pd.isna(eps_actual):
                    analysis_row['EPS Act'] = f"${eps_actual:.2f}"
                if surprise_pct and not pd.isna(surprise_pct):
                    analysis_row['Surprise'] = f"{surprise_pct:+.1f}%"
                
                analysis_data.append(analysis_row)
                successful_analyses += 1
                
                print(f"✅ Analysis successful: {overnight_change:+.2f}% overnight, {week_performance:+.2f}% week")
                
            except Exception as e:
                print(f"❌ Error processing {earnings_date}: {e}")
                continue
        
        if analysis_data:
            df = pd.DataFrame(analysis_data)
            print(f"Created analysis DataFrame with {len(df)} rows")
            return df, successful_analyses
        else:
            print("No successful analyses")
            return None, 0
            
    except Exception as e:
        print(f"Overall analysis error: {e}")
        return None, 0

def display_institutional_financial_metrics(info, ticker_obj, symbol):
    """
    Display comprehensive institutional-grade financial metrics
    
    Args:
        info (dict): Stock information from yfinance
        ticker_obj: yfinance Ticker object
        symbol (str): Stock symbol
    """
    try:
        st.success(f"✅ Financial metrics loaded for {symbol}")
        
        # Create tabs for different metric categories
        metrics_tab1, metrics_tab2, metrics_tab3, metrics_tab4 = st.tabs([
            "📈 Valuation Metrics", 
            "💰 Profitability", 
            "🏛️ Financial Strength", 
            "🚀 Growth Metrics"
        ])
        
        with metrics_tab1:
            display_valuation_metrics(info)
        
        with metrics_tab2:
            display_profitability_metrics(info)
        
        with metrics_tab3:
            display_financial_strength_metrics(info, ticker_obj)
        
        with metrics_tab4:
            display_growth_metrics(info, ticker_obj)
            
    except Exception as e:
        st.error(f"Error displaying financial metrics: {e}")

def display_valuation_metrics(info):
    """Display comprehensive valuation metrics"""
    st.markdown("### Valuation Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Price-to-Earnings metrics
    with col1:
        pe_ratio = info.get('trailingPE', None)
        forward_pe = info.get('forwardPE', None)
        
        st.metric(
            label="P/E Ratio (TTM)",
            value=f"{pe_ratio:.2f}" if pe_ratio and not pd.isna(pe_ratio) else "N/A",
            help="Price-to-Earnings ratio based on trailing twelve months"
        )
        
        if forward_pe and not pd.isna(forward_pe):
            st.metric(
                label="Forward P/E",
                value=f"{forward_pe:.2f}",
                help="Forward Price-to-Earnings ratio"
            )
    
    with col2:
        # Price-to-Book and Price-to-Sales
        pb_ratio = info.get('priceToBook', None)
        ps_ratio = info.get('priceToSalesTrailing12Months', None)
        
        st.metric(
            label="P/B Ratio",
            value=f"{pb_ratio:.2f}" if pb_ratio and not pd.isna(pb_ratio) else "N/A",
            help="Price-to-Book ratio"
        )
        
        st.metric(
            label="P/S Ratio (TTM)",
            value=f"{ps_ratio:.2f}" if ps_ratio and not pd.isna(ps_ratio) else "N/A",
            help="Price-to-Sales ratio trailing twelve months"
        )
    
    with col3:
        # Enterprise Value metrics
        ev_revenue = info.get('enterpriseToRevenue', None)
        ev_ebitda = info.get('enterpriseToEbitda', None)
        
        st.metric(
            label="EV/Revenue",
            value=f"{ev_revenue:.2f}" if ev_revenue and not pd.isna(ev_revenue) else "N/A",
            help="Enterprise Value to Revenue ratio"
        )
        
        st.metric(
            label="EV/EBITDA",
            value=f"{ev_ebitda:.2f}" if ev_ebitda and not pd.isna(ev_ebitda) else "N/A",
            help="Enterprise Value to EBITDA ratio"
        )
    
    with col4:
        # PEG and Market Cap
        peg_ratio = info.get('pegRatio', None)
        market_cap = info.get('marketCap', None)
        
        st.metric(
            label="PEG Ratio",
            value=f"{peg_ratio:.2f}" if peg_ratio and not pd.isna(peg_ratio) else "N/A",
            help="Price/Earnings to Growth ratio"
        )
        
        if market_cap and not pd.isna(market_cap):
            if market_cap >= 1e12:
                cap_display = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                cap_display = f"${market_cap/1e9:.2f}B"
            else:
                cap_display = f"${market_cap/1e6:.2f}M"
            
            st.metric(
                label="Market Cap",
                value=cap_display,
                help="Total market capitalization"
            )

def display_profitability_metrics(info):
    """Display comprehensive profitability metrics"""
    st.markdown("### Profitability Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Margin metrics
        gross_margin = info.get('grossMargins', None)
        operating_margin = info.get('operatingMargins', None)
        
        st.metric(
            label="Gross Margin",
            value=f"{gross_margin*100:.2f}%" if gross_margin and not pd.isna(gross_margin) else "N/A",
            help="Gross profit margin percentage"
        )
        
        st.metric(
            label="Operating Margin",
            value=f"{operating_margin*100:.2f}%" if operating_margin and not pd.isna(operating_margin) else "N/A",
            help="Operating profit margin percentage"
        )
    
    with col2:
        # Profit margins
        profit_margin = info.get('profitMargins', None)
        ebitda_margin = info.get('ebitdaMargins', None)
        
        st.metric(
            label="Net Profit Margin",
            value=f"{profit_margin*100:.2f}%" if profit_margin and not pd.isna(profit_margin) else "N/A",
            help="Net profit margin percentage"
        )
        
        st.metric(
            label="EBITDA Margin",
            value=f"{ebitda_margin*100:.2f}%" if ebitda_margin and not pd.isna(ebitda_margin) else "N/A",
            help="EBITDA margin percentage"
        )
    
    with col3:
        # Return metrics
        roe = info.get('returnOnEquity', None)
        roa = info.get('returnOnAssets', None)
        
        st.metric(
            label="ROE",
            value=f"{roe*100:.2f}%" if roe and not pd.isna(roe) else "N/A",
            help="Return on Equity"
        )
        
        st.metric(
            label="ROA",
            value=f"{roa*100:.2f}%" if roa and not pd.isna(roa) else "N/A",
            help="Return on Assets"
        )
    
    with col4:
        # Earnings metrics
        eps = info.get('trailingEps', None)
        forward_eps = info.get('forwardEps', None)
        
        st.metric(
            label="EPS (TTM)",
            value=f"${eps:.2f}" if eps and not pd.isna(eps) else "N/A",
            help="Earnings per Share trailing twelve months"
        )
        
        st.metric(
            label="Forward EPS",
            value=f"${forward_eps:.2f}" if forward_eps and not pd.isna(forward_eps) else "N/A",
            help="Forward Earnings per Share"
        )

def display_financial_strength_metrics(info, ticker_obj):
    """Display comprehensive financial strength metrics"""
    st.markdown("### Financial Strength Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Debt metrics
        debt_to_equity = info.get('debtToEquity', None)
        total_debt = info.get('totalDebt', None)
        
        st.metric(
            label="Debt-to-Equity",
            value=f"{debt_to_equity:.2f}" if debt_to_equity and not pd.isna(debt_to_equity) else "N/A",
            help="Total debt to equity ratio"
        )
        
        if total_debt and not pd.isna(total_debt):
            if total_debt >= 1e9:
                debt_display = f"${total_debt/1e9:.2f}B"
            else:
                debt_display = f"${total_debt/1e6:.2f}M"
            
            st.metric(
                label="Total Debt",
                value=debt_display,
                help="Total debt amount"
            )
    
    with col2:
        # Liquidity metrics
        current_ratio = info.get('currentRatio', None)
        quick_ratio = info.get('quickRatio', None)
        
        st.metric(
            label="Current Ratio",
            value=f"{current_ratio:.2f}" if current_ratio and not pd.isna(current_ratio) else "N/A",
            help="Current assets to current liabilities"
        )
        
        st.metric(
            label="Quick Ratio",
            value=f"{quick_ratio:.2f}" if quick_ratio and not pd.isna(quick_ratio) else "N/A",
            help="Quick assets to current liabilities"
        )
    
    with col3:
        # Cash metrics
        total_cash = info.get('totalCash', None)
        cash_per_share = info.get('totalCashPerShare', None)
        
        if total_cash and not pd.isna(total_cash):
            if total_cash >= 1e9:
                cash_display = f"${total_cash/1e9:.2f}B"
            else:
                cash_display = f"${total_cash/1e6:.2f}M"
            
            st.metric(
                label="Total Cash",
                value=cash_display,
                help="Total cash and cash equivalents"
            )
        
        st.metric(
            label="Cash per Share",
            value=f"${cash_per_share:.2f}" if cash_per_share and not pd.isna(cash_per_share) else "N/A",
            help="Cash per share outstanding"
        )
    
    with col4:
        # Other strength metrics
        book_value = info.get('bookValue', None)
        beta = info.get('beta', None)
        
        st.metric(
            label="Book Value/Share",
            value=f"${book_value:.2f}" if book_value and not pd.isna(book_value) else "N/A",
            help="Book value per share"
        )
        
        st.metric(
            label="Beta",
            value=f"{beta:.2f}" if beta and not pd.isna(beta) else "N/A",
            help="Stock volatility relative to market"
        )

def display_growth_metrics(info, ticker_obj):
    """Display comprehensive growth metrics"""
    st.markdown("### Growth Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Revenue growth
        revenue_growth = info.get('revenueGrowth', None)
        quarterly_revenue_growth = info.get('quarterlyRevenueGrowth', None)
        
        st.metric(
            label="Revenue Growth (YoY)",
            value=f"{revenue_growth*100:.2f}%" if revenue_growth and not pd.isna(revenue_growth) else "N/A",
            help="Year-over-year revenue growth"
        )
        
        st.metric(
            label="Quarterly Revenue Growth",
            value=f"{quarterly_revenue_growth*100:.2f}%" if quarterly_revenue_growth and not pd.isna(quarterly_revenue_growth) else "N/A",
            help="Quarterly revenue growth year-over-year"
        )
    
    with col2:
        # Earnings growth
        earnings_growth = info.get('earningsGrowth', None)
        quarterly_earnings_growth = info.get('quarterlyEarningsGrowth', None)
        
        st.metric(
            label="Earnings Growth (YoY)",
            value=f"{earnings_growth*100:.2f}%" if earnings_growth and not pd.isna(earnings_growth) else "N/A",
            help="Year-over-year earnings growth"
        )
        
        st.metric(
            label="Quarterly Earnings Growth",
            value=f"{quarterly_earnings_growth*100:.2f}%" if quarterly_earnings_growth and not pd.isna(quarterly_earnings_growth) else "N/A",
            help="Quarterly earnings growth year-over-year"
        )
    
    with col3:
        # Target and recommendation
        target_high = info.get('targetHighPrice', None)
        target_mean = info.get('targetMeanPrice', None)
        
        st.metric(
            label="Target High Price",
            value=f"${target_high:.2f}" if target_high and not pd.isna(target_high) else "N/A",
            help="Analyst target high price"
        )
        
        st.metric(
            label="Target Mean Price",
            value=f"${target_mean:.2f}" if target_mean and not pd.isna(target_mean) else "N/A",
            help="Analyst target mean price"
        )
    
    with col4:
        # Analyst metrics
        recommendation = info.get('recommendationMean', None)
        num_analysts = info.get('numberOfAnalystOpinions', None)
        
        if recommendation and not pd.isna(recommendation):
            rec_text = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
            rec_index = min(max(int(recommendation - 1), 0), 4)
            rec_display = rec_text[rec_index]
        else:
            rec_display = "N/A"
        
        st.metric(
            label="Analyst Recommendation",
            value=rec_display,
            help="Average analyst recommendation (1=Strong Buy, 5=Strong Sell)"
        )
        
        st.metric(
            label="Number of Analysts",
            value=f"{int(num_analysts)}" if num_analysts and not pd.isna(num_analysts) else "N/A",
            help="Number of analysts covering the stock"
        )
    
    # Additional growth insights
    st.markdown("---")
    st.markdown("#### Growth Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        # Calculate growth score
        growth_factors = [
            ("Revenue Growth", revenue_growth),
            ("Earnings Growth", earnings_growth),
            ("ROE", info.get('returnOnEquity', None)),
            ("Profit Margin", info.get('profitMargins', None))
        ]
        
        positive_factors = sum(1 for name, value in growth_factors 
                             if value and not pd.isna(value) and value > 0)
        
        st.info(f"""
        **Growth Score: {positive_factors}/4 positive factors**
        
        Positive growth indicators found in {positive_factors} out of 4 key metrics:
        - Revenue Growth, Earnings Growth, ROE, Profit Margin
        """)
    
    with insights_col2:
        # Valuation vs Growth analysis
        pe_ratio = info.get('trailingPE', None)
        peg_ratio = info.get('pegRatio', None)
        
        valuation_insight = "N/A"
        if pe_ratio and peg_ratio and not pd.isna(pe_ratio) and not pd.isna(peg_ratio):
            if peg_ratio < 1:
                valuation_insight = "Potentially undervalued relative to growth"
            elif peg_ratio < 1.5:
                valuation_insight = "Fairly valued relative to growth"
            else:
                valuation_insight = "Premium valuation relative to growth"
        
        peg_display = f"{peg_ratio:.2f}" if peg_ratio and not pd.isna(peg_ratio) else "N/A"
        
        st.info(f"""
        **Valuation vs Growth Assessment:**
        
        {valuation_insight}
        
        PEG Ratio: {peg_display}
        """)

def estimate_next_dividend_date(ticker_obj, dividend_info):
    """
    Estimate next dividend date based on historical dividend patterns
    
    Args:
        ticker_obj: yfinance Ticker object
        dividend_info (dict): Current dividend information
    
    Returns:
        dict: Next dividend date information with estimated flag
    """
    next_div_info = {
        'next_dividend_date': 'N/A',
        'estimated': False
    }
    
    try:
        # Get dividend history
        dividends = ticker_obj.dividends
        if not dividends.empty and len(dividends) >= 2:
            # Get last few dividend dates to determine frequency
            dividend_dates = dividends.index[-4:].tolist()  # Last 4 dividends
            
            if len(dividend_dates) >= 2:
                # Calculate average interval between dividends
                intervals = []
                for i in range(1, len(dividend_dates)):
                    interval = (dividend_dates[i] - dividend_dates[i-1]).days
                    intervals.append(interval)
                
                if intervals:
                    # Use median interval to avoid outliers
                    avg_interval = sorted(intervals)[len(intervals)//2]
                    
                    # Estimate next dividend date
                    last_div_date = dividend_dates[-1]
                    estimated_next = last_div_date + pd.Timedelta(days=avg_interval)
                    
                    # Only show if it's in the future
                    if estimated_next > pd.Timestamp.now():
                        next_div_info['next_dividend_date'] = estimated_next.strftime('%Y-%m-%d')
                        next_div_info['estimated'] = True
                    
    except Exception as e:
        pass
    
    return next_div_info


def get_dividend_info(ticker_obj, ticker_info, market="US"):
    """
    Extract dividend information from ticker object and info
    
    Args:
        ticker_obj: yfinance Ticker object
        ticker_info (dict): Company information from yfinance
        market (str): Market type ("US" or "India")
    
    Returns:
        dict: Dividend information
    """
    dividend_info = {
        'last_dividend_date': None,
        'last_dividend_amount': 0,
        'last_dividend_formatted': 'N/A',
        'dividend_yield': 'N/A',
        'forward_dividend': 'N/A',
        'payout_ratio': 'N/A'
    }
    
    try:
        # Get dividend history
        dividends = ticker_obj.dividends
        if not dividends.empty:
            # Get most recent dividend
            last_dividend_date = dividends.index[-1]
            last_dividend_amount = dividends.iloc[-1]
            
            dividend_info['last_dividend_date'] = last_dividend_date
            dividend_info['last_dividend_amount'] = last_dividend_amount
            currency_symbol = get_currency_symbol(market)
            dividend_info['last_dividend_formatted'] = f"{currency_symbol}{last_dividend_amount:.2f} on {last_dividend_date.strftime('%Y-%m-%d')}"
        
        # Get dividend yield from ticker info
        if 'dividendYield' in ticker_info and ticker_info['dividendYield']:
            # dividendYield is already in percentage format (0.45 = 0.45%, not 45%)
            dividend_info['dividend_yield'] = f"{ticker_info['dividendYield']:.2f}%"
        elif 'trailingAnnualDividendYield' in ticker_info and ticker_info['trailingAnnualDividendYield']:
            # Check if this value needs multiplication based on its scale
            trailing_yield = ticker_info['trailingAnnualDividendYield']
            if trailing_yield < 1:  # Likely a decimal (0.05 = 5%)
                dividend_info['dividend_yield'] = f"{trailing_yield*100:.2f}%"
            else:  # Already in percentage format
                dividend_info['dividend_yield'] = f"{trailing_yield:.2f}%"
        
        # Get forward dividend rate
        currency_symbol = get_currency_symbol(market)
        if 'dividendRate' in ticker_info and ticker_info['dividendRate']:
            dividend_info['forward_dividend'] = f"{currency_symbol}{ticker_info['dividendRate']:.2f}"
        elif 'trailingAnnualDividendRate' in ticker_info and ticker_info['trailingAnnualDividendRate']:
            dividend_info['forward_dividend'] = f"{currency_symbol}{ticker_info['trailingAnnualDividendRate']:.2f}"
        
        # Get payout ratio
        if 'payoutRatio' in ticker_info and ticker_info['payoutRatio']:
            payout = ticker_info['payoutRatio']
            # Check if payout ratio needs multiplication (usually it's a decimal like 0.3 = 30%)
            if payout <= 1:
                dividend_info['payout_ratio'] = f"{payout*100:.1f}%"
            else:  # Already in percentage format
                dividend_info['payout_ratio'] = f"{payout:.1f}%"
            
    except Exception as e:
        pass
    
    return dividend_info

def format_currency(value, market="US"):
    """
    Format currency based on market
    
    Args:
        value (float): Currency value
        market (str): Market type ("US" or "India")
    
    Returns:
        str: Formatted currency string
    """
    if market == "India":
        return f"₹{value:,.2f}"
    else:
        return f"${value:.2f}"

def get_currency_symbol(market="US"):
    """
    Get currency symbol for the market
    
    Args:
        market (str): Market type ("US" or "India")
    
    Returns:
        str: Currency symbol
    """
    return "₹" if market == "India" else "$"

def get_stock_metrics(symbol, period="1y", market="US"):
    """
    Get comprehensive metrics for a single stock
    
    Args:
        symbol (str): Stock ticker symbol
        period (str): Time period for analysis
        market (str): Market type ("US" or "India")
    
    Returns:
        dict: Dictionary containing all key metrics
    """
    try:
        # Fetch data
        data, ticker_info, ticker_obj = fetch_stock_data(symbol, period, market)
        
        if data is None or data.empty:
            return {
                'Symbol': symbol,
                'Error': 'No data available',
                'Current Price': 'N/A',
                '52-Week High': 'N/A',
                '52-Week Low': 'N/A',
                'Distance from High (%)': 'N/A',
                'Distance from Low (%)': 'N/A',
                '50-Day MA': 'N/A',
                '200-Day MA': 'N/A',
                'Price vs 50-Day MA (%)': 'N/A',
                'Price vs 200-Day MA (%)': 'N/A',
                'Support Level': 'N/A',
                'Resistance Level': 'N/A',
                'RSI': 'N/A',
                'MACD Signal': 'N/A',
                'Chaikin Money Flow': 'N/A',
                'Last Earnings Date': 'N/A',
                'Next Earnings Date': 'N/A',
                'Earnings Performance (%)': 'N/A',
                'Last Dividend Date': 'N/A',
                'Last Dividend Amount': 'N/A',
                'Dividend Yield (%)': 'N/A',
                'Forward Dividend': 'N/A',
                'Payout Ratio (%)': 'N/A',
                'Est. Next Dividend Date': 'N/A'
            }
        
        # Calculate all metrics
        latest_price = data['Close'].iloc[-1]
        year_high = data['Close'].max()
        year_low = data['Close'].min()
        
        # Moving averages
        ma_50 = calculate_moving_average(data, window=50)
        ma_200 = calculate_moving_average(data, window=200)
        latest_ma_50 = ma_50.iloc[-1] if not ma_50.empty else None
        latest_ma_200 = ma_200.iloc[-1] if not ma_200.empty else None
        
        # Technical indicators
        macd_line, signal_line, histogram = calculate_macd(data)
        rsi = calculate_rsi(data)
        cmf = calculate_chaikin_money_flow(data)
        
        # Support and resistance
        support_level, resistance_level = calculate_support_resistance(data)
        
        # Earnings and dividend info
        earnings_info = get_earnings_info(ticker_obj, ticker_info)
        dividend_info = get_dividend_info(ticker_obj, ticker_info, market)
        
        # Calculate performance metrics
        distance_from_high = ((year_high - latest_price) / year_high) * 100
        distance_from_low = ((latest_price - year_low) / year_low) * 100
        
        price_vs_ma_50 = ((latest_price - latest_ma_50) / latest_ma_50) * 100 if latest_ma_50 and not pd.isna(latest_ma_50) else None
        price_vs_ma_200 = ((latest_price - latest_ma_200) / latest_ma_200) * 100 if latest_ma_200 and not pd.isna(latest_ma_200) else None
        
        # Earnings performance
        earnings_performance = None
        if earnings_info['last_earnings'] is not None:
            try:
                earnings_date = earnings_info['last_earnings']
                
                # Handle timezone compatibility for comparison
                if hasattr(data.index[0], 'tz') and data.index[0].tz:
                    # Data has timezone, ensure earnings_date has the same timezone
                    if hasattr(earnings_date, 'tz') and earnings_date.tz:
                        earnings_date_tz = earnings_date
                    else:
                        earnings_date_tz = earnings_date.tz_localize(data.index[0].tz)
                else:
                    # Data is timezone-naive, make earnings_date naive too
                    if hasattr(earnings_date, 'tz') and earnings_date.tz:
                        earnings_date_tz = earnings_date.tz_localize(None)
                    else:
                        earnings_date_tz = earnings_date
                
                # Find the trading day closest to or after earnings
                post_earnings_mask = data.index >= earnings_date_tz
                if post_earnings_mask.any():
                    earnings_day_price = data[post_earnings_mask]['Close'].iloc[0]
                    earnings_performance = ((latest_price - earnings_day_price) / earnings_day_price) * 100
                    print(f"Earnings performance calculated: {earnings_performance:.2f}%")
                else:
                    # Try finding pre-earnings data and use that as baseline
                    pre_earnings_mask = data.index <= earnings_date_tz
                    if pre_earnings_mask.any():
                        pre_earnings_price = data[pre_earnings_mask]['Close'].iloc[-1]
                        earnings_performance = ((latest_price - pre_earnings_price) / pre_earnings_price) * 100
                        print(f"Earnings performance (pre-earnings baseline): {earnings_performance:.2f}%")
            except Exception as e:
                print(f"Earnings performance error: {e}")
                pass
        
        # Latest indicator values
        latest_rsi = rsi.iloc[-1] if not rsi.empty else None
        latest_macd = macd_line.iloc[-1] if not macd_line.empty else None
        latest_signal = signal_line.iloc[-1] if not signal_line.empty else None
        latest_cmf = cmf.iloc[-1] if not cmf.empty else None
        
        # MACD signal
        macd_signal = "N/A"
        if latest_macd is not None and latest_signal is not None:
            if latest_macd > latest_signal:
                macd_signal = "Bullish"
            else:
                macd_signal = "Bearish"
        
        # Estimated next dividend
        next_dividend_estimate = "N/A"
        if dividend_info['last_dividend_date'] is not None:
            try:
                estimated_next = dividend_info['last_dividend_date'] + pd.DateOffset(days=90)
                next_dividend_estimate = estimated_next.strftime('%Y-%m-%d')
            except:
                pass
        
        # Beta value and Safe Levels
        beta_value = get_beta_value(ticker_info)
        ctp_levels = calculate_ctp_levels(latest_price)
        
        # Earnings performance analysis summary
        earnings_analysis, quarters_found = get_earnings_performance_analysis(ticker_obj, data, market)
        earnings_summary = "N/A"
        if earnings_analysis is not None and not earnings_analysis.empty:
            try:
                # Calculate summary statistics
                week_changes = [float(x.replace('%', '').replace('+', '')) for x in earnings_analysis['Week Performance (%)'] if x != 'N/A']
                if week_changes:
                    avg_week = sum(week_changes) / len(week_changes)
                    positive_week = sum(1 for x in week_changes if x > 0)
                    earnings_summary = f"{positive_week}/{len(week_changes)} positive (avg: {avg_week:+.1f}%)"
            except:
                pass
        
        # Return comprehensive metrics with proper currency formatting
        currency_symbol = get_currency_symbol(market)
        return {
            'Symbol': symbol,
            'Current Price': format_currency(latest_price, market),
            '52-Week High': format_currency(year_high, market),
            '52-Week Low': format_currency(year_low, market),
            'Distance from High (%)': f"{distance_from_high:.1f}%",
            'Distance from Low (%)': f"{distance_from_low:.1f}%",
            '50-Day MA': format_currency(latest_ma_50, market) if latest_ma_50 and not pd.isna(latest_ma_50) else "N/A",
            '200-Day MA': format_currency(latest_ma_200, market) if latest_ma_200 and not pd.isna(latest_ma_200) else "N/A",
            'Price vs 50-Day MA (%)': f"{price_vs_ma_50:.1f}%" if price_vs_ma_50 is not None else "N/A",
            'Price vs 200-Day MA (%)': f"{price_vs_ma_200:.1f}%" if price_vs_ma_200 is not None else "N/A",
            'Support Level': format_currency(support_level, market),
            'Resistance Level': format_currency(resistance_level, market),
            'RSI': f"{latest_rsi:.1f}" if latest_rsi and not pd.isna(latest_rsi) else "N/A",
            'MACD Signal': macd_signal,
            'Chaikin Money Flow': f"{latest_cmf:.3f}" if latest_cmf and not pd.isna(latest_cmf) else "N/A",
            'Last Earnings Date': earnings_info['last_earnings_formatted'],
            'Next Earnings Date': earnings_info['next_earnings_formatted'],
            'Earnings Performance (%)': f"{earnings_performance:.1f}%" if earnings_performance is not None else "N/A",
            'Last Dividend Date': dividend_info['last_dividend_formatted'],
            'Last Dividend Amount': format_currency(dividend_info['last_dividend_amount'], market) if dividend_info['last_dividend_amount'] > 0 else "N/A",
            'Dividend Yield (%)': dividend_info['dividend_yield'],
            'Forward Dividend': dividend_info['forward_dividend'],
            'Payout Ratio (%)': dividend_info['payout_ratio'],
            'Est. Next Dividend Date': next_dividend_estimate,
            'Beta': beta_value,
            'Safe Level Low': format_currency(ctp_levels['lower_ctp'], market) if ctp_levels['lower_ctp'] else "N/A",
            'Safe Level High': format_currency(ctp_levels['upper_ctp'], market) if ctp_levels['upper_ctp'] else "N/A",
            'Earnings Performance (4Q)': earnings_summary
        }
        
    except Exception as e:
        return {
            'Symbol': symbol,
            'Error': str(e),
            'Current Price': 'Error',
            '52-Week High': 'Error',
            '52-Week Low': 'Error',
            'Distance from High (%)': 'Error',
            'Distance from Low (%)': 'Error',
            '50-Day MA': 'Error',
            '200-Day MA': 'Error',
            'Price vs 50-Day MA (%)': 'Error',
            'Price vs 200-Day MA (%)': 'Error',
            'Support Level': 'Error',
            'Resistance Level': 'Error',
            'RSI': 'Error',
            'MACD Signal': 'Error',
            'Chaikin Money Flow': 'Error',
            'Last Earnings Date': 'Error',
            'Next Earnings Date': 'Error',
            'Earnings Performance (%)': 'Error',
            'Last Dividend Date': 'Error',
            'Last Dividend Amount': 'Error',
            'Dividend Yield (%)': 'Error',
            'Forward Dividend': 'Error',
            'Payout Ratio (%)': 'Error',
            'Est. Next Dividend Date': 'Error',
            'Beta': 'Error',
            'Safe Level Low': 'Error',
            'Safe Level High': 'Error',
            'Earnings Performance (4Q)': 'Error'
        }

def create_excel_report(stock_metrics_list, period_label="1 Year"):
    """
    Create an Excel report with stock metrics
    
    Args:
        stock_metrics_list (list): List of stock metrics dictionaries
        period_label (str): Time period label for the report
    
    Returns:
        io.BytesIO: Excel file as bytes
    """
    # Create DataFrame
    df = pd.DataFrame(stock_metrics_list)
    
    # Create Excel file in memory
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write main data
        df.to_excel(writer, sheet_name='Stock Analysis', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Stock Analysis']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write headers with formatting
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-adjust column widths
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(i, i, min(max_len, 20))
        
        # Add summary sheet
        summary_data = {
            'Report Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Analysis Period': [period_label],
            'Total Stocks': [len(stock_metrics_list)],
            'Successful Analysis': [len([s for s in stock_metrics_list if 'Error' not in s or s.get('Error') == ''])],
            'Failed Analysis': [len([s for s in stock_metrics_list if 'Error' in s and s.get('Error') != ''])]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    return output

def create_chart(data, symbol, ma_50, ma_200, period_label="1 Year", market="US", show_fibonacci=True):
    """
    Create an interactive Plotly chart with stock price, moving averages, and Fibonacci retracements
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol for chart title
        ma_50 (pd.Series): 50-day moving average data
        ma_200 (pd.Series): 200-day moving average data
        period_label (str): Time period label for chart title
        market (str): Market type ("US" or "India")
        show_fibonacci (bool): Whether to show Fibonacci retracement levels
    
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    fig = go.Figure()
    
    # Get currency symbol
    currency_symbol = get_currency_symbol(market)
    
    # Add stock price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name=f'{symbol.upper()} Close Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      f'Price: {currency_symbol}%{{y:,.2f}}<br>' +
                      '<extra></extra>'
    ))
    
    # Add 50-day moving average line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ma_50,
        mode='lines',
        name='50-Day Moving Average',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      f'MA(50): {currency_symbol}%{{y:,.2f}}<br>' +
                      '<extra></extra>'
    ))
    
    # Add 200-day moving average line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ma_200,
        mode='lines',
        name='200-Day Moving Average',
        line=dict(color='#d62728', width=2, dash='dot'),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      f'MA(200): {currency_symbol}%{{y:,.2f}}<br>' +
                      '<extra></extra>'
    ))
    
    # Add Fibonacci retracement levels
    if show_fibonacci:
        fib_data = calculate_fibonacci_retracements(data)
        if fib_data:
            fib_colors = ['rgba(255, 215, 0, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 140, 0, 0.3)', 
                         'rgba(255, 69, 0, 0.3)', 'rgba(255, 0, 0, 0.3)', 'rgba(139, 0, 0, 0.3)', 'rgba(75, 0, 130, 0.3)']
            
            # Add horizontal lines for each Fibonacci level
            for i, (level_name, level_price) in enumerate(fib_data['fib_levels'].items()):
                fig.add_hline(
                    y=level_price,
                    line=dict(
                        color=fib_colors[i % len(fib_colors)].replace('0.3', '0.8'),
                        width=1,
                        dash='dash'
                    ),
                    annotation=dict(
                        text=f"Fib {level_name}: {currency_symbol}{level_price:.2f}",
                        bgcolor=fib_colors[i % len(fib_colors)],
                        bordercolor=fib_colors[i % len(fib_colors)].replace('0.3', '0.8'),
                        font=dict(size=10)
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol.upper()} Stock Price with 50-Day & 200-Day Moving Averages ({period_label})',
        xaxis_title='Date',
        yaxis_title=f'Price ({currency_symbol})',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        height=600
    )
    
    # Update x-axis to show better date formatting
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    # Update y-axis with market-specific formatting
    if market == "India":
        tick_format = '₹,.2f'
    else:
        tick_format = '$,.2f'
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickformat=tick_format
    )
    
    return fig

def create_macd_chart(data, symbol, macd_line, signal_line, histogram, period_label="1 Year", market="US"):
    """
    Create MACD indicator chart
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol for chart title
        macd_line (pd.Series): MACD line data
        signal_line (pd.Series): Signal line data
        histogram (pd.Series): MACD histogram data
        period_label (str): Time period label for chart title
    
    Returns:
        plotly.graph_objects.Figure: MACD chart
    """
    fig = go.Figure()
    
    # Add MACD line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=macd_line,
        mode='lines',
        name='MACD Line',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'MACD: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add Signal line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=signal_line,
        mode='lines',
        name='Signal Line',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'Signal: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add histogram bars
    # Create colors for bars (green for positive, red for negative)
    colors = ['green' if val >= 0 else 'red' for val in histogram]
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=histogram,
        name='MACD Histogram',
        marker_color=colors,
        opacity=0.6,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'Histogram: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol.upper()} MACD Indicator ({period_label})',
        xaxis_title='Date',
        yaxis_title='MACD Value',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        height=400
    )
    
    # Update x-axis
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    # Update y-axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickformat='.4f'
    )
    
    return fig

def create_rsi_chart(data, symbol, rsi, period_label="1 Year", market="US"):
    """
    Create RSI indicator chart
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol for chart title
        rsi (pd.Series): RSI data
        period_label (str): Time period label for chart title
    
    Returns:
        plotly.graph_objects.Figure: RSI chart
    """
    fig = go.Figure()
    
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=rsi,
        mode='lines',
        name='RSI',
        line=dict(color='#9467bd', width=2),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'RSI: %{y:.2f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add overbought and oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, annotation_text="Neutral (50)")
    
    # Update layout
    fig.update_layout(
        title=f'{symbol.upper()} RSI Indicator ({period_label})',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        height=350
    )
    
    # Update x-axis
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    # Update y-axis
    fig.update_yaxes(
        range=[0, 100],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickformat='.0f'
    )
    
    return fig

def create_chaikin_chart(data, symbol, cmf, period_label="1 Year", market="US"):
    """
    Create Chaikin Money Flow chart
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol for chart title
        cmf (pd.Series): Chaikin Money Flow data
        period_label (str): Time period label for chart title
    
    Returns:
        plotly.graph_objects.Figure: Chaikin Money Flow chart
    """
    fig = go.Figure()
    
    # Create colors for bars (green for positive, red for negative)
    colors = ['green' if val >= 0 else 'red' for val in cmf]
    
    # Add CMF bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=cmf,
        name='Chaikin Money Flow',
        marker_color=colors,
        opacity=0.7,
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'CMF: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol.upper()} Chaikin Money Flow ({period_label})',
        xaxis_title='Date',
        yaxis_title='CMF Value',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        height=350
    )
    
    # Update x-axis
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    # Update y-axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickformat='.4f'
    )
    
    return fig

def display_key_metrics(data, symbol, ma_50, ma_200, rsi, ticker_info, ticker_obj, support_level, resistance_level, market="US"):
    """
    Display comprehensive key metrics about the stock
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol
        ma_50 (pd.Series): 50-day moving average data
        ma_200 (pd.Series): 200-day moving average data
        ticker_info (dict): Company information from yfinance
        ticker_obj: yfinance Ticker object for dividend data
        support_level (float): Calculated support level
        resistance_level (float): Calculated resistance level
    """
    # Get latest values
    latest_price = data['Close'].iloc[-1]
    latest_ma_50 = ma_50.iloc[-1]
    latest_ma_200 = ma_200.iloc[-1]
    
    # Calculate comprehensive metrics
    year_high = data['Close'].max()
    year_low = data['Close'].min()
    
    # Distance from 52-week high/low
    distance_from_high = ((year_high - latest_price) / year_high) * 100
    distance_from_low = ((latest_price - year_low) / year_low) * 100
    
    # Price vs MA comparison
    price_vs_ma_50 = ((latest_price - latest_ma_50) / latest_ma_50) * 100 if not pd.isna(latest_ma_50) else 0
    price_vs_ma_200 = ((latest_price - latest_ma_200) / latest_ma_200) * 100 if not pd.isna(latest_ma_200) else 0
    
    # Get earnings and dividend information
    earnings_info = get_earnings_info(ticker_obj, ticker_info)
    dividend_info = get_dividend_info(ticker_obj, ticker_info, market)
    
    # Calculate performance since last earnings (if available)
    earnings_performance = "N/A"
    if earnings_info['last_earnings'] is not None:
        try:
            # Find closest trading day to earnings date
            earnings_date = earnings_info['last_earnings']
            mask = data.index >= earnings_date
            if mask.any():
                earnings_day_price = data[mask]['Close'].iloc[0]
                earnings_performance = f"{((latest_price - earnings_day_price) / earnings_day_price) * 100:.1f}%"
        except:
            pass
    
    # First row of metrics
    st.markdown("**📈 Current Price & Position**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Current Price",
            value=format_currency(latest_price, market),
            delta=f"{price_vs_ma_50:.1f}% vs MA(50)" if not pd.isna(latest_ma_50) else None
        )
    
    with col2:
        st.metric(
            label="52-Week High",
            value=format_currency(year_high, market),
            delta=f"{distance_from_high:.1f}% below high"
        )
    
    with col3:
        st.metric(
            label="52-Week Low",
            value=format_currency(year_low, market),
            delta=f"+{distance_from_low:.1f}% above low"
        )
    
    with col4:
        st.metric(
            label="Support Level",
            value=format_currency(support_level, market),
            help="Recent 20-day support level"
        )
    
    with col5:
        st.metric(
            label="Resistance Level",
            value=format_currency(resistance_level, market),
            help="Recent 20-day resistance level"
        )
    
    # Second row of metrics
    st.markdown("**📊 Technical Indicators & Moving Averages**")
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        st.metric(
            label="50-Day MA",
            value=format_currency(latest_ma_50, market) if not pd.isna(latest_ma_50) else "N/A"
        )
    
    with col7:
        st.metric(
            label="200-Day MA",
            value=format_currency(latest_ma_200, market) if not pd.isna(latest_ma_200) else "N/A",
            delta=f"{price_vs_ma_200:.1f}% vs Price" if not pd.isna(latest_ma_200) else None
        )
    
    with col8:
        st.metric(
            label="Since Earnings",
            value=earnings_performance,
            help="Price change since last earnings"
        )
    
    with col9:
        latest_rsi = rsi.iloc[-1] if not rsi.empty else None
        st.metric(
            label="RSI (14)",
            value=f"{latest_rsi:.1f}" if latest_rsi and not pd.isna(latest_rsi) else "N/A",
            help="Relative Strength Index (14-period)"
        )
    
    with col10:
        # Get beta value
        beta_value = get_beta_value(ticker_info)
        st.metric(
            label="Beta",
            value=beta_value,
            help="Stock's volatility relative to the market (1.0 = market average)"
        )
    
    # New row for Safe Levels and earnings info
    st.markdown("**🎯 Price Targets & Earnings Data**")
    col_ctp1, col_ctp2, col_ctp3, col_ctp4, col_ctp5 = st.columns(5)
    
    with col_ctp1:
        # Safe Level Low
        ctp_levels = calculate_ctp_levels(latest_price)
        st.metric(
            label="Safe Level Low",
            value=format_currency(ctp_levels['lower_ctp'], market) if ctp_levels['lower_ctp'] else "N/A",
            help="Lower support reference level"
        )
    
    with col_ctp2:
        # Current price for reference
        st.metric(
            label="Current Level",
            value=format_currency(latest_price, market),
            help="Current market price reference point"
        )
    
    with col_ctp3:
        # Safe Level High
        st.metric(
            label="Safe Level High",
            value=format_currency(ctp_levels['upper_ctp'], market) if ctp_levels['upper_ctp'] else "N/A",
            help="Upper resistance reference level"
        )
    
    with col_ctp4:
        st.metric(
            label="Last Earnings",
            value=earnings_info['last_earnings_formatted'],
            help="Most recent earnings announcement date"
        )
    
    with col_ctp5:
        st.metric(
            label="Next Earnings",
            value=earnings_info['next_earnings_formatted'],
            help="Expected next earnings date"
        )

    # After-market data section
    after_market = get_after_market_data(symbol, market)
    if after_market['pre_market_change'] != 'N/A' or after_market['post_market_change'] != 'N/A':
        st.markdown("**🕘 Extended Hours Trading**")
        col_am1, col_am2, col_am3, col_am4 = st.columns(4)
        
        with col_am1:
            st.metric(
                label="Regular Close",
                value=after_market['regular_session_close'],
                help="Official market close price"
            )
        
        with col_am2:
            st.metric(
                label="Current Price", 
                value=after_market['current_price'],
                help="Most recent available price"
            )
        
        with col_am3:
            if after_market['pre_market_change'] != 'N/A':
                st.metric(
                    label="Pre-Market",
                    value=after_market['pre_market_change'],
                    delta=after_market['pre_market_change_percent'],
                    help="Pre-market price movement"
                )
            else:
                st.metric(
                    label="Pre-Market",
                    value="N/A",
                    help="No pre-market data available"
                )
        
        with col_am4:
            if after_market['post_market_change'] != 'N/A':
                st.metric(
                    label="After-Hours",
                    value=after_market['post_market_change'],
                    delta=after_market['post_market_change_percent'],
                    help="After-hours price movement"
                )
            else:
                st.metric(
                    label="After-Hours",
                    value="N/A",
                    help="No after-hours data available"
                )

    # Third row of metrics - Dividend Information
    st.markdown("**💰 Dividend Information**")
    col11, col12, col13, col14, col15 = st.columns(5)
    
    with col11:
        st.metric(
            label="Last Dividend",
            value=dividend_info['last_dividend_formatted'],
            help="Most recent dividend payment"
        )
    
    with col12:
        st.metric(
            label="Dividend Yield",
            value=dividend_info['dividend_yield'],
            help="Annual dividend yield percentage"
        )
    
    with col13:
        st.metric(
            label="Forward Dividend",
            value=dividend_info['forward_dividend'],
            help="Expected annual dividend per share"
        )
    
    with col14:
        st.metric(
            label="Payout Ratio",
            value=dividend_info['payout_ratio'],
            help="Percentage of earnings paid as dividends"
        )
    
    with col15:
        # Calculate estimated next dividend date if we have dividend history
        next_dividend_estimate = "N/A"
        if dividend_info['last_dividend_date'] is not None:
            try:
                # Estimate next dividend (typically quarterly, so add ~90 days)
                estimated_next = dividend_info['last_dividend_date'] + pd.DateOffset(days=90)
                next_dividend_estimate = estimated_next.strftime('%Y-%m-%d')
            except:
                pass
        
        st.metric(
            label="Est. Next Dividend",
            value=next_dividend_estimate,
            help="Estimated next dividend date (90 days from last)"
        )
    
    # Fourth row of metrics - Fibonacci Analysis
    fib_data = calculate_fibonacci_retracements(data)
    if fib_data:
        st.markdown("---")
        st.markdown("**🔢 Fibonacci Analysis**")
        
        # Calculate both uptrend and downtrend Fibonacci levels
        fib_uptrend = calculate_fibonacci_retracements(data, period=50)
        fib_downtrend = None
        
        # Force calculate downtrend levels 
        if fib_uptrend:
            swing_high = fib_uptrend['swing_high']
            swing_low = fib_uptrend['swing_low']
            
            # Create downtrend levels manually
            fibonacci_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
            price_range = abs(swing_high - swing_low)
            
            # Downtrend retracements go up from the low
            downtrend_levels = {}
            for ratio in fibonacci_ratios:
                downtrend_levels[f"{ratio*100:.1f}%"] = swing_low + (price_range * ratio)
                
            # Uptrend retracements go down from the high  
            uptrend_levels = {}
            for ratio in fibonacci_ratios:
                uptrend_levels[f"{ratio*100:.1f}%"] = swing_high - (price_range * ratio)
        
        # Display trend information
        trend_direction = fib_uptrend['trend_direction'] if fib_uptrend else "unknown"
        trend_emoji = "📈" if trend_direction == "uptrend" else "📉"
        
        col_trend1, col_trend2, col_trend3, col_trend4 = st.columns(4)
        
        with col_trend1:
            st.metric(
                label=f"{trend_emoji} Detected Trend",
                value=trend_direction.title(),
                help=f"Current market trend based on recent price movement and swing points"
            )
        
        with col_trend2:
            st.metric(
                label="Swing High",
                value=format_currency(swing_high, market),
                help=f"Recent high on {fib_data['swing_high_date'].strftime('%Y-%m-%d')}"
            )
        
        with col_trend3:
            st.metric(
                label="Swing Low", 
                value=format_currency(swing_low, market),
                help=f"Recent low on {fib_data['swing_low_date'].strftime('%Y-%m-%d')}"
            )
        
        with col_trend4:
            price_range = fib_data['price_range']
            st.metric(
                label="Price Range",
                value=format_currency(price_range, market),
                help="Difference between swing high and low"
            )
        
        # Display both uptrend and downtrend Fibonacci levels
        st.markdown("**📈 Uptrend Fibonacci Levels (Retracements from High):**")
        if fib_uptrend:
            col_up1, col_up2, col_up3, col_up4, col_up5 = st.columns(5)
            up_level_names = list(uptrend_levels.keys())
            
            with col_up1:
                if len(up_level_names) > 1:  # 23.6%
                    level = up_level_names[1]
                    price = uptrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↗️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="23.6% retracement from swing high"
                    )
            
            with col_up2:
                if len(up_level_names) > 2:  # 38.2%
                    level = up_level_names[2]
                    price = uptrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↗️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="38.2% retracement from swing high"
                    )
            
            with col_up3:
                if len(up_level_names) > 3:  # 50%
                    level = up_level_names[3]
                    price = uptrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↗️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="50% retracement from swing high"
                    )
            
            with col_up4:
                if len(up_level_names) > 4:  # 61.8%
                    level = up_level_names[4]
                    price = uptrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↗️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="61.8% retracement (Golden Ratio)"
                    )
            
            with col_up5:
                if len(up_level_names) > 5:  # 78.6%
                    level = up_level_names[5]
                    price = uptrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↗️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="78.6% retracement from swing high"
                    )
        
        st.markdown("**📉 Downtrend Fibonacci Levels (Retracements from Low):**")
        if fib_uptrend:
            col_down1, col_down2, col_down3, col_down4, col_down5 = st.columns(5)
            down_level_names = list(downtrend_levels.keys())
            
            with col_down1:
                if len(down_level_names) > 1:  # 23.6%
                    level = down_level_names[1]
                    price = downtrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↘️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="23.6% retracement from swing low"
                    )
            
            with col_down2:
                if len(down_level_names) > 2:  # 38.2%
                    level = down_level_names[2]
                    price = downtrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↘️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="38.2% retracement from swing low"
                    )
            
            with col_down3:
                if len(down_level_names) > 3:  # 50%
                    level = down_level_names[3]
                    price = downtrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↘️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="50% retracement from swing low"
                    )
            
            with col_down4:
                if len(down_level_names) > 4:  # 61.8%
                    level = down_level_names[4]
                    price = downtrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↘️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="61.8% retracement (Golden Ratio)"
                    )
            
            with col_down5:
                if len(down_level_names) > 5:  # 78.6%
                    level = down_level_names[5]
                    price = downtrend_levels[level]
                    distance = ((latest_price - price) / price * 100)
                    st.metric(
                        label=f"↘️ {level}",
                        value=format_currency(price, market),
                        delta=f"{distance:+.1f}%" if abs(distance) < 50 else None,
                        help="78.6% retracement from swing low"
                    )
    
    # Add Social Sharing Section
    st.markdown("---")
    st.subheader("📤 Share Your Analysis")
    st.markdown("Share your investment insights with customizable privacy settings")
    
    col_privacy, col_share = st.columns([1, 2])
    
    with col_privacy:
        privacy_level = st.selectbox(
            "Privacy Level:",
            ["public", "anonymized", "private"],
            format_func=lambda x: {
                "public": "🌐 Public - Full Details",
                "anonymized": "🔒 Anonymized - No Stock Name", 
                "private": "🔐 Private - Limited Info"
            }.get(x, x),
            help="Choose how much information to include when sharing"
        )
    
    with col_share:
        if st.button("🚀 Generate Shareable Insight", type="primary"):
            # Collect all metrics for sharing
            share_metrics = {
                'longName': ticker_info.get('longName', symbol),
                'RSI': f"{rsi.iloc[-1]:.2f}" if not rsi.empty else "N/A",
                'Price vs 50-Day MA (%)': f"{price_vs_ma_50:+.2f}%",
                'Price vs 200-Day MA (%)': f"{price_vs_ma_200:+.2f}%",
                'Beta': get_beta_value(ticker_info),
                'Volume': f"{data['Volume'].iloc[-1]:,.0f}" if 'Volume' in data.columns else "N/A"
            }
            
            # Get Fibonacci analysis data
            fib_data = calculate_fibonacci_retracements(data)
            
            # Create shareable insight
            insight = create_shareable_insight(symbol, data, share_metrics, privacy_level, fib_data)
            
            if 'error' not in insight:
                # Display the generated insight
                st.success("✅ Shareable insight generated!")
                
                with st.expander("📋 Preview Your Insight", expanded=True):
                    if privacy_level == "public":
                        st.markdown(f"**📊 {insight['symbol']} Analysis**")
                        st.markdown(f"**Company:** {insight['company_name']}")
                        st.markdown(f"**Current Price:** {insight['current_price']} ({insight['daily_change']})")
                        st.markdown(f"**Recommendation:** {insight['recommendation']}")
                        st.markdown(f"**Risk Level:** {insight['risk_level']}")
                        
                        col_rsi, col_ma50, col_ma200 = st.columns(3)
                        with col_rsi:
                            st.metric("RSI", insight['key_metrics']['rsi'])
                        with col_ma50:
                            st.metric("vs 50-Day MA", insight['key_metrics']['ma_50_signal'])
                        with col_ma200:
                            st.metric("vs 200-Day MA", insight['key_metrics']['ma_200_signal'])
                        
                        # Show Fibonacci analysis if available
                        if 'fibonacci_analysis' in insight:
                            st.markdown("**📈 Fibonacci Analysis:**")
                            fib_data = insight['fibonacci_analysis']
                            col_trend, col_level, col_range = st.columns(3)
                            with col_trend:
                                trend_emoji = "📈" if fib_data['trend'] == "Uptrend" else "📉" if fib_data['trend'] == "Downtrend" else "➡️"
                                st.metric("Trend", f"{trend_emoji} {fib_data['trend']}")
                            with col_level:
                                level_status = "🎯" if fib_data.get('near_key_level') else "📊"
                                st.metric("Key Level", f"{level_status} {fib_data['key_level']}")
                            with col_range:
                                st.metric("Swing Range", fib_data['swing_range'])
                    
                    elif privacy_level == "anonymized":
                        st.markdown(f"**📊 Anonymous Stock Analysis**")
                        st.markdown(f"**Current Price:** {insight['current_price']} ({insight['daily_change']})")
                        st.markdown(f"**Recommendation:** {insight['recommendation']}")
                        st.info(insight['note'])
                        
                        # Show Fibonacci analysis if available
                        if 'fibonacci_analysis' in insight:
                            fib_data = insight['fibonacci_analysis']
                            col_anon_trend, col_anon_level = st.columns(2)
                            with col_anon_trend:
                                trend_emoji = "📈" if fib_data['trend'] == "Uptrend" else "📉" if fib_data['trend'] == "Downtrend" else "➡️"
                                st.metric("Trend Pattern", f"{trend_emoji} {fib_data['trend']}")
                            with col_anon_level:
                                st.metric("Near Key Level", fib_data['key_level'])
                    
                    else:  # private
                        st.markdown(f"**📊 Private Analysis for {insight['symbol']}**")
                        st.markdown(f"**Summary:** {insight['analysis_summary']}")
                        st.markdown(f"**Recommendation:** {insight['recommendation']}")
                
                # Generate share URLs
                share_urls = create_share_urls(insight)
                
                st.markdown("### 🔗 Share Options")
                col_twitter, col_linkedin, col_copy, col_email = st.columns(4)
                
                with col_twitter:
                    st.markdown(f"[📱 Twitter]({share_urls['twitter']})")
                
                with col_linkedin:
                    st.markdown(f"[💼 LinkedIn]({share_urls['linkedin']})")
                
                with col_copy:
                    if st.button("📋 Copy Text"):
                        st.code(share_urls['copy_text'], language="text")
                        st.success("Text ready to copy!")
                
                with col_email:
                    email_url = f"mailto:?subject={urllib.parse.quote(share_urls['email_subject'])}&body={urllib.parse.quote(share_urls['email_body'])}"
                    st.markdown(f"[📧 Email]({email_url})")
                
                # Store insight in session state for potential later use
                if 'shared_insights' not in st.session_state:
                    st.session_state.shared_insights = []
                st.session_state.shared_insights.append(insight)
                
                # Show insight ID for reference
                st.caption(f"Insight ID: {insight['id']} | Generated: {insight['timestamp'][:19]}")
            
            else:
                st.error(f"Failed to generate insight: {insight['error']}")


def main():
    """
    Main application function
    """
    # Custom CSS to make tabs bigger and more visible
    st.markdown("""
    <style>
    /* Make main tabs bigger and more visible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f0f2f6;
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 65px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 10px;
        color: #262730;
        font-size: 18px;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        padding: 18px 30px;
        border: 3px solid #e1e5e9;
        transition: all 0.3s ease;
        box-shadow: 0 3px 8px rgba(0,0,0,0.12);
        min-width: 280px;
        text-align: center;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
        border-color: #0066cc;
        box-shadow: 0 6px 15px rgba(0,102,204,0.4);
        transform: translateY(-3px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f8f9fa;
        border-color: #0066cc;
        transform: translateY(-2px);
        box-shadow: 0 5px 12px rgba(0,0,0,0.18);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background-color: #0052a3;
        color: white;
        transform: translateY(-3px);
    }
    
    /* Sub-tabs styling */
    .stTabs .stTabs [data-baseweb="tab-list"] {
        background-color: #fafbfc;
        padding: 10px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    
    .stTabs .stTabs [data-baseweb="tab"] {
        height: 55px;
        font-size: 16px;
        font-weight: 600;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        padding: 15px 25px;
        background-color: white;
        border: 2px solid #d1d5db;
        border-radius: 8px;
        min-width: 200px;
        text-align: center;
    }
    
    .stTabs .stTabs [aria-selected="true"] {
        background-color: #10b981;
        color: white;
        border-color: #10b981;
        box-shadow: 0 4px 10px rgba(16,185,129,0.3);
        transform: translateY(-2px);
    }
    
    .stTabs .stTabs [data-baseweb="tab"]:hover {
        background-color: #f3f4f6;
        border-color: #10b981;
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }
    
    .stTabs .stTabs [aria-selected="true"]:hover {
        background-color: #059669;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.title("📈 Stock Technical Analysis Tool")
    st.markdown("Get comprehensive technical analysis with moving averages, MACD, RSI, Chaikin Money Flow, earnings data, and dividend information for any stock symbol.")
    
    # Create data source tabs
    tab_yahoo, tab_guru = st.tabs(["📊 Fundamental Analysis", "🎯 Advanced Analysis"])
    
    with tab_yahoo:
        yahoo_finance_tab()
    
    with tab_guru:
        gurufocus_tab()

def yahoo_finance_tab():
    """Fundamental analysis tab content"""
    st.markdown("### Real-time stock analysis with comprehensive fundamental metrics")
    
    # Show shared insights history if any exist
    if 'shared_insights' in st.session_state and st.session_state.shared_insights:
        with st.expander(f"📋 Shared Insights History ({len(st.session_state.shared_insights)})", expanded=False):
            for i, insight in enumerate(reversed(st.session_state.shared_insights[-5:]), 1):  # Show last 5
                col_hist1, col_hist2, col_hist3 = st.columns([2, 1, 1])
                with col_hist1:
                    symbol_display = insight.get('symbol', 'N/A')
                    if symbol_display == '***':
                        symbol_display = 'Anonymous'
                    st.markdown(f"**{symbol_display}** - {insight.get('recommendation', 'N/A')}")
                with col_hist2:
                    privacy_icon = {"public": "🌐", "anonymized": "🔒", "private": "🔐"}.get(insight.get('privacy_level'), "📊")
                    st.markdown(f"{privacy_icon} {insight.get('privacy_level', 'N/A').title()}")
                with col_hist3:
                    timestamp = insight.get('timestamp', '')[:16].replace('T', ' ')
                    st.markdown(f"🕒 {timestamp}")
            
            if len(st.session_state.shared_insights) > 5:
                st.caption(f"Showing latest 5 of {len(st.session_state.shared_insights)} total insights")
    
    st.markdown("---")
    
    # Market and analysis mode selection
    col_market, col_mode = st.columns([1, 2])
    
    with col_market:
        market_selection = st.selectbox(
            "Select Market:",
            ["US Stocks", "Indian Stocks"],
            help="Choose between US stocks (USD) or Indian stocks (INR)"
        )
        market = "India" if market_selection == "Indian Stocks" else "US"
    
    with col_mode:
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["Single Stock Analysis", "Bulk Stock Analysis (Excel Export)"],
            horizontal=True
        )
    
    if analysis_mode == "Single Stock Analysis":
        # Create input section for single stock
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            if market == "India":
                placeholder_text = "e.g., RELIANCE, TCS, INFY"
                help_text = "Enter Indian stock symbols (e.g., RELIANCE for Reliance Industries, TCS for Tata Consultancy Services, INFY for Infosys)"
                default_symbol = "RELIANCE"
            else:
                placeholder_text = "e.g., AAPL, GOOGL, TSLA"
                help_text = "Enter US stock symbols (e.g., AAPL for Apple, GOOGL for Google, TSLA for Tesla)"
                default_symbol = "AAPL"
            
            symbol = st.text_input(
                f"Enter {market_selection} Symbol:",
                value=default_symbol,
                placeholder=placeholder_text,
                help=help_text
            )
        
        with col2:
            period_options = {
                "1 Month": "1mo",
                "3 Months": "3mo", 
                "6 Months": "6mo",
                "1 Year": "1y",
                "2 Years": "2y",
                "5 Years": "5y",
                "10 Years": "10y",
                "Maximum": "max"
            }
            
            selected_period = st.selectbox(
                "Select Time Period:",
                options=list(period_options.keys()),
                index=3,  # Default to "1 Year"
                help="Choose the time period for historical data analysis"
            )
            
            period_code = period_options[selected_period]
        
        with col3:
            # Auto-refresh toggle
            auto_refresh = st.checkbox(
                "🔄 Auto-refresh (10 min)",
                value=False,
                help="Automatically update data every 10 minutes during market hours"
            )
            
            analyze_button = st.button("Generate Chart", type="primary")
        
        # Create sub-tabs for organized content when stock data is available
        if symbol:
            # Pre-fetch data to determine if tabs should be shown
            with st.spinner(f'Fetching data for {symbol.upper()}...'):
                data, ticker_info, ticker_obj = fetch_stock_data(symbol.upper(), period=period_code, market=market)
            
            if data is not None and ticker_info is not None and ticker_obj is not None and not data.empty:
                # Create sub-tabs for better organization
                tab_price, tab_charts, tab_sentiment = st.tabs([
                    "📊 Price Action & Metrics", 
                    "📈 Technical Charts", 
                    "📰 News Sentiment Analysis"
                ])
                
                # Calculate all technical indicators once
                ma_50 = calculate_moving_average(data, window=50)
                ma_200 = calculate_moving_average(data, window=200)
                macd_line, signal_line, histogram = calculate_macd(data)
                rsi = calculate_rsi(data)
                cmf = calculate_chaikin_money_flow(data)
                support_level, resistance_level = calculate_support_resistance(data)
                
                with tab_price:
                    display_price_action_tab(symbol, data, ticker_info, ticker_obj, ma_50, ma_200, rsi, support_level, resistance_level, selected_period, market, auto_refresh)
                
                with tab_charts:
                    display_technical_charts_tab(symbol, data, ma_50, ma_200, macd_line, signal_line, histogram, rsi, cmf, selected_period, market)
                
                with tab_sentiment:
                    display_news_sentiment_analysis(symbol)
            
            else:
                st.error(f"""
                ❌ **Unable to fetch data for symbol '{symbol}'**
                
                Please check that:
                - The stock symbol is valid and correctly spelled
                - The stock is publicly traded
                - You have an internet connection
                
                **Examples of valid symbols:** AAPL, GOOGL, MSFT, TSLA, AMZN
                """)
        else:
            st.info("👆 Enter a stock symbol above to begin analysis")
    
    else:
        # Bulk analysis interface
        st.markdown("### 📊 Bulk Stock Analysis")
        st.markdown("Enter multiple stock symbols to generate a comprehensive Excel report with key metrics for all stocks.")
        
        # Initialize session state for saved lists
        if 'saved_stock_lists' not in st.session_state:
            st.session_state.saved_stock_lists = {}
        
        # Saved lists management
        st.markdown("#### 📂 Saved Stock Lists")
        col_saved1, col_saved2, col_saved3 = st.columns([2, 1, 1])
        
        with col_saved1:
            if st.session_state.saved_stock_lists:
                selected_saved_list = st.selectbox(
                    "Load a saved list:",
                    options=[""] + list(st.session_state.saved_stock_lists.keys()),
                    help="Select a previously saved stock list"
                )
            else:
                st.info("No saved lists yet. Create one below!")
                selected_saved_list = ""
        
        with col_saved2:
            if selected_saved_list:
                if st.button("📥 Load List", help="Load the selected stock list"):
                    st.session_state.bulk_stock_input = st.session_state.saved_stock_lists[selected_saved_list]
                    st.success(f"Loaded '{selected_saved_list}'")
                    st.rerun()
        
        with col_saved3:
            if selected_saved_list:
                if st.button("🗑️ Delete", help="Delete the selected list"):
                    del st.session_state.saved_stock_lists[selected_saved_list]
                    st.success(f"Deleted '{selected_saved_list}'")
                    st.rerun()
        
        st.divider()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if market == "India":
                bulk_placeholder = "RELIANCE\nTCS\nINFY\nHDFC\nICICIBANK\n\nOr: RELIANCE, TCS, INFY, HDFC, ICICIBANK"
                bulk_help = "Enter Indian stock symbols separated by commas or new lines"
            else:
                bulk_placeholder = "AAPL\nMSFT\nGOOGL\nTSLA\nAMZN\n\nOr: AAPL, MSFT, GOOGL, TSLA, AMZN"
                bulk_help = "Enter US stock symbols separated by commas or new lines"
            
            # Use session state for persistent input
            if 'bulk_stock_input' not in st.session_state:
                st.session_state.bulk_stock_input = ""
            
            stock_list = st.text_area(
                f"Enter {market_selection} Symbols (one per line or comma-separated)",
                value=st.session_state.bulk_stock_input,
                placeholder=bulk_placeholder,
                height=150,
                help=bulk_help,
                key="stock_list_input"
            )
            
            # Update session state when input changes
            if stock_list != st.session_state.bulk_stock_input:
                st.session_state.bulk_stock_input = stock_list
            
            # Save list functionality
            col_save1, col_save2 = st.columns([2, 1])
            with col_save1:
                list_name = st.text_input(
                    "Save this list as:",
                    placeholder="e.g., Tech Stocks, Banking Stocks, My Portfolio",
                    help="Enter a name to save the current stock list for future use"
                )
            with col_save2:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                if st.button("💾 Save List", disabled=not (stock_list.strip() and list_name.strip())):
                    if list_name.strip() and stock_list.strip():
                        st.session_state.saved_stock_lists[list_name.strip()] = stock_list.strip()
                        st.success(f"Saved list '{list_name.strip()}'!")
                        st.rerun()
        
        with col2:
            # Period selection for bulk
            period_options = {
                "1 Month": "1mo",
                "3 Months": "3mo", 
                "6 Months": "6mo",
                "1 Year": "1y",
                "2 Years": "2y",
                "5 Years": "5y",
                "10 Years": "10y",
                "Maximum": "max"
            }
            
            bulk_selected_period = st.selectbox(
                "Analysis Period",
                options=list(period_options.keys()),
                index=3,  # Default to 1 Year
                help="Select the time period for technical analysis",
                key="bulk_period"
            )
            
            bulk_period_code = period_options[bulk_selected_period]
            
            generate_excel_button = st.button("📋 Generate Excel Report", type="primary")
        
        # Process bulk analysis
        if generate_excel_button and stock_list.strip():
            # Parse stock symbols
            symbols = []
            if ',' in stock_list:
                symbols = [s.strip().upper() for s in stock_list.split(',') if s.strip()]
            else:
                symbols = [s.strip().upper() for s in stock_list.split('\n') if s.strip()]
            
            if symbols:
                st.info(f"Analyzing {len(symbols)} stocks: {', '.join(symbols)}")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_metrics = []
                
                for i, symbol in enumerate(symbols):
                    status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
                    
                    metrics = get_stock_metrics(symbol, bulk_period_code, market)
                    all_metrics.append(metrics)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(symbols))
                
                status_text.text("Generating Excel report...")
                
                # Create Excel report
                excel_file = create_excel_report(all_metrics, bulk_selected_period)
                
                # Success message and download
                st.success(f"Analysis complete! Generated report for {len(symbols)} stocks.")
                
                # Display summary
                successful = len([m for m in all_metrics if 'Error' not in m or m.get('Error') == ''])
                failed = len(all_metrics) - successful
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Stocks", len(symbols))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", failed)
                
                # Download button
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_file,
                    file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Show preview of data
                if all_metrics:
                    st.markdown("### Preview of Generated Data")
                    preview_df = pd.DataFrame(all_metrics)
                    st.dataframe(preview_df, use_container_width=True)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            
            else:
                st.error("Please enter at least one valid stock symbol.")
        
        # Set default values for single stock mode (when in bulk mode)
        symbol = ""
        analyze_button = False
        selected_period = "1 Year"
        period_code = "1y"
        auto_refresh = False  # No auto-refresh in bulk mode
    
    # Auto-refresh functionality (only for single stock mode)
    if analysis_mode == "Single Stock Analysis" and auto_refresh and symbol:
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


def gurufocus_tab():
    """Advanced analysis tab content"""
    st.markdown("### Professional institutional-grade financial analysis")
    st.markdown("Advanced earnings performance analysis with up to 8 quarters of historical data")
    st.markdown("---")
    
    # Create input section
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        symbol_guru = st.text_input(
            "Enter Stock Symbol:",
            value="AAPL",
            placeholder="e.g., AAPL, GOOGL, TSLA",
            help="Enter US stock symbols for detailed earnings analysis",
            key="gurufocus_symbol"
        )
    
    with col2:
        quarters_selection = st.selectbox(
            "Historical Quarters:",
            ["4 Quarters", "6 Quarters", "8 Quarters"],
            index=2,
            help="Number of past quarters to analyze"
        )
        quarters_count = int(quarters_selection.split()[0])
    
    with col3:
        if st.button("🔍 Analyze Earnings", key="guru_analyze", type="primary"):
            st.session_state.guru_analyze_clicked = True
    
    # Analysis section
    if st.session_state.get('guru_analyze_clicked', False) and symbol_guru:
        with st.spinner(f"Analyzing {quarters_count} quarters of earnings data for {symbol_guru.upper()}..."):
            # Fetch extended earnings data
            data, info, ticker_obj = fetch_stock_data(symbol_guru.upper(), period="3y", market="US")
            
            if data is not None and ticker_obj is not None:
                # Get detailed earnings performance analysis
                earnings_analysis, quarters_found = get_detailed_earnings_performance_analysis(
                    ticker_obj, data, market="US", max_quarters=quarters_count
                )
                
                if earnings_analysis is not None and not earnings_analysis.empty:
                    # Display earnings analysis results
                    st.subheader(f"📊 Earnings Performance Analysis - {quarters_found} Quarter{'s' if quarters_found != 1 else ''}")
                    
                    # Show the analysis details (rest of the GuruFocus functionality remains unchanged)
                    st.dataframe(earnings_analysis, use_container_width=True)
                else:
                    st.warning(f"No earnings data available for {symbol_guru.upper()} in the selected period")
            else:
                st.error(f"Unable to fetch data for {symbol_guru.upper()}. Please verify the symbol is correct.")
    
    elif symbol_guru and not st.session_state.get('guru_analyze_clicked', False):
        st.info("👆 Click 'Analyze Earnings' to start the detailed analysis")


def display_price_action_tab(symbol, data, ticker_info, ticker_obj, ma_50, ma_200, rsi, support_level, resistance_level, selected_period, market, auto_refresh):
    """Display price action metrics and key financial data"""
    
    # Auto-refresh status display
    if auto_refresh:
        col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
        with col_status1:
            st.success(f"✅ Live tracking {symbol} - Updates every 10 minutes")
        with col_status2:
            st.metric("Refresh #", st.session_state.get('refresh_count', 0))
        with col_status3:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric("Last Update", current_time)
    
    # Company information and current price
    col1, col2 = st.columns([2, 1])
    
    with col1:
        company_name = ticker_info.get('longName', ticker_info.get('shortName', symbol))
        st.markdown(f"### 🏢 {company_name} ({symbol})")
        
        if 'sector' in ticker_info:
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"**Sector:** {ticker_info.get('sector', 'N/A')}")
            with col_info2:
                st.markdown(f"**Industry:** {ticker_info.get('industry', 'N/A')}")
    
    with col2:
        # Current price and change
        current_price = data['Close'].iloc[-1]
        previous_close = ticker_info.get('previousClose', data['Close'].iloc[-2] if len(data) > 1 else current_price)
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100 if previous_close != 0 else 0
        
        # Currency based on market
        currency = "₹" if market == "India" else "$"
        
        st.metric(
            label=f"Current Price ({currency})",
            value=f"{currency}{current_price:.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    st.markdown("---")
    
    # After-market information
    st.subheader("🕐 Extended Hours Trading")
    
    try:
        after_market_data = get_after_market_data(symbol, market)
        
        col_am1, col_am2, col_am3, col_am4 = st.columns(4)
        
        with col_am1:
            if after_market_data['pre_market_change'] != 'N/A':
                st.metric(
                    "Pre-Market Change",
                    after_market_data['pre_market_change'],
                    after_market_data['pre_market_change_percent']
                )
            else:
                st.metric("Pre-Market Change", "N/A")
        
        with col_am2:
            if after_market_data['post_market_change'] != 'N/A':
                st.metric(
                    "After-Hours Change", 
                    after_market_data['post_market_change'],
                    after_market_data['post_market_change_percent']
                )
            else:
                st.metric("After-Hours Change", "N/A")
        
        with col_am3:
            if after_market_data['regular_session_close'] != 'N/A':
                st.metric("Regular Session Close", after_market_data['regular_session_close'])
            else:
                st.metric("Regular Session Close", "N/A")
        
        with col_am4:
            # Market status indicator
            import datetime
            current_time = datetime.datetime.now()
            
            # Simple market hours check (9:30 AM - 4:00 PM ET for US markets)
            if market == "US":
                market_open_time = current_time.replace(hour=9, minute=30)
                market_close_time = current_time.replace(hour=16, minute=0)
                
                if market_open_time <= current_time <= market_close_time:
                    market_status = "🟢 Open"
                elif current_time < market_open_time:
                    market_status = "🟡 Pre-Market"
                else:
                    market_status = "🔴 After-Hours"
            else:
                market_status = "🔵 Active"
            
            st.metric("Market Status", market_status)
            
    except Exception as e:
        st.info("Extended hours data not available")
    
    st.markdown("---")
    
    # Key price metrics in organized columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 52-week analysis
        week_52_high = ticker_info.get('fiftyTwoWeekHigh', 0)
        week_52_low = ticker_info.get('fiftyTwoWeekLow', 0)
        
        if week_52_high and week_52_low:
            position_52w = ((current_price - week_52_low) / (week_52_high - week_52_low)) * 100
            st.metric("52W Position", f"{position_52w:.1f}%")
            st.caption(f"Range: {currency}{week_52_low:.2f} - {currency}{week_52_high:.2f}")
        else:
            st.metric("52W Position", "N/A")
    
    with col2:
        # Volume analysis
        avg_volume = ticker_info.get('averageVolume', 0)
        current_volume = data['Volume'].iloc[-1]
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
            st.caption(f"Avg: {avg_volume:,.0f}")
        else:
            st.metric("Volume Ratio", "N/A")
    
    with col3:
        # Market cap and beta
        market_cap = ticker_info.get('marketCap', 0)
        if market_cap:
            if market_cap >= 1e12:
                cap_display = f"{currency}{market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                cap_display = f"{currency}{market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                cap_display = f"{currency}{market_cap/1e6:.2f}M"
            else:
                cap_display = f"{currency}{market_cap:.0f}"
            st.metric("Market Cap", cap_display)
        else:
            st.metric("Market Cap", "N/A")
        
        beta = ticker_info.get('beta', 0)
        if beta:
            st.caption(f"Beta: {beta:.2f}")
    
    with col4:
        # RSI and trend
        current_rsi = rsi.iloc[-1] if not rsi.empty else 0
        rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
        st.metric("RSI", f"{current_rsi:.1f}")
        st.caption(rsi_status)
    
    # Moving average analysis
    st.markdown("---")
    st.subheader("📈 Moving Average Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ma_50_current = ma_50.iloc[-1] if not ma_50.empty else 0
        ma_50_change = ((current_price - ma_50_current) / ma_50_current * 100) if ma_50_current != 0 else 0
        st.metric("50-Day MA", f"{currency}{ma_50_current:.2f}", f"{ma_50_change:+.2f}%")
    
    with col2:
        ma_200_current = ma_200.iloc[-1] if not ma_200.empty else 0
        ma_200_change = ((current_price - ma_200_current) / ma_200_current * 100) if ma_200_current != 0 else 0
        st.metric("200-Day MA", f"{currency}{ma_200_current:.2f}", f"{ma_200_change:+.2f}%")
    
    with col3:
        # Trend analysis
        if ma_50_current > ma_200_current:
            trend = "🟢 Bullish"
        elif ma_50_current < ma_200_current:
            trend = "🔴 Bearish"
        else:
            trend = "🟡 Neutral"
        st.metric("Trend", trend)
    
    # Support and resistance levels
    st.markdown("---")
    st.subheader("🎯 Support & Resistance Levels")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Support Level", f"{currency}{support_level:.2f}")
    
    with col2:
        st.metric("Resistance Level", f"{currency}{resistance_level:.2f}")
    
    with col3:
        # Safe level low (CTP -12.5%)
        safe_low = current_price * 0.875
        st.metric("Safe Level Low", f"{currency}{safe_low:.2f}")
        st.caption("CTP -12.5%")
    
    with col4:
        # Safe level high (CTP +12.5%)
        safe_high = current_price * 1.125
        st.metric("Safe Level High", f"{currency}{safe_high:.2f}")
        st.caption("CTP +12.5%")
    
    # Earnings and Dividend Information
    st.markdown("---")
    st.subheader("📅 Earnings & Dividend Information")
    
    # Get earnings and dividend information
    earnings_info = get_earnings_info(ticker_obj, ticker_info)
    dividend_info = get_dividend_info(ticker_obj, ticker_info, market)
    
    # Estimate next dividend date
    next_dividend_info = estimate_next_dividend_date(ticker_obj, dividend_info)
    
    col_earnings1, col_earnings2, col_earnings3 = st.columns(3)
    
    with col_earnings1:
        if earnings_info['last_earnings_formatted'] != 'N/A':
            st.metric("Last Earnings Date", earnings_info['last_earnings_formatted'])
        else:
            st.metric("Last Earnings Date", "N/A")
    
    with col_earnings2:
        if earnings_info['next_earnings_formatted'] != 'N/A':
            st.metric("Next Earnings Date", earnings_info['next_earnings_formatted'])
            # Add estimated indicator if it's an estimate
            if earnings_info['next_earnings_formatted'].startswith('~'):
                st.caption("(Estimated)")
        else:
            st.metric("Next Earnings Date", "N/A")
    
    with col_earnings3:
        # Calculate days since last earnings or days until next earnings
        days_calculated = False
        
        # Try to show days since last earnings first
        if earnings_info['last_earnings'] is not None:
            try:
                days_since_last = (pd.Timestamp.now() - earnings_info['last_earnings']).days
                st.metric("Days Since Last", f"{days_since_last} days")
                days_calculated = True
            except:
                pass
        
        # If we couldn't calculate days since last, try days until next
        if not days_calculated and earnings_info['next_earnings'] is not None:
            try:
                days_until_next = (earnings_info['next_earnings'] - pd.Timestamp.now()).days
                if days_until_next > 0:
                    st.metric("Days Until Next", f"{days_until_next} days")
                    days_calculated = True
                else:
                    st.metric("Days Until Next", "Overdue")
                    days_calculated = True
            except:
                pass
        
        # If neither calculation worked, show N/A
        if not days_calculated:
            st.metric("Days Since/Until", "N/A")
    
    # Add dividend information section
    st.markdown("---")
    st.subheader("💰 Dividend Information")
    
    col_div1, col_div2, col_div3, col_div4 = st.columns(4)
    
    with col_div1:
        if dividend_info['last_dividend_formatted'] != 'N/A':
            # Extract just the date part for cleaner display
            if dividend_info['last_dividend_date'] is not None:
                last_div_date = dividend_info['last_dividend_date'].strftime('%Y-%m-%d')
                st.metric("Last Dividend Date", last_div_date)
            else:
                st.metric("Last Dividend Date", "N/A")
        else:
            st.metric("Last Dividend Date", "N/A")
    
    with col_div2:
        if dividend_info['last_dividend_amount'] > 0:
            currency = "₹" if market == "India" else "$"
            st.metric("Last Dividend Amount", f"{currency}{dividend_info['last_dividend_amount']:.2f}")
        else:
            st.metric("Last Dividend Amount", "N/A")
    
    with col_div3:
        if next_dividend_info['next_dividend_date'] != 'N/A':
            st.metric("Next Dividend Date", next_dividend_info['next_dividend_date'])
            if next_dividend_info['estimated']:
                st.caption("(Estimated)")
        else:
            st.metric("Next Dividend Date", "N/A")
    
    with col_div4:
        if dividend_info['dividend_yield'] != 'N/A':
            st.metric("Dividend Yield", dividend_info['dividend_yield'])
        else:
            st.metric("Dividend Yield", "N/A")
    
    # Fibonacci retracement analysis
    st.markdown("---")
    st.subheader("📐 Fibonacci Retracement Analysis")
    
    # Calculate Fibonacci levels
    fibonacci_data = calculate_fibonacci_retracement(data)
    if fibonacci_data:
        # Determine trend direction
        recent_high = data['High'].tail(50).max()
        recent_low = data['Low'].tail(50).min()
        
        if current_price > (recent_high + recent_low) / 2:
            trend_direction = "🟢 Uptrend"
            swing_range = f"Swing Low: {currency}{recent_low:.2f} → Swing High: {currency}{recent_high:.2f}"
        else:
            trend_direction = "🔴 Downtrend"
            swing_range = f"Swing High: {currency}{recent_high:.2f} → Swing Low: {currency}{recent_low:.2f}"
        
        col_fib1, col_fib2 = st.columns([1, 2])
        
        with col_fib1:
            st.info(f"""
            **Trend Analysis:**
            - Direction: {trend_direction}
            - Current Price: {currency}{current_price:.2f}
            
            {swing_range}
            """)
        
        with col_fib2:
            # Display Fibonacci levels in a clean format
            st.markdown("**Key Fibonacci Levels:**")
            for level, price, label in fibonacci_data:
                distance = abs(current_price - price)
                distance_pct = (distance / current_price) * 100
                proximity = "🎯 Close" if distance_pct < 2 else "📍 Moderate" if distance_pct < 5 else "📊 Far"
                
                # Color coding based on proximity
                if distance_pct < 2:
                    st.success(f"**{label}**: {currency}{price:.2f} ({proximity} - {distance_pct:.1f}% away)")
                elif distance_pct < 5:
                    st.warning(f"**{label}**: {currency}{price:.2f} ({proximity} - {distance_pct:.1f}% away)")
                else:
                    st.info(f"**{label}**: {currency}{price:.2f} ({proximity} - {distance_pct:.1f}% away)")
    else:
        st.info("Fibonacci analysis requires sufficient price history for calculation")
    
    # Dividend and earnings information
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Dividend Information")
        dividend_yield = ticker_info.get('dividendYield', 0)
        if dividend_yield and dividend_yield > 0:
            # Convert to percentage and ensure reasonable display
            dividend_yield_pct = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
            st.metric("Dividend Yield", f"{dividend_yield_pct:.2f}%")
            
            dividend_rate = ticker_info.get('dividendRate', 0)
            if dividend_rate:
                st.caption(f"Annual Rate: {currency}{dividend_rate:.2f}")
        else:
            st.info("No dividend information available")
    
    with col2:
        st.subheader("📅 Earnings Information")
        # Get earnings dates from ticker info
        try:
            # Try to get next earnings date from ticker info
            next_earnings = ticker_info.get('earningsDate', None)
            if next_earnings:
                if isinstance(next_earnings, list) and len(next_earnings) > 0:
                    next_earnings_date = next_earnings[0]
                else:
                    next_earnings_date = next_earnings
                st.metric("Next Earnings", pd.to_datetime(next_earnings_date).strftime('%Y-%m-%d'))
            else:
                st.info("Next earnings date not available")
        except:
            st.info("Earnings information not available")
    
    # Earnings Performance Analysis
    st.markdown("---")
    st.subheader("📊 Earnings Performance Analysis")
    
    try:
        earnings_analysis, quarters_found = get_earnings_performance_analysis(ticker_obj, data, market)
        
        if earnings_analysis is not None and not earnings_analysis.empty:
            st.markdown(f"""
            **Track how the stock performed after each earnings announcement ({quarters_found} quarters available):**
            - **Overnight Change**: Price movement from close before earnings to open after earnings
            - **Week Performance**: Total change from pre-earnings close to end of week (5 trading days)
            """)
            
            # Display earnings analysis
            st.dataframe(earnings_analysis, use_container_width=True)
            
            # Summary statistics
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            
            try:
                # Calculate summary stats
                overnight_changes = [float(x.replace('%', '').replace('+', '')) for x in earnings_analysis['Overnight Change (%)'] if x != 'N/A']
                week_changes = [float(x.replace('%', '').replace('+', '')) for x in earnings_analysis['Week Performance (%)'] if x != 'N/A']
                
                if overnight_changes:
                    with col_stats1:
                        avg_overnight = sum(overnight_changes) / len(overnight_changes)
                        st.metric("Avg Overnight Change", f"{avg_overnight:+.2f}%")
                    
                    with col_stats2:
                        positive_overnight = sum(1 for x in overnight_changes if x > 0)
                        success_rate_overnight = (positive_overnight / len(overnight_changes)) * 100
                        st.metric("Overnight Success Rate", f"{success_rate_overnight:.0f}%")
                
                if week_changes:
                    with col_stats3:
                        avg_week = sum(week_changes) / len(week_changes)
                        st.metric("Avg Week Performance", f"{avg_week:+.2f}%")
                    
                    with col_stats4:
                        positive_week = sum(1 for x in week_changes if x > 0)
                        success_rate_week = (positive_week / len(week_changes)) * 100
                        st.metric("Week Success Rate", f"{success_rate_week:.0f}%")
            except:
                pass  # Skip summary stats if parsing fails
        else:
            st.info("No recent earnings data available for performance analysis")
    except Exception as e:
        st.info(f"Earnings performance analysis not available")


def display_technical_charts_tab(symbol, data, ma_50, ma_200, macd_line, signal_line, histogram, rsi, cmf, selected_period, market):
    """Display technical analysis charts"""
    
    st.subheader(f"📈 Technical Charts for {symbol}")
    
    # Technical indicators
    st.markdown("#### 📊 Price Chart with Moving Averages")
    
    # Currency based on market
    currency = "₹" if market == "India" else "$"
    
    # Main price chart with moving averages
    price_fig = go.Figure()
    
    # Add candlestick chart
    price_fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f'{symbol} Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Add moving averages
    if not ma_50.empty:
        price_fig.add_trace(go.Scatter(
            x=data.index,
            y=ma_50,
            mode='lines',
            name='50-Day MA',
            line=dict(color='blue', width=2)
        ))
    
    if not ma_200.empty:
        price_fig.add_trace(go.Scatter(
            x=data.index,
            y=ma_200,
            mode='lines',
            name='200-Day MA',
            line=dict(color='red', width=2)
        ))
    
    price_fig.update_layout(
        title=f'{symbol} Price Chart with Moving Averages',
        xaxis_title='Date',
        yaxis_title=f'Price ({currency})',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(price_fig, use_container_width=True)
    create_export_buttons(price_fig, "Price_Chart", symbol)
    
    # Technical indicators in separate charts
    col1, col2 = st.columns(2)
    
    with col1:
        # MACD Chart
        st.markdown("#### 📊 MACD Analysis")
        
        macd_fig = go.Figure()
        
        if not macd_line.empty and not signal_line.empty and not histogram.empty:
            macd_fig.add_trace(go.Scatter(
                x=data.index,
                y=macd_line,
                mode='lines',
                name='MACD Line',
                line=dict(color='blue')
            ))
            
            macd_fig.add_trace(go.Scatter(
                x=data.index,
                y=signal_line,
                mode='lines',
                name='Signal Line',
                line=dict(color='red')
            ))
            
            macd_fig.add_trace(go.Bar(
                x=data.index,
                y=histogram,
                name='Histogram',
                marker_color='gray',
                opacity=0.7
            ))
        
        macd_fig.update_layout(
            title='MACD Indicator',
            xaxis_title='Date',
            yaxis_title='MACD',
            height=400
        )
        
        st.plotly_chart(macd_fig, use_container_width=True)
        create_export_buttons(macd_fig, "MACD_Analysis", symbol)
    
    with col2:
        # RSI Chart
        st.markdown("#### 📊 RSI Analysis")
        
        rsi_fig = go.Figure()
        
        if not rsi.empty:
            rsi_fig.add_trace(go.Scatter(
                x=data.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ))
            
            # Add overbought/oversold lines
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        
        rsi_fig.update_layout(
            title='RSI (Relative Strength Index)',
            xaxis_title='Date',
            yaxis_title='RSI',
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(rsi_fig, use_container_width=True)
        create_export_buttons(rsi_fig, "RSI_Analysis", symbol)


def display_news_sentiment_analysis(symbol):
    """Display AI-powered news sentiment analysis"""
    
    st.subheader(f"📰 AI-Powered News Sentiment Analysis for {symbol}")
    st.markdown("Analyze financial news sentiment using advanced AI to gauge market sentiment and potential impact on stock performance.")
    
    # Create analyze button
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        **What this analysis provides:**
        - Sentiment scoring (Bullish/Bearish/Neutral)
        - Investment impact assessment
        - Key themes identification
        - Risk factor analysis
        - Confidence levels for each analysis
        """)
    
    with col2:
        if st.button("🤖 Analyze News Sentiment", type="primary", key=f"sentiment_{symbol}"):
            st.session_state[f'analyze_sentiment_{symbol}'] = True
    
    # Perform sentiment analysis if button clicked
    if st.session_state.get(f'analyze_sentiment_{symbol}', False):
        
        # Check if OpenAI API key is available
        import os
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            st.error("""
            🔑 **OpenAI API Key Required**
            
            To use AI-powered sentiment analysis, please add your OpenAI API key to the environment variables.
            
            **Steps:**
            1. Go to your Replit project secrets
            2. Add a new secret with key: `OPENAI_API_KEY`
            3. Add your OpenAI API key as the value
            4. Restart the application
            """)
            return
        
        # Use the enhanced news sentiment analyzer with multiple sources
        try:
            from news_sentiment_analyzer import run_sentiment_analysis
            run_sentiment_analysis(symbol)
                    
        except Exception as e:
            st.error(f"Error loading sentiment analysis: {str(e)}")
            st.info("Please ensure all required dependencies are installed and OpenAI API key is configured correctly.")


if __name__ == "__main__":
    main()
