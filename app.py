import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
import io
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill

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

def calculate_ctp_levels(current_price):
    """
    Calculate +/- 12.5% CTP (Close to Price) levels
    
    Args:
        current_price (float): Current stock price
    
    Returns:
        dict: Dictionary with upper and lower CTP levels
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

def calculate_fibonacci_retracements(data, period=50):
    """
    Calculate Fibonacci retracement levels for uptrends and downtrends
    
    Args:
        data (pd.DataFrame): Stock price data
        period (int): Period to look back for swing high/low (default 50)
    
    Returns:
        dict: Dictionary containing Fibonacci levels and trend direction
    """
    fibonacci_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    try:
        # Get recent data for analysis
        recent_data = data.tail(period)
        
        # Find swing high and low within the period
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        
        # Get the dates for swing high and low
        swing_high_date = recent_data['High'].idxmax()
        swing_low_date = recent_data['Low'].idxmin()
        
        # Determine trend direction based on recent price movement
        # Get the current price and compare with swing levels
        current_price = data['Close'].iloc[-1]
        
        # Calculate distance from swing points to determine trend
        distance_from_high = abs(current_price - swing_high) / swing_high
        distance_from_low = abs(current_price - swing_low) / swing_low
        
        # Also consider which came more recently
        recent_trend_period = min(10, len(recent_data) // 2)
        recent_data_slice = data.tail(recent_trend_period)
        price_change = (recent_data_slice['Close'].iloc[-1] - recent_data_slice['Close'].iloc[0]) / recent_data_slice['Close'].iloc[0]
        
        # Determine trend: if price is closer to high and trending up, it's uptrend
        if distance_from_high < distance_from_low and price_change >= 0:
            trend_direction = "uptrend"
            base_level = swing_low
            target_level = swing_high
        elif distance_from_low < distance_from_high and price_change < 0:
            trend_direction = "downtrend"
            base_level = swing_high
            target_level = swing_low
        elif swing_low_date > swing_high_date:
            # Recent swing low after swing high suggests downtrend
            trend_direction = "downtrend"
            base_level = swing_high
            target_level = swing_low
        else:
            # Recent swing high after swing low suggests uptrend
            trend_direction = "uptrend"
            base_level = swing_low
            target_level = swing_high
        
        # Calculate Fibonacci levels
        price_range = abs(target_level - base_level)
        fib_levels = {}
        
        if trend_direction == "uptrend":
            # For uptrend, retracements go down from the high
            for ratio in fibonacci_ratios:
                fib_levels[f"{ratio*100:.1f}%"] = swing_high - (price_range * ratio)
        else:
            # For downtrend, retracements go up from the low
            for ratio in fibonacci_ratios:
                fib_levels[f"{ratio*100:.1f}%"] = swing_low + (price_range * ratio)
        
        return {
            'trend_direction': trend_direction,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'swing_high_date': swing_high_date,
            'swing_low_date': swing_low_date,
            'fib_levels': fib_levels,
            'price_range': price_range
        }
        
    except Exception as e:
        return None

def get_earnings_info(ticker_obj, ticker_info):
    """
    Extract earnings information from ticker object and info with improved accuracy
    
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
    
    try:
        # Try to get earnings calendar first (most accurate for next earnings)
        try:
            calendar = ticker_obj.calendar
            if calendar is not None and not calendar.empty:
                # Get the most recent upcoming earnings date
                next_earnings = calendar.index[0]
                earnings_info['next_earnings'] = next_earnings
                earnings_info['next_earnings_formatted'] = next_earnings.strftime('%Y-%m-%d')
        except:
            pass
        
        # Try to get earnings history for last earnings
        try:
            earnings_dates = ticker_obj.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                # Get the most recent past earnings date
                current_date = pd.Timestamp.now()
                past_earnings = earnings_dates[earnings_dates.index <= current_date]
                if not past_earnings.empty:
                    last_earnings = past_earnings.index[-1]
                    earnings_info['last_earnings'] = last_earnings
                    earnings_info['last_earnings_formatted'] = last_earnings.strftime('%Y-%m-%d')
                
                # If we don't have next earnings from calendar, try from earnings_dates
                if earnings_info['next_earnings'] is None:
                    future_earnings = earnings_dates[earnings_dates.index > current_date]
                    if not future_earnings.empty:
                        next_earnings = future_earnings.index[0]
                        earnings_info['next_earnings'] = next_earnings
                        earnings_info['next_earnings_formatted'] = next_earnings.strftime('%Y-%m-%d')
        except:
            pass
        
        # Fallback to ticker_info if earnings_dates/calendar unavailable
        if earnings_info['last_earnings'] is None:
            try:
                if 'lastFiscalYearEnd' in ticker_info and ticker_info['lastFiscalYearEnd']:
                    last_earnings = pd.to_datetime(ticker_info['lastFiscalYearEnd'], unit='s')
                    earnings_info['last_earnings'] = last_earnings
                    earnings_info['last_earnings_formatted'] = last_earnings.strftime('%Y-%m-%d')
            except:
                pass
        
        # Fallback for next earnings
        if earnings_info['next_earnings'] is None:
            try:
                if 'earningsDate' in ticker_info and ticker_info['earningsDate']:
                    if isinstance(ticker_info['earningsDate'], list) and ticker_info['earningsDate']:
                        next_earnings = pd.to_datetime(ticker_info['earningsDate'][0], unit='s')
                        earnings_info['next_earnings'] = next_earnings
                        earnings_info['next_earnings_formatted'] = next_earnings.strftime('%Y-%m-%d')
                    elif not isinstance(ticker_info['earningsDate'], list):
                        next_earnings = pd.to_datetime(ticker_info['earningsDate'], unit='s')
                        earnings_info['next_earnings'] = next_earnings
                        earnings_info['next_earnings_formatted'] = next_earnings.strftime('%Y-%m-%d')
            except:
                pass
                
    except Exception as e:
        pass
    
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
        except:
            pass
        
        # Method 2: Try calendar if earnings_dates failed
        if earnings is None or earnings.empty:
            try:
                calendar = ticker_obj.calendar
                if calendar is not None and not calendar.empty:
                    earnings = calendar
                    earnings_count = len(calendar)
            except:
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
        
        # Get available earnings dates (up to 4, but show what's available)
        earnings_dates = earnings.index.tolist()
        earnings_dates.sort(reverse=True)  # Most recent first
        available_earnings = earnings_dates[:min(4, len(earnings_dates))]
        
        analysis_data = []
        successful_analyses = 0
        
        for earnings_date in available_earnings:
            try:
                # Convert to timezone-naive for comparison
                earnings_date_naive = earnings_date.tz_localize(None) if earnings_date.tz else earnings_date
                
                # Find the trading day before earnings (baseline price)
                pre_earnings_mask = data.index < earnings_date_naive
                if not pre_earnings_mask.any():
                    continue
                    
                pre_earnings_price = data[pre_earnings_mask]['Close'].iloc[-1]
                pre_earnings_date = data[pre_earnings_mask].index[-1]
                
                # Find the first trading day after earnings
                post_earnings_mask = data.index > earnings_date_naive
                if not post_earnings_mask.any():
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
                continue
        
        if analysis_data:
            df = pd.DataFrame(analysis_data)
            return df, successful_analyses
        else:
            return None, 0
            
    except Exception as e:
        return None, 0

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
                mask = data.index >= earnings_date
                if mask.any():
                    earnings_day_price = data[mask]['Close'].iloc[0]
                    earnings_performance = ((latest_price - earnings_day_price) / earnings_day_price) * 100
            except:
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
        
        # Beta value and CTP levels
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
            'CTP -12.5%': format_currency(ctp_levels['lower_ctp'], market) if ctp_levels['lower_ctp'] else "N/A",
            'CTP +12.5%': format_currency(ctp_levels['upper_ctp'], market) if ctp_levels['upper_ctp'] else "N/A",
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
            'CTP -12.5%': 'Error',
            'CTP +12.5%': 'Error',
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
    
    # New row for CTP levels and earnings info
    st.markdown("**🎯 Price Targets & Earnings Data**")
    col_ctp1, col_ctp2, col_ctp3, col_ctp4, col_ctp5 = st.columns(5)
    
    with col_ctp1:
        # CTP -12.5% level
        ctp_levels = calculate_ctp_levels(latest_price)
        st.metric(
            label="CTP -12.5%",
            value=format_currency(ctp_levels['lower_ctp'], market) if ctp_levels['lower_ctp'] else "N/A",
            help="Support target at -12.5% from current level"
        )
    
    with col_ctp2:
        # Current price for reference
        st.metric(
            label="Current Level",
            value=format_currency(latest_price, market),
            help="Reference point for CTP calculations"
        )
    
    with col_ctp3:
        # CTP +12.5% level (duplicate from above for better layout)
        st.metric(
            label="CTP +12.5%",
            value=format_currency(ctp_levels['upper_ctp'], market) if ctp_levels['upper_ctp'] else "N/A",
            help="Resistance target at +12.5% from current level"
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


def main():
    """
    Main application function
    """
    # Navigation bar
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("🏠 Main Menu", help="Return to landing page"):
            st.markdown('<meta http-equiv="refresh" content="0; url=http://localhost:3000">', unsafe_allow_html=True)
    with col_nav2:
        st.markdown("**📈 Yahoo Finance Version**")
    with col_nav3:
        if st.button("🏦 GuruFocus Version", help="Switch to professional data"):
            st.markdown('<meta http-equiv="refresh" content="0; url=http://localhost:5001">', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # App header
    st.title("📈 Stock Technical Analysis Tool")
    st.markdown("Get comprehensive technical analysis with moving averages, MACD, RSI, Chaikin Money Flow, earnings data, and dividend information for any stock symbol.")
    
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
            symbol = symbol.strip().upper()
            
            # Show loading spinner
            with st.spinner(f'Fetching {selected_period.lower()} data for {symbol}...'):
                # Fetch stock data and company info
                data, ticker_info, ticker_obj = fetch_stock_data(symbol, period=period_code, market=market)
            
            if data is not None and ticker_info is not None and ticker_obj is not None and not data.empty:
                # Calculate moving averages
                ma_50 = calculate_moving_average(data, window=50)
                ma_200 = calculate_moving_average(data, window=200)
                
                # Calculate technical indicators
                macd_line, signal_line, histogram = calculate_macd(data)
                rsi = calculate_rsi(data)
                cmf = calculate_chaikin_money_flow(data)
                
                # Calculate support and resistance levels
                support_level, resistance_level = calculate_support_resistance(data)
                
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
                
                # Display key metrics
                st.subheader(f"Key Metrics for {symbol}")
                display_key_metrics(data, symbol, ma_50, ma_200, rsi, ticker_info, ticker_obj, support_level, resistance_level, market)
                
                # Create and display price chart
                st.subheader(f"Price Chart with Moving Averages")
                fig = create_chart(data, symbol, ma_50, ma_200, selected_period, market)
                st.plotly_chart(fig, use_container_width=True)
                
                # Create and display MACD chart
                st.subheader(f"MACD Indicator")
                st.markdown("""
                **MACD (Moving Average Convergence Divergence)** is a trend-following momentum indicator that shows the relationship between two moving averages:
                - **MACD Line** (blue): 12-day EMA minus 26-day EMA
                - **Signal Line** (orange): 9-day EMA of the MACD line  
                - **Histogram** (bars): MACD line minus Signal line
                """)
                macd_fig = create_macd_chart(data, symbol, macd_line, signal_line, histogram, selected_period, market)
                st.plotly_chart(macd_fig, use_container_width=True)
                
                # Create and display RSI chart
                st.subheader(f"RSI Indicator")
                st.markdown("""
                **RSI (Relative Strength Index)** measures the speed and change of price movements:
                - **Above 70**: Potentially overbought (selling pressure may increase)
                - **Below 30**: Potentially oversold (buying opportunity may exist)
                - **Around 50**: Neutral momentum
                """)
                rsi_fig = create_rsi_chart(data, symbol, rsi, selected_period, market)
                st.plotly_chart(rsi_fig, use_container_width=True)
                
                # Create and display Chaikin Money Flow chart
                st.subheader(f"Chaikin Money Flow")
                st.markdown("""
                **Chaikin Money Flow (CMF)** measures the amount of Money Flow Volume over a period:
                - **Positive values**: Buying pressure (accumulation)
                - **Negative values**: Selling pressure (distribution)
                - **Values near zero**: Balanced buying/selling pressure
                """)
                cmf_fig = create_chaikin_chart(data, symbol, cmf, selected_period, market)
                st.plotly_chart(cmf_fig, use_container_width=True)
                
                # Earnings Performance Analysis
                earnings_analysis, quarters_found = get_earnings_performance_analysis(ticker_obj, data, market)
                st.subheader(f"📊 Earnings Performance Analysis ({quarters_found} Quarter{'s' if quarters_found != 1 else ''} Available)")
                st.markdown("""
                **Track how the stock performed after each earnings announcement:**
                - **Overnight Change**: Price movement from close before earnings to open after earnings
                - **Week Performance**: Total change from pre-earnings close to end of week (5 trading days)
                """)
                
                if earnings_analysis is not None and not earnings_analysis.empty:
                    # Display summary statistics
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                    
                    # Calculate summary stats
                    overnight_changes = [float(x.replace('%', '').replace('+', '')) for x in earnings_analysis['Overnight Change (%)'] if x != 'N/A']
                    week_changes = [float(x.replace('%', '').replace('+', '')) for x in earnings_analysis['Week Performance (%)'] if x != 'N/A']
                    
                    if overnight_changes:
                        avg_overnight = sum(overnight_changes) / len(overnight_changes)
                        positive_overnight = sum(1 for x in overnight_changes if x > 0)
                        
                    if week_changes:
                        avg_week = sum(week_changes) / len(week_changes)
                        positive_week = sum(1 for x in week_changes if x > 0)
                    
                    with col_stats1:
                        st.metric(
                            label="Avg Overnight Change",
                            value=f"{avg_overnight:+.1f}%" if overnight_changes else "N/A",
                            help="Average overnight change after earnings"
                        )
                    
                    with col_stats2:
                        st.metric(
                            label="Positive Overnight",
                            value=f"{positive_overnight}/{len(overnight_changes)}" if overnight_changes else "N/A",
                            help="Number of positive overnight reactions"
                        )
                    
                    with col_stats3:
                        st.metric(
                            label="Avg Week Performance",
                            value=f"{avg_week:+.1f}%" if week_changes else "N/A",
                            help="Average week performance after earnings"
                        )
                    
                    with col_stats4:
                        st.metric(
                            label="Positive Weeks",
                            value=f"{positive_week}/{len(week_changes)}" if week_changes else "N/A",
                            help="Number of positive week outcomes"
                        )
                    
                    # Display the detailed table
                    st.markdown("**Detailed Earnings Performance:**")
                    st.dataframe(
                        earnings_analysis,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("📋 Earnings performance data not available for this stock. This could be due to:")
                    st.markdown("""
                    - No earnings history found in available data sources
                    - Stock may be relatively new to the market
                    - Data availability issues with yfinance API
                    - Company may not report regular quarterly earnings
                    """)
                    
                    # Debug information for troubleshooting
                    with st.expander("🔍 Debug Information"):
                        try:
                            earnings_dates = ticker_obj.earnings_dates
                            calendar = ticker_obj.calendar
                            earnings_history = ticker_obj.earnings
                            
                            st.write(f"**Earnings Dates Available:** {len(earnings_dates) if earnings_dates is not None and not earnings_dates.empty else 0}")
                            st.write(f"**Calendar Available:** {len(calendar) if calendar is not None and not calendar.empty else 0}")
                            st.write(f"**Earnings History Available:** {len(earnings_history) if earnings_history is not None and not earnings_history.empty else 0}")
                            
                            if earnings_dates is not None and not earnings_dates.empty:
                                st.write("**Recent Earnings Dates:**")
                                st.write(earnings_dates.head())
                        except Exception as e:
                            st.write(f"Debug error: {str(e)}")
                
                # Additional information
                st.subheader("Chart Information")
                
                # Calculate statistics
                total_days = len(data)
                ma_50_days = len(ma_50.dropna()) if hasattr(ma_50, 'dropna') else len([x for x in ma_50 if not pd.isna(x)])
                ma_200_days = len(ma_200.dropna()) if hasattr(ma_200, 'dropna') else len([x for x in ma_200 if not pd.isna(x)])
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.info(f"""
                    **Data Period:** {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}
                    
                    **Total Trading Days:** {total_days}
                    
                    **50-Day MA Points:** {ma_50_days}
                    
                    **200-Day MA Points:** {ma_200_days}
                    """)
                    
                    # Add period-specific note
                    if selected_period in ["1 Month", "3 Months"]:
                        st.warning("⚠️ **Note:** Short time periods may not provide enough data for reliable 200-day moving average analysis. Consider using longer periods for better trend identification.")
                
                with info_col2:
                    # Trend analysis
                    current_price = data['Close'].iloc[-1]
                    current_ma_50 = ma_50.iloc[-1] if hasattr(ma_50, 'iloc') else ma_50[-1]
                    current_ma_200 = ma_200.iloc[-1] if hasattr(ma_200, 'iloc') else ma_200[-1]
                    
                    trend_50 = ""
                    trend_200 = ""
                    
                    if not pd.isna(current_ma_50):
                        if current_price > current_ma_50:
                            trend_50 = "Above 50-day MA (Short-term Bullish)"
                        else:
                            trend_50 = "Below 50-day MA (Short-term Bearish)"
                    
                    if not pd.isna(current_ma_200):
                        if current_price > current_ma_200:
                            trend_200 = "Above 200-day MA (Long-term Bullish)"
                        else:
                            trend_200 = "Below 200-day MA (Long-term Bearish)"
                    
                    # Technical indicators analysis
                    latest_macd = macd_line.iloc[-1] if hasattr(macd_line, 'iloc') else macd_line[-1]
                    latest_signal = signal_line.iloc[-1] if hasattr(signal_line, 'iloc') else signal_line[-1]
                    latest_rsi = rsi.iloc[-1] if hasattr(rsi, 'iloc') else rsi[-1]
                    latest_cmf = cmf.iloc[-1] if hasattr(cmf, 'iloc') else cmf[-1]
                    
                    macd_trend = ""
                    rsi_trend = ""
                    cmf_trend = ""
                    
                    if not pd.isna(latest_macd) and not pd.isna(latest_signal):
                        if latest_macd > latest_signal:
                            macd_trend = "MACD above Signal (Bullish momentum)"
                        else:
                            macd_trend = "MACD below Signal (Bearish momentum)"
                    
                    if not pd.isna(latest_rsi):
                        if latest_rsi > 70:
                            rsi_trend = f"RSI: {latest_rsi:.1f} (Overbought)"
                        elif latest_rsi < 30:
                            rsi_trend = f"RSI: {latest_rsi:.1f} (Oversold)"
                        else:
                            rsi_trend = f"RSI: {latest_rsi:.1f} (Neutral)"
                    
                    if not pd.isna(latest_cmf):
                        if latest_cmf > 0.1:
                            cmf_trend = f"CMF: {latest_cmf:.3f} (Strong Buying Pressure)"
                        elif latest_cmf < -0.1:
                            cmf_trend = f"CMF: {latest_cmf:.3f} (Strong Selling Pressure)"
                        else:
                            cmf_trend = f"CMF: {latest_cmf:.3f} (Balanced)"
                    
                    trend_text = "**Current Technical Analysis:**\n\n"
                    if trend_50:
                        trend_text += f"• {trend_50}\n"
                    if trend_200:
                        trend_text += f"• {trend_200}\n"
                    if macd_trend:
                        trend_text += f"• {macd_trend}\n"
                    if rsi_trend:
                        trend_text += f"• {rsi_trend}\n"
                    if cmf_trend:
                        trend_text += f"• {cmf_trend}\n"
                    trend_text += "\n**Note:** This is for educational purposes only and should not be considered as investment advice."
                    
                    st.success(trend_text)
                
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
            st.warning("Please enter a stock symbol to generate the chart.")
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    **About this application:**
    - Data is sourced from Yahoo Finance via the yfinance library
    - Supports both US stocks (USD) and Indian stocks (INR) with proper currency formatting
    - Two analysis modes: Single stock with charts or bulk analysis with Excel export
    - Choose from multiple time periods: 1 month to maximum available history
    - Comprehensive metrics include price position relative to 52-week range
    - Moving averages (50 & 200-day) show short and long-term trends
    - Support/resistance levels calculated from recent 20-day price action
    - Earnings data and performance tracking since last earnings announcement
    - Comprehensive dividend information including yield, payment dates, and projections
    - Multiple technical indicators: MACD, RSI, and Chaikin Money Flow
    - Bulk analysis generates downloadable Excel reports for multiple stocks
    - Charts are interactive - you can zoom, pan, and hover for detailed information
    - All data is real-time and reflects actual market conditions
    
    **Key Metrics Explained:**
    - **52-Week Position:** Shows how close the stock is trading to yearly highs/lows
    - **Support/Resistance:** Recent price levels that may act as floors/ceilings
    - **Earnings Performance:** Price movement since last earnings announcement
    - **Dividend Metrics:** Yield, payment dates, forward dividends, and payout ratios
    - **Technical Indicators:** MACD (momentum), RSI (overbought/oversold), CMF (money flow)
    
    **Note:** For reliable analysis, longer time periods (1 year or more) are recommended.
    
    **Disclaimer:** This tool is for educational and informational purposes only. It should not be considered as investment advice.
    """)

if __name__ == "__main__":
    main()
