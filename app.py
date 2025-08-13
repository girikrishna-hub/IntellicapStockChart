import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Stock 50-Day Moving Average Chart",
    page_icon="📈",
    layout="wide"
)

def fetch_stock_data(symbol, period="1y"):
    """
    Fetch stock data for the given symbol and period
    
    Args:
        symbol (str): Stock ticker symbol
        period (str): Time period for data retrieval
    
    Returns:
        pd.DataFrame: Stock data or None if error
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch historical data
        data = ticker.history(period=period)
        
        if data.empty:
            return None
            
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

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

def create_chart(data, symbol, ma_50, ma_200, period_label="1 Year"):
    """
    Create an interactive Plotly chart with stock price, 50-day and 200-day moving averages
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol for chart title
        ma_50 (pd.Series): 50-day moving average data
        ma_200 (pd.Series): 200-day moving average data
        period_label (str): Time period label for chart title
    
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    fig = go.Figure()
    
    # Add stock price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name=f'{symbol.upper()} Close Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'Price: $%{y:.2f}<br>' +
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
                      'MA(50): $%{y:.2f}<br>' +
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
                      'MA(200): $%{y:.2f}<br>' +
                      '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{symbol.upper()} Stock Price with 50-Day & 200-Day Moving Averages ({period_label})',
        xaxis_title='Date',
        yaxis_title='Price ($)',
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
    
    # Update y-axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        tickformat='$.2f'
    )
    
    return fig

def create_macd_chart(data, symbol, macd_line, signal_line, histogram, period_label="1 Year"):
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

def create_rsi_chart(data, symbol, rsi, period_label="1 Year"):
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

def create_chaikin_chart(data, symbol, cmf, period_label="1 Year"):
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

def display_key_metrics(data, symbol, ma_50, ma_200):
    """
    Display key metrics about the stock and moving averages
    
    Args:
        data (pd.DataFrame): Stock price data
        symbol (str): Stock symbol
        ma_50 (pd.Series): 50-day moving average data
        ma_200 (pd.Series): 200-day moving average data
    """
    # Get latest values
    latest_price = data['Close'].iloc[-1]
    latest_ma_50 = ma_50.iloc[-1]
    latest_ma_200 = ma_200.iloc[-1]
    
    # Calculate some metrics
    year_high = data['Close'].max()
    year_low = data['Close'].min()
    
    # Price vs MA comparison
    price_vs_ma_50 = ((latest_price - latest_ma_50) / latest_ma_50) * 100 if not pd.isna(latest_ma_50) else 0
    price_vs_ma_200 = ((latest_price - latest_ma_200) / latest_ma_200) * 100 if not pd.isna(latest_ma_200) else 0
    
    # Create columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${latest_price:.2f}",
            delta=f"{price_vs_ma_50:.1f}% vs MA(50)" if not pd.isna(latest_ma_50) else None
        )
    
    with col2:
        st.metric(
            label="50-Day MA",
            value=f"${latest_ma_50:.2f}" if not pd.isna(latest_ma_50) else "N/A"
        )
    
    with col3:
        st.metric(
            label="200-Day MA",
            value=f"${latest_ma_200:.2f}" if not pd.isna(latest_ma_200) else "N/A",
            delta=f"{price_vs_ma_200:.1f}% vs Price" if not pd.isna(latest_ma_200) else None
        )
    
    with col4:
        st.metric(
            label="52-Week High",
            value=f"${year_high:.2f}"
        )
    
    with col5:
        st.metric(
            label="52-Week Low",
            value=f"${year_low:.2f}"
        )

def main():
    """
    Main application function
    """
    # App header
    st.title("📈 Stock Technical Analysis")
    st.markdown("Enter a stock symbol and select a time period to view comprehensive technical analysis with moving averages and MACD indicator.")
    
    # Create input section
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol (e.g., AAPL, GOOGL, TSLA):",
            value="AAPL",
            placeholder="Enter a valid stock ticker symbol",
            help="Enter any valid stock ticker symbol (e.g., AAPL for Apple, GOOGL for Google, TSLA for Tesla)"
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
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        analyze_button = st.button("Generate Chart", type="primary")
    
    # Process the request when button is clicked or symbol is entered
    if analyze_button or symbol:
        if symbol:
            symbol = symbol.strip().upper()
            
            # Show loading spinner
            with st.spinner(f'Fetching {selected_period.lower()} data for {symbol}...'):
                # Fetch stock data
                data = fetch_stock_data(symbol, period=period_code)
            
            if data is not None and not data.empty:
                # Calculate moving averages
                ma_50 = calculate_moving_average(data, window=50)
                ma_200 = calculate_moving_average(data, window=200)
                
                # Calculate technical indicators
                macd_line, signal_line, histogram = calculate_macd(data)
                rsi = calculate_rsi(data)
                cmf = calculate_chaikin_money_flow(data)
                
                # Display key metrics
                st.subheader(f"Key Metrics for {symbol}")
                display_key_metrics(data, symbol, ma_50, ma_200)
                
                # Create and display price chart
                st.subheader(f"Price Chart with Moving Averages")
                fig = create_chart(data, symbol, ma_50, ma_200, selected_period)
                st.plotly_chart(fig, use_container_width=True)
                
                # Create and display MACD chart
                st.subheader(f"MACD Indicator")
                st.markdown("""
                **MACD (Moving Average Convergence Divergence)** is a trend-following momentum indicator that shows the relationship between two moving averages:
                - **MACD Line** (blue): 12-day EMA minus 26-day EMA
                - **Signal Line** (orange): 9-day EMA of the MACD line  
                - **Histogram** (bars): MACD line minus Signal line
                """)
                macd_fig = create_macd_chart(data, symbol, macd_line, signal_line, histogram, selected_period)
                st.plotly_chart(macd_fig, use_container_width=True)
                
                # Create and display RSI chart
                st.subheader(f"RSI Indicator")
                st.markdown("""
                **RSI (Relative Strength Index)** measures the speed and change of price movements:
                - **Above 70**: Potentially overbought (selling pressure may increase)
                - **Below 30**: Potentially oversold (buying opportunity may exist)
                - **Around 50**: Neutral momentum
                """)
                rsi_fig = create_rsi_chart(data, symbol, rsi, selected_period)
                st.plotly_chart(rsi_fig, use_container_width=True)
                
                # Create and display Chaikin Money Flow chart
                st.subheader(f"Chaikin Money Flow")
                st.markdown("""
                **Chaikin Money Flow (CMF)** measures the amount of Money Flow Volume over a period:
                - **Positive values**: Buying pressure (accumulation)
                - **Negative values**: Selling pressure (distribution)
                - **Values near zero**: Balanced buying/selling pressure
                """)
                cmf_fig = create_chaikin_chart(data, symbol, cmf, selected_period)
                st.plotly_chart(cmf_fig, use_container_width=True)
                
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
    - Choose from multiple time periods: 1 month to maximum available history
    - The 50-day moving average shows short-term trends (last 50 trading days)
    - The 200-day moving average shows long-term trends (last 200 trading days)
    - MACD indicator shows momentum and trend changes using exponential moving averages
    - RSI measures momentum and identifies overbought/oversold conditions
    - Chaikin Money Flow analyzes buying/selling pressure using price and volume
    - Charts are interactive - you can zoom, pan, and hover for detailed information
    - All data is real-time and reflects actual market conditions
    
    **Technical Indicators Guide:**
    - **MACD:** Crossovers may signal trend changes and momentum shifts
    - **RSI:** Values above 70 suggest overbought, below 30 suggest oversold conditions
    - **Chaikin Money Flow:** Positive values indicate accumulation, negative indicate distribution
    
    **Note:** For reliable analysis, longer time periods (1 year or more) are recommended.
    
    **Disclaimer:** This tool is for educational and informational purposes only. It should not be considered as investment advice.
    """)

if __name__ == "__main__":
    main()
