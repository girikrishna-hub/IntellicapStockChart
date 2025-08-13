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
    st.title("📈 Stock Moving Average Chart")
    st.markdown("Enter a stock symbol and select a time period to view its price chart with 50-day and 200-day moving averages.")
    
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
                
                # Display key metrics
                st.subheader(f"Key Metrics for {symbol}")
                display_key_metrics(data, symbol, ma_50, ma_200)
                
                # Create and display chart
                st.subheader(f"Price Chart with Moving Averages")
                fig = create_chart(data, symbol, ma_50, ma_200, selected_period)
                st.plotly_chart(fig, use_container_width=True)
                
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
                    
                    trend_text = "**Current Trend Analysis:**\n\n"
                    if trend_50:
                        trend_text += f"• {trend_50}\n"
                    if trend_200:
                        trend_text += f"• {trend_200}\n"
                    trend_text += "\n**Note:** This is for informational purposes only and should not be considered as investment advice."
                    
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
    - Charts are interactive - you can zoom, pan, and hover for detailed information
    - All data is real-time and reflects actual market conditions
    
    **Note:** For reliable moving average analysis, longer time periods (1 year or more) are recommended.
    
    **Disclaimer:** This tool is for educational and informational purposes only. It should not be considered as financial advice.
    """)

if __name__ == "__main__":
    main()
