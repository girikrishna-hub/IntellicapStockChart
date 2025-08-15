# Stock Technical Analysis Application

## Overview

This project includes two comprehensive Streamlit-based web applications for advanced stock analysis:

**1. Yahoo Finance Version (app.py):** Advanced technical analysis with interactive charts, real-time tracking, and bulk analysis. Features multiple technical indicators (50-day and 200-day moving averages, MACD, RSI, Chaikin Money Flow), Fibonacci retracement analysis for uptrends and downtrends, comprehensive financial metrics (52-week positions, earnings data, dividend information), support/resistance levels, professional Excel report generation, saved stock list management, and auto-refresh functionality for live market tracking.

**2. GuruFocus Version (app_gurufocus_minimal.py):** Currently disabled for deployment. Previously provided streamlined stock analysis using GuruFocus API for institutional-quality data.

**3. Landing Page (landing.html):** Static HTML main entry point with clear URL navigation allowing users to choose between Yahoo Finance and GuruFocus versions. Features detailed comparisons, copy-to-clipboard URL buttons, and seamless navigation between applications without WebSocket conflicts.

## Recent Changes (August 2025)

- **Fixed Dividend Display Bug:** Corrected dividend yield calculations that were showing inflated percentages (e.g., 45.00% instead of 0.45%)
- **Added Fibonacci Analysis:** Integrated comprehensive Fibonacci retracement analysis for both uptrends and downtrends with visual chart overlays and detailed metrics display
- **Added Beta Value:** Integrated stock beta calculation showing volatility relative to market
- **Added CTP Levels:** Added +/- 12.5% Close to Price target levels for support and resistance analysis
- **Added Earnings Performance Analysis:** Framework prepared for GuruFocus integration (Yahoo Finance lacks sufficient earnings performance data)
- **Resolved WebSocket Issues:** Replaced Streamlit landing page with static HTML to eliminate WebSocket connection errors
- **Improved Navigation:** Added clear URL display with copy buttons instead of confusing port change instructions
- **Streamlined GuruFocus:** Created minimal working version to avoid DataFrame processing hangs

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **UI Components**: Built-in Streamlit widgets for user input and data display
- **Visualization**: Plotly for interactive charting capabilities
- **Layout**: Wide layout configuration for better chart visibility
- **Triple Application Setup**: Three applications running on different ports (3000 for Static HTML Landing, 5000 for Yahoo Finance, 5001 for GuruFocus Minimal)
- **Landing Page**: Static HTML with JavaScript for WebSocket-free navigation and clear URL display

### Data Processing Layer
- **Yahoo Finance Version**: Yahoo Finance API integration via yfinance library for free real-time and historical data
- **GuruFocus Version**: GuruFocus REST API integration for institutional-grade financial data with authentication
- **Data Structure**: Pandas DataFrames for efficient data manipulation across both versions
- **Analytics**: NumPy for numerical computations and moving average calculations
- **Time Series Handling**: Built-in datetime operations for historical data processing
- **HTTP Requests**: Requests library for GuruFocus API communication with proper error handling

### Application Logic
- **Dual-Mode Architecture**: Single stock analysis with interactive visualizations and bulk analysis with Excel export
- **Modular Design**: Separate functions for data fetching, technical calculations, and metrics extraction
- **Error Handling**: Comprehensive try-catch blocks for robust API interaction and bulk processing
- **Data Validation**: Empty data checks and error tracking for data quality assurance
- **Period Configuration**: User-selectable time periods from 1 month to maximum available history (default 1-year)
- **Bulk Processing**: Progress tracking and comprehensive metrics export for multiple stocks
- **List Management**: Session-based storage for saved stock lists with load, save, and delete functionality
- **Auto-Refresh System**: Optional 10-minute automatic data updates for live market tracking with countdown timer and refresh status

### Configuration Management
- **Page Settings**: Centralized Streamlit configuration for consistent UI
- **Default Parameters**: 50-day and 200-day moving averages, MACD (12,26,9), RSI (14-period), Chaikin Money Flow (20-period), and support/resistance levels (20-period) for comprehensive technical analysis
- **Financial Metrics**: 52-week high/low position analysis, earnings date tracking, performance since earnings, and comprehensive dividend information
- **Symbol Input**: Dynamic stock ticker symbol processing for single or bulk analysis
- **Export Capabilities**: Professional Excel report generation with formatted headers and summary sheets

## External Dependencies

### Financial Data Services
- **Yahoo Finance API**: Free data source for historical stock prices and market data (app.py)
- **yfinance Library**: Python wrapper for Yahoo Finance API integration
- **GuruFocus API**: Professional institutional-grade financial data service (app_gurufocus.py)
- **Requests Library**: HTTP client for GuruFocus API communication with authentication headers

### Visualization and Analytics
- **Plotly**: Interactive charting library for creating responsive financial charts
- **Pandas**: Data manipulation and analysis framework
- **NumPy**: Numerical computing library for mathematical operations

### Web Framework
- **Streamlit**: Core web application framework for rapid prototyping and deployment
- **Python Standard Library**: Datetime module for time-based operations

### Runtime Environment
- **Python 3.x**: Primary programming language and runtime environment
- **Web Browser**: Client-side rendering for Streamlit applications