# Stock Technical Analysis Application

## Overview

This project includes two comprehensive Streamlit-based web applications for advanced stock analysis:

**1. Fundamental Analysis Tab (app.py):** Advanced technical analysis with interactive charts, real-time tracking, and bulk analysis. Features multiple technical indicators (50-day and 200-day moving averages, MACD, RSI, Chaikin Money Flow), Fibonacci retracement analysis for uptrends and downtrends, comprehensive financial metrics (52-week positions, earnings data, dividend information), support/resistance levels, professional Excel report generation, saved stock list management, and auto-refresh functionality for live market tracking.

**2. Advanced Analysis Tab:** Available as the second main tab within the application. Features detailed earnings performance analysis for up to 8 quarters and comprehensive institutional-grade financial metrics including valuation ratios, profitability analysis, financial strength indicators, and growth metrics.

**3. Landing Page (landing.html):** Static HTML main entry point with clear URL navigation allowing users to choose between Yahoo Finance and GuruFocus versions. Features detailed comparisons, copy-to-clipboard URL buttons, and seamless navigation between applications without WebSocket conflicts.

## Recent Changes (August 2025)

- **Fixed Dividend Display Bug:** Corrected dividend yield calculations that were showing inflated percentages (e.g., 45.00% instead of 0.45%)
- **Added Fibonacci Analysis:** Integrated comprehensive Fibonacci retracement analysis for both uptrends and downtrends with visual chart overlays and detailed metrics display
- **Added Beta Value:** Integrated stock beta calculation showing volatility relative to market
- **Added CTP Levels:** Added +/- 12.5% Close to Price target levels for support and resistance analysis (renamed to "Safe Level Low/High")
- **Added Earnings Performance Analysis:** Successfully integrated comprehensive earnings performance tracking showing stock movement after earnings announcements, with overnight changes and week performance analysis for up to 4 quarters
- **Resolved WebSocket Issues:** Replaced Streamlit landing page with static HTML to eliminate WebSocket connection errors
- **Improved Navigation:** Added clear URL display with copy buttons instead of confusing port change instructions
- **Streamlined Navigation:** App now opens directly on Yahoo Finance tab with GuruFocus available as secondary tab
- **Enhanced GuruFocus Tab:** Added detailed 8-quarter earnings performance analysis with advanced charts and institutional-grade financial metrics including valuation, profitability, financial strength, and growth analysis
- **Fixed Bulk Analysis Error:** Resolved auto-refresh variable error in bulk analysis mode by restricting auto-refresh to single stock analysis only
- **Implemented Social Sharing:** Added comprehensive social sharing feature with three privacy levels (public, anonymized, private) including Twitter, LinkedIn, email sharing options, and shared insights history tracking
- **Enhanced Social Sharing with Fibonacci Analysis:** Integrated comprehensive Fibonacci analysis information into social sharing including trend direction, swing ranges, closest retracement levels, and proximity alerts for support/resistance levels
- **Fixed Earnings Date Accuracy:** Resolved incorrect earnings dates showing outdated 2022 data by implementing 18-month date validation filters and prioritizing current year (2025) earnings data with proper chronological sorting to display accurate Q2 2025 earnings instead of old quarterly data
- **Enhanced Earnings Warning Display:** Improved earnings warning message visibility by separating warning text onto new line with smaller 11px font size and red color coding, preventing UI clutter while ensuring users can clearly see data quality alerts for outdated earnings information
- **Fixed Earnings Information Consistency:** Resolved inconsistent next earnings display by unifying data sources to use the same earnings_info logic throughout the application, ensuring next earnings dates appear consistently in both Price Action tab and Earnings Information sections
- **Implemented Chart Export Functionality:** Added comprehensive chart export capabilities supporting PNG and PDF formats for all charts including price charts, technical indicators (MACD, RSI, Chaikin), and earnings performance charts with high-resolution output and professional PDF formatting
- **Enhanced Multi-Source News Sentiment Analysis:** Upgraded sentiment analysis to include multiple news sources (Yahoo Finance RSS, Google News RSS, NewsAPI, Alpha Vantage) with user choice selection, comprehensive API key management, source-specific article tracking, and improved OpenAI GPT-4o powered analysis with detailed metrics including sentiment scoring, confidence levels, investment impact assessment, key themes identification, and risk factor analysis
- **Added Extended Hours Trading Information:** Restored comprehensive after-market data display in Price Action tab including pre-market changes, after-hours changes, regular session close prices, and real-time market status indicators (Open/Pre-Market/After-Hours) for complete trading day coverage
- **Renamed Main Navigation Tabs:** Updated tab names for better clarity - "Yahoo Finance Analysis" to "Fundamental Analysis" and "GuruFocus Analysis" to "Advanced Analysis" to better reflect the content and purpose of each section
- **Enhanced Multi-Source News Sentiment Analysis:** Upgraded sentiment analysis to support 4 different news sources (Yahoo Finance, Google News, NewsAPI, Alpha Vantage) with user selection interface, API key status indicators, comprehensive source management, and real-time article fetching from RSS feeds and professional APIs
- **Added Automatic Symbol Uppercase Conversion:** All stock symbol inputs now automatically convert to uppercase regardless of how users enter them, ensuring consistent formatting throughout the application for single analysis, bulk analysis, advanced analysis tabs, and sentiment analysis
- **Restored Institutional Financial Parameters:** Enhanced Advanced Analysis tab with comprehensive professional-grade financial metrics including valuation ratios (P/E, P/B, PEG, EV/EBITDA), profitability analysis (margins, ROE, ROA), financial strength indicators (debt ratios, liquidity ratios, cash flow), growth metrics (revenue/earnings growth rates), and analyst recommendations for institutional-level stock analysis
- **Added Cross-Tab Symbol Synchronization:** Stock symbols are now automatically synced between Fundamental Analysis and Advanced Analysis tabs, so entering a symbol in one tab automatically populates it in the other for seamless navigation and analysis workflow
- **Enhanced Advanced Analysis for Indian Stocks:** Advanced Analysis tab now supports both US and Indian markets with automatic market detection, proper currency formatting (₹ for Indian stocks, $ for US stocks), and automatic .NS suffix handling for Indian stock symbols to ensure proper data retrieval
- **Fixed News Sentiment Analysis for Indian Stocks:** Enhanced news sentiment analysis to properly detect Indian stocks and search for market-specific news using appropriate search terms (India NSE BSE) and region-specific RSS feeds to avoid mixing US and Indian stock results
- **Improved Earnings Date Accuracy:** Enhanced earnings data processing to ensure recent earnings dates are properly displayed, with robust fallback logic that prioritizes current year data while ensuring no earnings are missed due to overly restrictive date filtering. Added detection for potentially outdated earnings data and user warnings when data sources may be missing recent announcements. Fixed duplicate earnings dates issue where next earnings showed same date as last earnings by adding proper future date validation and estimation fallback logic. Implemented 80-day threshold for major stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, MSTR) to effectively detect when Yahoo Finance data lags behind actual earnings announcements
- **Enhanced Fibonacci Analysis with Next Two Levels:** Redesigned Fibonacci analysis to show exactly 2 levels above and 2 levels below current price with automatic switching between retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) when price is within range and extension levels (127.2%, 161.8%, 200%, 261.8%) when price is outside range. Added 3M/6M reference period toggle and clean display format showing distance calculations for optimal support/resistance level identification
- **Major UI Restructuring with Separate Earnings & Dividends Tab:** Restructured main navigation to include dedicated "Earnings & Dividends" tab alongside "Price Action", "Charts", and "News Sentiment" tabs to reduce content density per tab and improve user experience. Moved comprehensive earnings performance analysis, dividend information, next dividend estimation, and data source warnings to dedicated tab for better organization
- **Implemented Global CSS for Reduced Scrolling:** Added comprehensive CSS styling throughout application to reduce font sizes (metrics, dataframes, headers), minimize whitespace and margins, compact tab spacing, and optimize element padding to significantly reduce scrolling requirements across all tabs and improve user experience on smaller screens

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