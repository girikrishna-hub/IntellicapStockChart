# Stock Technical Analysis Application

## Overview
This project provides two Streamlit-based web applications for advanced stock analysis. It offers fundamental analysis with interactive charts, real-time tracking, and bulk analysis, incorporating various technical indicators, Fibonacci retracement, financial metrics, and support/resistance levels. The application also includes an advanced analysis tab for detailed earnings performance and institutional-grade financial metrics. A static HTML landing page facilitates seamless navigation between the Yahoo Finance and GuruFocus versions, designed to avoid WebSocket conflicts. The overarching vision is to provide comprehensive, user-friendly tools for in-depth stock market insights, catering to both fundamental and technical analysis needs.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses Streamlit for rapid web development, leveraging its built-in widgets and Plotly for interactive charting. It employs a wide layout for optimal chart visibility. The setup involves three distinct applications: a static HTML landing page (port 3000) and two Streamlit applications for Yahoo Finance (port 5000) and GuruFocus (port 5001). The landing page utilizes JavaScript for navigation, avoiding WebSocket issues. UI elements are optimized with global CSS for reduced scrolling and offer a compact/standard view mode. PDF export functionality is included for comprehensive multi-page reports.

### Data Processing Layer
The Yahoo Finance version integrates with the `yfinance` library for free real-time and historical data. The GuruFocus version utilizes the GuruFocus REST API for institutional financial data, requiring authentication. Both versions extensively use Pandas DataFrames for data manipulation and NumPy for numerical computations, including moving averages. The `requests` library handles HTTP communications with proper error handling.

### Application Logic
The system supports both single stock analysis with interactive visualizations and bulk analysis with Excel export. It features a modular design with dedicated functions for data fetching, technical calculations, and metric extraction. Robust error handling is implemented for API interactions and bulk processing, along with data validation checks. Users can select time periods for data, and the application includes session-based storage for managing saved stock lists. An optional 10-minute auto-refresh system provides live market tracking. Key configurations include default parameters for various technical indicators and financial metrics, dynamic stock ticker processing, and professional Excel report generation. Stock symbols are automatically converted to uppercase and synchronized between fundamental and advanced analysis tabs. The Advanced Analysis tab supports both US and Indian markets with automatic detection, currency formatting, and `.NS` suffix handling for Indian stocks. News sentiment analysis also adapts to detect Indian stocks for market-specific news. Earnings data processing includes robust date validation, fallback logic for current year data, and warnings for potentially outdated information. Fibonacci analysis displays exactly two levels above and two levels below the current price, with automatic switching between retracement and extension levels based on price proximity. A major UI restructuring introduces a dedicated "Earnings & Dividends" tab for improved content organization. Enhanced compact view mode (Aug 2025) features dramatically reduced whitespace, grid layouts, CSS transform scaling for metric numbers, JavaScript number compacting, and optimized button layouts for minimal scrolling while maintaining readability.

## External Dependencies

### Financial Data Services
- **Yahoo Finance API**: Used via the `yfinance` Python library for historical and real-time stock market data.
- **GuruFocus API**: A professional service integrated for institutional-grade financial data.

### Visualization and Analytics
- **Plotly**: Utilized for creating interactive financial charts.
- **Pandas**: Essential for data manipulation and analysis.
- **NumPy**: Employed for numerical operations and computations.

### Web Framework
- **Streamlit**: The primary framework for building the web applications.
- **Python Standard Library**: Specifically the `datetime` module for handling time-based operations.

### Runtime Environment
- **Python 3.x**: The core programming language.
- **Web Browser**: Required for client-side rendering of the Streamlit applications.