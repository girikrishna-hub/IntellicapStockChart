# Stock Technical Analysis Application

## Overview

This is a comprehensive Streamlit-based web application for advanced technical analysis of stock price data. The application provides two analysis modes: single stock analysis with interactive charts and bulk stock analysis with Excel export capabilities. Features include multiple technical indicators (50-day and 200-day moving averages, MACD, RSI, Chaikin Money Flow), comprehensive financial metrics (52-week positions, earnings data, dividend information), support/resistance levels, professional Excel report generation for portfolio analysis, and saved stock list management for repeat bulk analysis. Users can analyze individual stocks with detailed visualizations, export comprehensive data for multiple stocks simultaneously, and save frequently used stock lists for quick access.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **UI Components**: Built-in Streamlit widgets for user input and data display
- **Visualization**: Plotly for interactive charting capabilities
- **Layout**: Wide layout configuration for better chart visibility

### Data Processing Layer
- **Data Source**: Yahoo Finance API integration via yfinance library
- **Data Structure**: Pandas DataFrames for efficient data manipulation
- **Analytics**: NumPy for numerical computations and moving average calculations
- **Time Series Handling**: Built-in datetime operations for historical data processing

### Application Logic
- **Dual-Mode Architecture**: Single stock analysis with interactive visualizations and bulk analysis with Excel export
- **Modular Design**: Separate functions for data fetching, technical calculations, and metrics extraction
- **Error Handling**: Comprehensive try-catch blocks for robust API interaction and bulk processing
- **Data Validation**: Empty data checks and error tracking for data quality assurance
- **Period Configuration**: User-selectable time periods from 1 month to maximum available history (default 1-year)
- **Bulk Processing**: Progress tracking and comprehensive metrics export for multiple stocks
- **List Management**: Session-based storage for saved stock lists with load, save, and delete functionality

### Configuration Management
- **Page Settings**: Centralized Streamlit configuration for consistent UI
- **Default Parameters**: 50-day and 200-day moving averages, MACD (12,26,9), RSI (14-period), Chaikin Money Flow (20-period), and support/resistance levels (20-period) for comprehensive technical analysis
- **Financial Metrics**: 52-week high/low position analysis, earnings date tracking, performance since earnings, and comprehensive dividend information
- **Symbol Input**: Dynamic stock ticker symbol processing for single or bulk analysis
- **Export Capabilities**: Professional Excel report generation with formatted headers and summary sheets

## External Dependencies

### Financial Data Services
- **Yahoo Finance API**: Primary data source for historical stock prices and market data
- **yfinance Library**: Python wrapper for Yahoo Finance API integration

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