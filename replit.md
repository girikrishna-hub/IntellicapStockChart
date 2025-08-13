# Stock Moving Average Chart

## Overview

This is a Streamlit-based web application for visualizing stock price data with both 50-day and 200-day moving averages. The application allows users to input stock ticker symbols and view interactive charts displaying historical stock prices alongside calculated short-term (50-day) and long-term (200-day) moving averages. It leverages financial data APIs to provide comprehensive real-time stock market analysis capabilities through an intuitive web interface.

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
- **Modular Design**: Separate functions for data fetching and moving average calculations
- **Error Handling**: Try-catch blocks for robust API interaction
- **Data Validation**: Empty data checks to ensure data quality
- **Period Configuration**: User-selectable time periods from 1 month to maximum available history (default 1-year)

### Configuration Management
- **Page Settings**: Centralized Streamlit configuration for consistent UI
- **Default Parameters**: 50-day and 200-day moving average windows for comprehensive trend analysis
- **Symbol Input**: Dynamic stock ticker symbol processing

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