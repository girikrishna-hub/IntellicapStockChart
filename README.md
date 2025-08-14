# 📊 Dual Stock Analysis Platform

A comprehensive stock analysis platform offering two distinct data sources and analysis approaches for different investor needs.

## 🚀 Quick Start

**Main Landing Page**: `http://localhost:3000`
- Interactive comparison interface
- Choose between Yahoo Finance and GuruFocus
- Quick start guides and feature comparisons

**Direct Access:**

1. **Yahoo Finance Version** (Free): `http://localhost:5000`
   - Real-time stock data and technical indicators
   - No API key required
   - Perfect for individual investors

2. **GuruFocus Version** (Professional): `http://localhost:5001`
   - Institutional-grade financial data
   - Requires GuruFocus API key
   - Advanced fundamentals and valuations

## 📈 Yahoo Finance Version Features

### Technical Analysis
- 50-day and 200-day moving averages
- MACD (12,26,9) momentum indicator
- RSI (14-period) overbought/oversold signals
- Chaikin Money Flow volume analysis
- Support and resistance levels

### Financial Metrics
- 52-week high/low position analysis
- Earnings date tracking and performance
- Comprehensive dividend information
- Real-time price and volume data

### Advanced Features
- Auto-refresh every 10 minutes for live tracking
- Multi-market support (US and Indian stocks)
- Bulk analysis with Excel export
- Saved stock lists for portfolio tracking

## 🏦 GuruFocus Version Features

### Professional Data
- Company profiles with detailed metrics
- Historical fundamentals and financials
- Comprehensive dividend history
- Valuation metrics and ratios

### Institutional Features
- API-driven data access
- Professional Excel reporting
- Bulk analysis for portfolio management
- Error handling and data validation

### Data Categories
- **Stock Profiles**: Company details and current metrics
- **Fundamentals**: Historical financial statements
- **Valuations**: Deep valuation metric datasets
- **Dividends**: Complete dividend payment history

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8+
- Streamlit
- Required packages (automatically installed)

### Yahoo Finance Version
1. No setup required - runs immediately
2. Access at `http://localhost:5000`
3. Enter any stock symbol (AAPL, MSFT, TSLA, etc.)

### GuruFocus Version
1. Sign up for GuruFocus account
2. Subscribe to data plan with API access
3. Get API key from GuruFocus download page
4. Access at `http://localhost:5001`
5. Enter API key in sidebar
6. Start analyzing with professional data

## 💡 Usage Examples

### Single Stock Analysis
- **Yahoo Finance**: Enter "AAPL" → Get technical charts with indicators
- **GuruFocus**: Enter "AAPL" + API key → Get fundamentals and valuations

### Bulk Analysis
- **Yahoo Finance**: Paste multiple symbols → Excel with technical metrics
- **GuruFocus**: Paste multiple symbols → Excel with fundamental analysis

### Auto-Refresh (Yahoo Finance)
1. Enter stock symbol
2. Check "Auto-refresh (10 min)"
3. Watch live updates during market hours

### Saved Lists (Both Versions)
1. Enter multiple stock symbols
2. Name your list (e.g., "Tech Portfolio")
3. Save for future analysis
4. Load saved lists anytime

## 📊 Data Comparison

| Feature | Yahoo Finance | GuruFocus |
|---------|---------------|-----------|
| **Cost** | Free | Paid subscription |
| **Data Quality** | Good | Institutional-grade |
| **Real-time** | Yes | Limited |
| **Technical Indicators** | Full suite | Basic |
| **Fundamentals** | Basic | Comprehensive |
| **Historical Data** | 5+ years | 30+ years |
| **API Reliability** | Good | Professional |
| **Global Markets** | Limited | Extensive |

## 🔍 When to Use Each Version

### Use Yahoo Finance Version When:
- Learning technical analysis
- Day trading or swing trading
- Need real-time price data
- Analyzing popular US/Indian stocks
- Budget-conscious investing

### Use GuruFocus Version When:
- Professional portfolio management
- Value investing research
- Need detailed fundamentals
- Analyzing small-cap or international stocks
- Require institutional-grade data accuracy

## 📁 File Structure

```
├── index.py              # Main landing page (Port 3000)
├── app.py                # Yahoo Finance version (Port 5000)
├── app_gurufocus.py      # GuruFocus version (Port 5001)
├── .streamlit/
│   ├── config_index.toml      # Landing page config
│   ├── config.toml            # Yahoo Finance config
│   └── config_gurufocus.toml  # GuruFocus config
├── replit.md             # Project documentation
└── README.md             # This file
```

## 🚨 Error Handling

### Common Issues
- **"Symbol not found"**: Check ticker symbol spelling
- **"API error"**: Check internet connection or API limits
- **"Invalid API key"**: Verify GuruFocus subscription and key

### Rate Limiting
- Yahoo Finance: Built-in delays between requests
- GuruFocus: Professional API with higher limits

## 🔐 Security

- API keys stored securely in session state
- No data persistence between sessions
- HTTPS enforced for API communications
- No sensitive data logged or cached

## 📈 Performance Tips

1. **Bulk Analysis**: Process 10-20 stocks at once for best performance
2. **Auto-refresh**: Use only during market hours to conserve resources
3. **Saved Lists**: Create themed lists for faster analysis
4. **API Limits**: Monitor GuruFocus usage to avoid overages

## 🆘 Support

### Yahoo Finance Issues
- Check symbol format (US: AAPL, India: RELIANCE.NS)
- Verify internet connection
- Try alternative symbols

### GuruFocus Issues
- Verify API key from download page
- Check subscription status
- Contact GuruFocus support for API issues

## 🎯 Next Steps

1. **Start with Yahoo Finance** to learn the interface
2. **Upgrade to GuruFocus** for professional analysis
3. **Create saved lists** for your portfolios
4. **Use bulk analysis** for portfolio reviews
5. **Enable auto-refresh** for live market tracking

---

**Happy Investing! 📈**

Both versions provide powerful stock analysis capabilities tailored to different investor needs and budgets. Choose the version that best fits your investment strategy and data requirements.