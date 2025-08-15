import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

class EarningsDataProvider:
    """
    Comprehensive earnings data provider using multiple free sources
    """
    
    def __init__(self):
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        self.sources_available = {
            'yahoo_finance': True,
            'financial_modeling_prep': bool(self.fmp_api_key)
        }
    
    def get_earnings_performance_analysis(self, symbol: str, stock_data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], int]:
        """
        Get earnings performance analysis using the best available data source
        
        Args:
            symbol (str): Stock ticker symbol
            stock_data (pd.DataFrame): Historical stock price data
            
        Returns:
            Tuple[pd.DataFrame, int]: (Analysis dataframe, number of quarters found)
        """
        # Try Yahoo Finance enhanced method first
        earnings_data = self._get_yahoo_earnings_enhanced(symbol)
        if earnings_data:
            return self._analyze_earnings_performance(earnings_data, stock_data, symbol)
        
        # Try Financial Modeling Prep if API key available
        if self.fmp_api_key:
            earnings_data = self._get_fmp_earnings(symbol)
            if earnings_data:
                return self._analyze_earnings_performance(earnings_data, stock_data, symbol)
        
        return None, 0
    
    def _get_yahoo_earnings_enhanced(self, symbol: str) -> Optional[List[Dict]]:
        """
        Enhanced Yahoo Finance earnings data extraction
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Try multiple Yahoo Finance data sources
            earnings_sources = []
            
            # Method 1: Calendar
            try:
                calendar = ticker.calendar
                if calendar is not None and not calendar.empty:
                    for date in calendar.index:
                        earnings_sources.append({
                            'date': date,
                            'source': 'calendar',
                            'eps_estimate': calendar.loc[date].get('Earnings Estimate', None) if hasattr(calendar.loc[date], 'get') else None
                        })
            except:
                pass
            
            # Method 2: Earnings dates
            try:
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    for date in earnings_dates.index:
                        row = earnings_dates.loc[date]
                        earnings_sources.append({
                            'date': date,
                            'source': 'earnings_dates',
                            'eps_estimate': row.get('EPS Estimate', None) if hasattr(row, 'get') else None,
                            'eps_actual': row.get('Reported EPS', None) if hasattr(row, 'get') else None
                        })
            except:
                pass
            
            # Method 3: Info earnings date
            try:
                info = ticker.info
                if 'earningsDate' in info and info['earningsDate']:
                    earnings_date = pd.to_datetime(info['earningsDate'])
                    earnings_sources.append({
                        'date': earnings_date,
                        'source': 'info',
                        'eps_estimate': info.get('forwardEPS', None)
                    })
            except:
                pass
            
            # Remove duplicates and sort by date
            unique_earnings = {}
            for earning in earnings_sources:
                date_key = earning['date'].strftime('%Y-%m-%d')
                if date_key not in unique_earnings or earning['source'] == 'earnings_dates':
                    unique_earnings[date_key] = earning
            
            # Convert back to list and sort
            earnings_list = list(unique_earnings.values())
            earnings_list.sort(key=lambda x: x['date'], reverse=True)
            
            return earnings_list[:4] if earnings_list else None
            
        except Exception as e:
            print(f"Yahoo earnings enhanced error for {symbol}: {str(e)}")
            return None
    
    def _get_fmp_earnings(self, symbol: str) -> Optional[List[Dict]]:
        """
        Get earnings data from Financial Modeling Prep API
        """
        if not self.fmp_api_key:
            return None
        
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol.upper()}"
            params = {
                'limit': 8,
                'apikey': self.fmp_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                earnings_list = []
                for item in data:
                    earnings_list.append({
                        'date': pd.to_datetime(item['date']),
                        'source': 'fmp',
                        'eps_estimate': item.get('epsEstimated'),
                        'eps_actual': item.get('eps'),
                        'revenue_estimate': item.get('revenueEstimated'),
                        'revenue_actual': item.get('revenue')
                    })
                
                return earnings_list[:4]
            
        except Exception as e:
            print(f"FMP API error for {symbol}: {str(e)}")
        
        return None
    
    def _analyze_earnings_performance(self, earnings_data: List[Dict], stock_data: pd.DataFrame, symbol: str) -> Tuple[Optional[pd.DataFrame], int]:
        """
        Analyze stock performance after earnings announcements
        """
        if not earnings_data:
            return None, 0
        
        analysis_results = []
        successful_analyses = 0
        
        for earning in earnings_data:
            try:
                earnings_date = earning['date']
                
                # Find the closest trading day before earnings
                pre_earnings_data = stock_data[stock_data.index < earnings_date]
                if pre_earnings_data.empty:
                    continue
                
                pre_earnings_close = pre_earnings_data['Close'].iloc[-1]
                pre_earnings_date = pre_earnings_data.index[-1]
                
                # Find the trading day after earnings
                post_earnings_data = stock_data[stock_data.index > earnings_date]
                if post_earnings_data.empty:
                    continue
                
                post_earnings_open = post_earnings_data['Open'].iloc[0]
                
                # Calculate overnight change
                overnight_change = ((post_earnings_open - pre_earnings_close) / pre_earnings_close) * 100
                
                # Find end of week (5 trading days after earnings)
                week_end_data = post_earnings_data.head(5)
                if len(week_end_data) >= 1:
                    week_end_close = week_end_data['Close'].iloc[-1]
                    week_performance = ((week_end_close - pre_earnings_close) / pre_earnings_close) * 100
                else:
                    week_end_close = post_earnings_open
                    week_performance = overnight_change
                
                # Format the result
                quarter_num = (earnings_date.month - 1) // 3 + 1
                result = {
                    'Quarter': f'{earnings_date.year} Q{quarter_num}',
                    'Earnings Date': earnings_date.strftime('%Y-%m-%d'),
                    'Pre-Earnings Close': f"${pre_earnings_close:.2f}",
                    'Post-Earnings Open': f"${post_earnings_open:.2f}",
                    'Overnight Change (%)': f"{overnight_change:+.2f}%",
                    'End of Week Close': f"${week_end_close:.2f}",
                    'Week Performance (%)': f"{week_performance:+.2f}%",
                    'Direction': '📈 Up' if week_performance > 0 else '📉 Down' if week_performance < 0 else '➡️ Flat'
                }
                
                # Add EPS data if available
                if earning.get('eps_estimate'):
                    result['EPS Estimate'] = f"${earning['eps_estimate']:.2f}"
                if earning.get('eps_actual'):
                    result['EPS Actual'] = f"${earning['eps_actual']:.2f}"
                
                analysis_results.append(result)
                successful_analyses += 1
                
            except Exception as e:
                print(f"Error analyzing earnings for {symbol} on {earning['date']}: {str(e)}")
                continue
        
        if analysis_results:
            df = pd.DataFrame(analysis_results)
            return df, successful_analyses
        
        return None, 0
    
    def get_status_info(self) -> Dict:
        """
        Get information about available data sources
        """
        return {
            'yahoo_finance': 'Available (Free)',
            'financial_modeling_prep': 'Available' if self.fmp_api_key else 'Requires API Key',
            'sources_count': sum(self.sources_available.values())
        }

def test_earnings_provider():
    """
    Test function for earnings data provider
    """
    provider = EarningsDataProvider()
    
    # Test with AAPL
    print("Testing with AAPL...")
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="2y")
    
    earnings_analysis, quarters_found = provider.get_earnings_performance_analysis("AAPL", data)
    
    print(f"Quarters found: {quarters_found}")
    if earnings_analysis is not None:
        print("Earnings analysis:")
        print(earnings_analysis.to_string())
    else:
        print("No earnings analysis available")
    
    print(f"\nData sources status: {provider.get_status_info()}")

if __name__ == "__main__":
    test_earnings_provider()