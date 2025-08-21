"""
Advanced Financial Data Provider
Integrates multiple data sources for comprehensive stock analysis including:
- Analyst ratings and price targets
- Institutional holdings (13F filings)
- Insider trading activity
- Investor sentiment and social metrics
- News sentiment analysis
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional, Any
import yfinance as yf

class AdvancedFinancialDataProvider:
    """Comprehensive financial data provider integrating multiple sources"""
    
    def __init__(self):
        # API endpoints
        self.fmp_base_url = "https://financialmodelingprep.com/api"
        self.finnhub_base_url = "https://finnhub.io/api/v1"
        self.sec_api_base_url = "https://api.sec-api.io"
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 0.2  # 200ms between requests
        
    def _rate_limit(self, source: str):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        if source in self.last_request_time:
            time_diff = current_time - self.last_request_time[source]
            if time_diff < self.min_request_interval:
                time.sleep(self.min_request_interval - time_diff)
        self.last_request_time[source] = time.time()
    
    def _make_request(self, url: str, params: Dict = None, source: str = "default") -> Optional[Dict]:
        """Make HTTP request with error handling and rate limiting"""
        try:
            self._rate_limit(source)
            response = requests.get(url, params=params, timeout=10)
            
            # Log the actual response for debugging
            if response.status_code == 403:
                print(f"403 Forbidden for {source}: Check API key subscription level")
                print(f"URL: {url}")
                return None
            elif response.status_code == 429:
                print(f"429 Rate Limit for {source}: Too many requests")
                return None
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed for {source}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error for {source}: {str(e)}")
            return None
    
    def get_analyst_ratings(self, symbol: str, api_key: str = None) -> Dict[str, Any]:
        """
        Get analyst ratings and price targets from Financial Modeling Prep
        
        Args:
            symbol (str): Stock symbol
            api_key (str): FMP API key
            
        Returns:
            Dict containing analyst ratings, price targets, and recommendations
        """
        if not api_key:
            return {
                'rating': 'API key required',
                'price_target': 'N/A',
                'recommendation': 'N/A',
                'analyst_count': 0,
                'strong_buy': 0,
                'buy': 0,
                'hold': 0,
                'sell': 0,
                'strong_sell': 0
            }
        
        try:
            # Get analyst estimates
            estimates_url = f"{self.fmp_base_url}/v3/analyst-estimates/{symbol}"
            estimates_params = {'apikey': api_key, 'limit': 4}
            estimates_data = self._make_request(estimates_url, estimates_params, "fmp_estimates")
            
            # Get price target
            target_url = f"{self.fmp_base_url}/v4/price-target"
            target_params = {'symbol': symbol, 'apikey': api_key}
            target_data = self._make_request(target_url, target_params, "fmp_target")
            
            # Get rating details
            rating_url = f"{self.fmp_base_url}/v3/rating/{symbol}"
            rating_params = {'apikey': api_key}
            rating_data = self._make_request(rating_url, rating_params, "fmp_rating")
            
            # Process the data
            result = {
                'rating': 'N/A',
                'price_target': 'N/A',
                'recommendation': 'N/A',
                'analyst_count': 0,
                'strong_buy': 0,
                'buy': 0,
                'hold': 0,
                'sell': 0,
                'strong_sell': 0,
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            
            if rating_data and len(rating_data) > 0:
                latest_rating = rating_data[0]
                result['rating'] = latest_rating.get('rating', 'N/A')
                result['recommendation'] = latest_rating.get('ratingRecommendation', 'N/A')
            
            if target_data and len(target_data) > 0:
                latest_target = target_data[0]
                result['price_target'] = f"${latest_target.get('priceTarget', 0):.2f}"
                result['analyst_count'] = latest_target.get('numberOfAnalysts', 0)
            
            if estimates_data and len(estimates_data) > 0:
                latest_estimate = estimates_data[0]
                result['estimated_revenue'] = latest_estimate.get('estimatedRevenueAvg', 0)
                result['estimated_eps'] = latest_estimate.get('estimatedEpsAvg', 0)
            
            return result
            
        except Exception as e:
            print(f"Error fetching analyst ratings for {symbol}: {str(e)}")
            return {
                'rating': 'Error',
                'price_target': 'Error',
                'recommendation': 'Error',
                'analyst_count': 0,
                'error': str(e)
            }
    
    def get_insider_activity(self, symbol: str, api_key: str = None) -> Dict[str, Any]:
        """
        Get insider trading activity from Financial Modeling Prep
        
        Args:
            symbol (str): Stock symbol
            api_key (str): FMP API key
            
        Returns:
            Dict containing insider trading summary and recent transactions
        """
        if not api_key:
            return {
                'recent_activity': 'API key required',
                'insider_summary': 'N/A',
                'total_transactions': 0,
                'net_insider_activity': 'N/A'
            }
        
        try:
            url = f"{self.fmp_base_url}/v3/insider-trade"
            params = {
                'symbol': symbol,
                'apikey': api_key,
                'limit': 50
            }
            
            data = self._make_request(url, params, "fmp_insider")
            
            if not data or len(data) == 0:
                return {
                    'recent_activity': 'No recent activity',
                    'insider_summary': 'No data available',
                    'total_transactions': 0,
                    'net_insider_activity': 'No activity'
                }
            
            # Analyze insider transactions
            recent_transactions = []
            total_purchases = 0
            total_sales = 0
            purchase_value = 0
            sale_value = 0
            
            for transaction in data[:10]:  # Last 10 transactions
                trans_type = transaction.get('transactionType', '')
                shares = transaction.get('securitiesTransacted', 0)
                price = transaction.get('pricePerSecurity', 0)
                value = shares * price
                
                recent_transactions.append({
                    'date': transaction.get('filingDate', ''),
                    'insider': transaction.get('reportingName', ''),
                    'title': transaction.get('typeOfOwner', ''),
                    'transaction_type': trans_type,
                    'shares': shares,
                    'price': price,
                    'value': value
                })
                
                if trans_type and 'P' in trans_type.upper():
                    total_purchases += 1
                    purchase_value += value
                elif trans_type and 'S' in trans_type.upper():
                    total_sales += 1
                    sale_value += value
            
            # Calculate net activity
            net_value = purchase_value - sale_value
            if net_value > 0:
                net_activity = f"Net buying: ${net_value:,.0f}"
            elif net_value < 0:
                net_activity = f"Net selling: ${abs(net_value):,.0f}"
            else:
                net_activity = "Neutral"
            
            return {
                'recent_activity': f"{len(recent_transactions)} transactions in last filings",
                'insider_summary': f"{total_purchases} buys, {total_sales} sells",
                'total_transactions': len(data),
                'net_insider_activity': net_activity,
                'recent_transactions': recent_transactions[:5],  # Top 5 for display
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            print(f"Error fetching insider activity for {symbol}: {str(e)}")
            return {
                'recent_activity': 'Error',
                'insider_summary': 'Error',
                'total_transactions': 0,
                'net_insider_activity': 'Error',
                'error': str(e)
            }
    
    def get_institutional_holdings(self, symbol: str, api_key: str = None) -> Dict[str, Any]:
        """
        Get institutional holdings data from multiple sources
        
        Args:
            symbol (str): Stock symbol
            api_key (str): FMP API key
            
        Returns:
            Dict containing institutional ownership data
        """
        if not api_key:
            return {
                'institutional_ownership': 'API key required for premium data',
                'top_holders': 'Try free Finnhub alternative',
                'ownership_change': 'N/A',
                'total_institutions': 0
            }
        
        try:
            # Try FMP first
            url = f"{self.fmp_base_url}/v3/institutional-holder/{symbol}"
            params = {'apikey': api_key}
            
            data = self._make_request(url, params, "fmp_institutional")
            
            if data and len(data) > 0:
                # FMP data available
                total_shares = sum(holding.get('sharesNumber', 0) for holding in data)
                total_value = sum(holding.get('marketValue', 0) for holding in data)
                
                top_holders = []
                for holding in data[:5]:
                    holder_name = holding.get('holder', 'Unknown')
                    shares = holding.get('sharesNumber', 0)
                    value = holding.get('marketValue', 0)
                    date_reported = holding.get('dateReported', '')
                    
                    top_holders.append({
                        'name': holder_name,
                        'shares': shares,
                        'value': value,
                        'date': date_reported
                    })
                
                return {
                    'institutional_ownership': f"{len(data)} institutions",
                    'top_holders': f"Top: {top_holders[0]['name'] if top_holders else 'N/A'}",
                    'ownership_change': f"${total_value:,.0f} total value",
                    'total_institutions': len(data),
                    'total_shares_held': total_shares,
                    'total_market_value': total_value,
                    'top_5_holders': top_holders,
                    'last_updated': datetime.now().strftime('%Y-%m-%d'),
                    'source': 'FMP Premium'
                }
            else:
                # Fallback to Yahoo Finance for institutional data
                print(f"FMP failed, trying Yahoo Finance for {symbol} institutional data...")
                try:
                    ticker = yf.Ticker(symbol)
                    institutional_holders = ticker.institutional_holders
                    major_holders = ticker.major_holders
                    
                    if institutional_holders is not None and not institutional_holders.empty:
                        total_institutions = len(institutional_holders)
                        
                        # Get top holders
                        top_holders = []
                        for idx, row in institutional_holders.head(5).iterrows():
                            shares = row.get('Shares', 0)
                            name = row.get('Holder', 'Unknown')
                            percent = row.get('% Out', 0)
                            
                            # Calculate approximate value (shares * current price)
                            try:
                                current_price = ticker.info.get('currentPrice', 0)
                                value = shares * current_price if current_price else 0
                            except:
                                value = 0
                            
                            top_holders.append({
                                'name': name,
                                'shares': shares,
                                'value': value,
                                'percent': percent,
                                'date': row.get('Date Reported', '')
                            })
                        
                        # Calculate totals
                        total_shares = institutional_holders['Shares'].sum() if 'Shares' in institutional_holders.columns else 0
                        
                        # Get institutional percentage
                        institutional_percent = 0
                        if major_holders is not None and not major_holders.empty and len(major_holders) > 0:
                            try:
                                institutional_percent = float(str(major_holders.iloc[0, 0]).replace('%', ''))
                            except:
                                institutional_percent = 0
                        
                        # Calculate total market value
                        total_value = sum(holder['value'] for holder in top_holders)
                        
                        return {
                            'institutional_ownership': f"{total_institutions} institutions ({institutional_percent:.1f}%)",
                            'top_holders': f"Top: {top_holders[0]['name'] if top_holders else 'N/A'}",
                            'ownership_change': f"${total_value:,.0f} estimated value",
                            'total_institutions': total_institutions,
                            'total_shares_held': total_shares,
                            'total_market_value': total_value,
                            'institutional_percentage': institutional_percent,
                            'top_5_holders': top_holders,
                            'last_updated': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'Yahoo Finance (Free)'
                        }
                    else:
                        return {
                            'institutional_ownership': 'No Yahoo data available',
                            'top_holders': 'N/A',
                            'ownership_change': 'N/A',
                            'total_institutions': 0,
                            'source': 'Yahoo Finance (No Data)'
                        }
                
                except Exception as yahoo_error:
                    print(f"Yahoo Finance fallback also failed: {str(yahoo_error)}")
                    return {
                        'institutional_ownership': 'All sources failed',
                        'top_holders': 'FMP: 403 Forbidden, Yahoo: Error',
                        'ownership_change': 'No data available',
                        'total_institutions': 0,
                        'source': 'Error',
                        'fmp_error': 'Premium subscription required',
                        'yahoo_error': str(yahoo_error)
                    }
            
        except Exception as e:
            print(f"Error fetching institutional holdings for {symbol}: {str(e)}")
            return {
                'institutional_ownership': 'API Error',
                'top_holders': 'Check API key & subscription',
                'ownership_change': 'Error',
                'total_institutions': 0,
                'error': str(e),
                'source': 'Error'
            }
    
    def get_social_sentiment(self, symbol: str, api_key: str = None) -> Dict[str, Any]:
        """
        Get social sentiment data from multiple sources
        
        Args:
            symbol (str): Stock symbol
            api_key (str): API key (FMP or other)
            
        Returns:
            Dict containing social sentiment metrics
        """
        try:
            # Try to get sentiment from Financial Modeling Prep
            if api_key:
                url = f"{self.fmp_base_url}/v4/historical-social-sentiment"
                params = {
                    'symbol': symbol,
                    'apikey': api_key,
                    'limit': 7  # Last 7 days
                }
                
                data = self._make_request(url, params, "fmp_sentiment")
                
                if data and len(data) > 0:
                    latest_sentiment = data[0]
                    avg_sentiment = sum(item.get('sentiment', 0) for item in data) / len(data)
                    
                    return {
                        'sentiment_score': f"{avg_sentiment:.2f}",
                        'sentiment_trend': 'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral',
                        'data_points': len(data),
                        'last_updated': latest_sentiment.get('date', ''),
                        'source': 'Social Media Analysis'
                    }
            
            # Fallback to basic sentiment calculation based on news
            return {
                'sentiment_score': 'N/A',
                'sentiment_trend': 'API key required for sentiment data',
                'data_points': 0,
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Limited Access'
            }
            
        except Exception as e:
            print(f"Error fetching social sentiment for {symbol}: {str(e)}")
            return {
                'sentiment_score': 'Error',
                'sentiment_trend': 'Error',
                'data_points': 0,
                'error': str(e)
            }
    
    def get_comprehensive_analysis(self, symbol: str, api_key: str = None) -> Dict[str, Any]:
        """
        Get comprehensive analysis combining all data sources with fallback support
        
        Args:
            symbol (str): Stock symbol
            api_key (str): API key for premium data
            
        Returns:
            Dict containing all analysis components
        """
        print(f"Gathering comprehensive analysis for {symbol}...")
        
        # Try premium data sources first
        analyst_data = self.get_analyst_ratings(symbol, api_key)
        insider_data = self.get_insider_activity(symbol, api_key)
        institutional_data = self.get_institutional_holdings(symbol, api_key)
        sentiment_data = self.get_social_sentiment(symbol, api_key)
        
        # Check if premium data failed and use fallbacks
        fallback_used = False
        
        # If analyst data failed, try Yahoo Finance fallback
        if analyst_data.get('error') or 'API key required' in str(analyst_data.get('rating', '')):
            try:
                print(f"FMP analyst data failed, trying Yahoo Finance for {symbol}...")
                ticker = yf.Ticker(symbol)
                recommendations = ticker.recommendations
                analyst_price_targets = ticker.analyst_price_targets
                
                if recommendations is not None and not recommendations.empty:
                    latest_rec = recommendations.tail(1)
                    if not latest_rec.empty:
                        latest_row = latest_rec.iloc[0]
                        grade = latest_row.get('To Grade', 'N/A')
                        analyst_data = {
                            'rating': grade,
                            'recommendation': grade,
                            'price_target': 'N/A',
                            'analyst_count': len(recommendations),
                            'source': 'Yahoo Finance (Free)',
                            'last_updated': datetime.now().strftime('%Y-%m-%d')
                        }
                        
                        # Try to get price targets
                        if analyst_price_targets is not None and not analyst_price_targets.empty:
                            current_target = analyst_price_targets.get('current', 0)
                            if current_target and current_target > 0:
                                analyst_data['price_target'] = f"${current_target:.2f}"
                        
                        fallback_used = True
            except Exception as e:
                print(f"Yahoo analyst fallback failed: {str(e)}")
        
        # Note: institutional_data fallback is already handled in get_institutional_holdings method
        
        data_quality = 'Premium' if api_key and not fallback_used else 'Mixed (Premium + Free)' if fallback_used else 'Limited'
        
        return {
            'symbol': symbol,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analyst_ratings': analyst_data,
            'insider_activity': insider_data,
            'institutional_holdings': institutional_data,
            'social_sentiment': sentiment_data,
            'data_quality': data_quality,
            'fallback_used': fallback_used
        }

# Global instance for use in the main app
advanced_data_provider = AdvancedFinancialDataProvider()

def get_advanced_metrics(symbol: str, api_key: str = None) -> Dict[str, Any]:
    """
    Simplified interface for getting advanced metrics
    
    Args:
        symbol (str): Stock symbol
        api_key (str): Optional API key for premium data
        
    Returns:
        Dict containing formatted metrics for display
    """
    try:
        # Get comprehensive analysis
        analysis = advanced_data_provider.get_comprehensive_analysis(symbol, api_key)
        
        # Format for display in the main app
        return {
            'Analyst Rating': analysis['analyst_ratings'].get('rating', 'N/A'),
            'Price Target': analysis['analyst_ratings'].get('price_target', 'N/A'),
            'Insider Activity': analysis['insider_activity'].get('net_insider_activity', 'N/A'),
            'Institutional Ownership': analysis['institutional_holdings'].get('institutional_ownership', 'N/A'),
            'Social Sentiment': analysis['social_sentiment'].get('sentiment_trend', 'N/A'),
            'Data Source': 'Multiple APIs',
            'Last Updated': analysis.get('analysis_date', ''),
            'raw_data': analysis  # For detailed view
        }
        
    except Exception as e:
        print(f"Error in get_advanced_metrics for {symbol}: {str(e)}")
        return {
            'Analyst Rating': 'Error',
            'Price Target': 'Error',
            'Insider Activity': 'Error',
            'Institutional Ownership': 'Error',
            'Social Sentiment': 'Error',
            'Data Source': 'Error',
            'error': str(e)
        }