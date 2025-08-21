"""
Free alternative data sources for institutional and analyst data
Uses Yahoo Finance and other free APIs as fallback options
"""

import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

def get_yahoo_institutional_data(symbol: str) -> Dict[str, Any]:
    """
    Get institutional holdings data from Yahoo Finance (free)
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        Dict containing institutional ownership data
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get institutional holders
        institutional_holders = ticker.institutional_holders
        
        # Get major holders summary
        major_holders = ticker.major_holders
        
        if institutional_holders is not None and not institutional_holders.empty:
            # Process institutional holders data
            total_institutions = len(institutional_holders)
            
            # Get top holders
            top_holders = []
            for idx, row in institutional_holders.head(5).iterrows():
                top_holders.append({
                    'name': row.get('Holder', 'Unknown'),
                    'shares': row.get('Shares', 0),
                    'date': row.get('Date Reported', ''),
                    'percent': row.get('% Out', 0)
                })
            
            # Calculate total shares held by institutions
            total_shares = institutional_holders['Shares'].sum() if 'Shares' in institutional_holders.columns else 0
            
            # Get percentage from major holders if available
            institutional_percent = 0
            if major_holders is not None and not major_holders.empty:
                # Usually the first row is institutional ownership %
                if len(major_holders) > 0:
                    institutional_percent = major_holders.iloc[0, 0] if len(major_holders.columns) > 0 else 0
            
            return {
                'institutional_ownership': f"{total_institutions} institutions",
                'top_holders': f"Top: {top_holders[0]['name'] if top_holders else 'N/A'}",
                'ownership_change': f"{institutional_percent:.1f}% of shares" if institutional_percent else "N/A",
                'total_institutions': total_institutions,
                'total_shares_held': total_shares,
                'institutional_percentage': institutional_percent,
                'top_5_holders': top_holders,
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Yahoo Finance (Free)'
            }
        else:
            return {
                'institutional_ownership': 'No institutional data',
                'top_holders': 'N/A',
                'ownership_change': 'N/A',
                'total_institutions': 0,
                'source': 'Yahoo Finance (No Data)'
            }
            
    except Exception as e:
        print(f"Error fetching Yahoo institutional data for {symbol}: {str(e)}")
        return {
            'institutional_ownership': 'Error fetching data',
            'top_holders': 'Error',
            'ownership_change': 'Error',
            'total_institutions': 0,
            'error': str(e),
            'source': 'Yahoo Finance (Error)'
        }

def get_yahoo_analyst_data(symbol: str) -> Dict[str, Any]:
    """
    Get analyst recommendations from Yahoo Finance (free)
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        Dict containing analyst data
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get analyst recommendations
        recommendations = ticker.recommendations
        analyst_price_targets = ticker.analyst_price_targets
        
        result = {
            'rating': 'N/A',
            'price_target': 'N/A',
            'recommendation': 'N/A',
            'analyst_count': 0,
            'source': 'Yahoo Finance (Free)'
        }
        
        # Process recommendations
        if recommendations is not None and not recommendations.empty:
            latest_rec = recommendations.tail(1)
            if not latest_rec.empty:
                # Get latest recommendation
                latest_row = latest_rec.iloc[0]
                result['recommendation'] = f"{latest_row.get('To Grade', 'N/A')}"
                result['rating'] = f"{latest_row.get('To Grade', 'N/A')}"
        
        # Process price targets
        if analyst_price_targets is not None and not analyst_price_targets.empty:
            current_target = analyst_price_targets.get('current', 0)
            if current_target and current_target > 0:
                result['price_target'] = f"${current_target:.2f}"
            
            # Get analyst count if available
            analyst_count = len(recommendations) if recommendations is not None else 0
            result['analyst_count'] = analyst_count
        
        result['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        return result
        
    except Exception as e:
        print(f"Error fetching Yahoo analyst data for {symbol}: {str(e)}")
        return {
            'rating': 'Error',
            'price_target': 'Error',
            'recommendation': 'Error',
            'analyst_count': 0,
            'error': str(e),
            'source': 'Yahoo Finance (Error)'
        }

def get_fallback_data(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive fallback data when premium APIs are unavailable
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        Dict containing all available free data
    """
    try:
        # Get Yahoo Finance data
        institutional_data = get_yahoo_institutional_data(symbol)
        analyst_data = get_yahoo_analyst_data(symbol)
        
        return {
            'symbol': symbol,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analyst_ratings': analyst_data,
            'institutional_holdings': institutional_data,
            'insider_activity': {
                'recent_activity': 'Premium API required',
                'insider_summary': 'Upgrade for insider data',
                'net_insider_activity': 'N/A',
                'source': 'Requires FMP Premium'
            },
            'social_sentiment': {
                'sentiment_score': 'N/A',
                'sentiment_trend': 'Premium API required',
                'data_points': 0,
                'source': 'Requires Premium Subscription'
            },
            'data_quality': 'Free Tier - Limited Access'
        }
        
    except Exception as e:
        print(f"Error in fallback data for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e),
            'data_quality': 'Error'
        }