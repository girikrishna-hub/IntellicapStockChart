import requests
import trafilatura
import json
import re
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional

def get_tradingview_earnings_data(symbol: str) -> Optional[Dict]:
    """
    Scrape TradingView earnings calendar data for a specific stock symbol
    
    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL")
    
    Returns:
        Dict: Earnings data or None if not available
    """
    try:
        # TradingView earnings calendar URL for specific symbol
        base_url = f"https://www.tradingview.com/symbols/{symbol.upper()}/earnings/"
        
        # Fetch the webpage content
        downloaded = trafilatura.fetch_url(base_url)
        if not downloaded:
            return None
        
        # Extract text content
        text_content = trafilatura.extract(downloaded)
        if not text_content:
            return None
        
        # Try to find earnings data patterns in the text
        earnings_data = parse_earnings_from_content(text_content, symbol)
        
        return earnings_data
        
    except Exception as e:
        print(f"Error fetching TradingView data for {symbol}: {str(e)}")
        return None

def parse_earnings_from_content(content: str, symbol: str) -> Optional[Dict]:
    """
    Parse earnings information from TradingView page content
    
    Args:
        content (str): Raw text content from TradingView page
        symbol (str): Stock symbol
    
    Returns:
        Dict: Parsed earnings data or None
    """
    try:
        earnings_info = {
            'symbol': symbol.upper(),
            'upcoming_earnings': [],
            'past_earnings': [],
            'source': 'TradingView'
        }
        
        # Look for earnings date patterns
        date_patterns = [
            r'(\w+\s+\d{1,2},?\s+\d{4})',  # January 15, 2024
            r'(\d{1,2}/\d{1,2}/\d{4})',     # 01/15/2024
            r'(\d{4}-\d{2}-\d{2})',         # 2024-01-15
        ]
        
        # Look for EPS patterns
        eps_patterns = [
            r'EPS.*?(\$?-?\d+\.\d+)',
            r'Earnings.*?(\$?-?\d+\.\d+)',
            r'Per Share.*?(\$?-?\d+\.\d+)',
        ]
        
        # Search for patterns in content
        found_dates = []
        found_eps = []
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_dates.extend(matches)
        
        for pattern in eps_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_eps.extend(matches)
        
        # Process found data
        if found_dates or found_eps:
            earnings_info['raw_dates'] = found_dates[:5]  # Limit to 5 most recent
            earnings_info['raw_eps'] = found_eps[:5]
            earnings_info['content_preview'] = content[:500] + "..." if len(content) > 500 else content
            return earnings_info
        
        return None
        
    except Exception as e:
        print(f"Error parsing earnings content: {str(e)}")
        return None

def get_alternative_earnings_data(symbol: str) -> Optional[Dict]:
    """
    Alternative method to get earnings data using different TradingView endpoints
    
    Args:
        symbol (str): Stock ticker symbol
    
    Returns:
        Dict: Earnings data or None
    """
    try:
        # Try the general TradingView earnings calendar
        calendar_url = "https://www.tradingview.com/markets/stocks-usa/earnings/"
        
        downloaded = trafilatura.fetch_url(calendar_url)
        if not downloaded:
            return None
        
        text_content = trafilatura.extract(downloaded)
        if not text_content:
            return None
        
        # Look for the specific symbol in the calendar
        if symbol.upper() in text_content:
            # Extract surrounding context for the symbol
            symbol_index = text_content.upper().find(symbol.upper())
            if symbol_index != -1:
                # Get 200 characters before and after the symbol mention
                start = max(0, symbol_index - 200)
                end = min(len(text_content), symbol_index + 200)
                context = text_content[start:end]
                
                return {
                    'symbol': symbol.upper(),
                    'source': 'TradingView Calendar',
                    'context': context,
                    'found_in_calendar': True
                }
        
        return None
        
    except Exception as e:
        print(f"Error with alternative earnings data: {str(e)}")
        return None

def test_tradingview_scraping(symbol: str = "AAPL"):
    """
    Test function to see what data we can extract from TradingView
    
    Args:
        symbol (str): Stock symbol to test with
    
    Returns:
        Dict: Test results
    """
    print(f"Testing TradingView scraping for {symbol}...")
    
    # Test direct symbol page
    direct_data = get_tradingview_earnings_data(symbol)
    
    # Test alternative method
    calendar_data = get_alternative_earnings_data(symbol)
    
    results = {
        'symbol': symbol,
        'direct_method': direct_data,
        'calendar_method': calendar_data,
        'timestamp': datetime.now().isoformat()
    }
    
    return results

if __name__ == "__main__":
    # Test the scraping functionality
    test_results = test_tradingview_scraping("AAPL")
    print(json.dumps(test_results, indent=2, default=str))