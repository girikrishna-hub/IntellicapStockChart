import requests
import json
import os
from datetime import datetime, timedelta
from openai import OpenAI
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from trafilatura import fetch_url, extract
import feedparser
from newsapi import NewsApiClient
from alpha_vantage.timeseries import TimeSeries
import time

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_openai_client():
    """Get OpenAI client with proper error handling"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not found")
    
    if not OPENAI_API_KEY.startswith('sk-'):
        raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-'")
    
    return OpenAI(api_key=OPENAI_API_KEY)

def fetch_from_yahoo_finance(symbol, max_articles=5):
    """Fetch news from Yahoo Finance RSS feed"""
    try:
        # Yahoo Finance RSS feed for stock news
        yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        feed = feedparser.parse(yahoo_url)
        
        articles = []
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.title,
                "summary": entry.get('summary', entry.get('description', '')),
                "url": entry.link,
                "published_date": entry.get('published', ''),
                "source": "Yahoo Finance"
            })
        return articles
    except Exception as e:
        print(f"Yahoo Finance error: {e}")
        return []

def fetch_from_newsapi(symbol, api_key, max_articles=5):
    """Fetch news from NewsAPI"""
    try:
        if not api_key:
            return []
        
        newsapi = NewsApiClient(api_key=api_key)
        
        # Get company name for better search results
        company_names = {
            'AAPL': 'Apple',
            'GOOGL': 'Google', 
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'META': 'Meta',
            'NVDA': 'NVIDIA'
        }
        
        search_query = f"{symbol} OR {company_names.get(symbol, symbol)}"
        
        articles = newsapi.get_everything(
            q=search_query,
            language='en',
            sort_by='publishedAt',
            from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            to=(datetime.now()).strftime('%Y-%m-%d'),
            page_size=max_articles
        )
        
        formatted_articles = []
        for article in articles.get('articles', []):
            formatted_articles.append({
                "title": article['title'],
                "summary": article['description'] or '',
                "url": article['url'],
                "published_date": article['publishedAt'],
                "source": f"NewsAPI ({article['source']['name']})"
            })
        
        return formatted_articles
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []

def fetch_from_google_news(symbol, max_articles=5):
    """Fetch news from Google News RSS feed"""
    try:
        # Google News RSS feed
        google_url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(google_url)
        
        articles = []
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.title,
                "summary": entry.get('summary', ''),
                "url": entry.link,
                "published_date": entry.get('published', ''),
                "source": "Google News"
            })
        return articles
    except Exception as e:
        print(f"Google News error: {e}")
        return []

def fetch_from_alpha_vantage(symbol, api_key, max_articles=5):
    """Fetch news from Alpha Vantage News & Sentiment API"""
    try:
        if not api_key:
            return []
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': api_key,
            'limit': max_articles
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        articles = []
        for item in data.get('feed', []):
            articles.append({
                "title": item['title'],
                "summary": item['summary'],
                "url": item['url'],
                "published_date": item['time_published'],
                "source": f"Alpha Vantage ({item.get('source', 'Unknown')})"
            })
        
        return articles
    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return []

def fetch_financial_news(symbol, selected_sources, days_back=7, max_articles=10):
    """
    Fetch financial news articles for a given stock symbol from multiple sources
    
    Args:
        symbol (str): Stock symbol
        selected_sources (list): List of selected news sources
        days_back (int): Number of days to look back for news
        max_articles (int): Maximum number of articles to fetch
    
    Returns:
        list: List of news articles with metadata
    """
    all_articles = []
    articles_per_source = max(1, max_articles // len(selected_sources)) if selected_sources else max_articles
    
    try:
        # Fetch from each selected source
        if "Yahoo Finance" in selected_sources:
            yahoo_articles = fetch_from_yahoo_finance(symbol, articles_per_source)
            all_articles.extend(yahoo_articles)
        
        if "Google News" in selected_sources:
            google_articles = fetch_from_google_news(symbol, articles_per_source)
            all_articles.extend(google_articles)
        
        if "NewsAPI" in selected_sources:
            # Check if NewsAPI key is available
            newsapi_key = os.environ.get('NEWSAPI_API_KEY')
            if newsapi_key:
                newsapi_articles = fetch_from_newsapi(symbol, newsapi_key, articles_per_source)
                all_articles.extend(newsapi_articles)
        
        if "Alpha Vantage" in selected_sources:
            # Check if Alpha Vantage key is available
            av_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
            if av_key:
                av_articles = fetch_from_alpha_vantage(symbol, av_key, articles_per_source)
                all_articles.extend(av_articles)
        
        # If no articles found from selected sources, provide fallback sample
        if not all_articles:
            sample_articles = [
                {
                    "title": f"{symbol} Reports Strong Q3 Earnings, Beats Expectations",
                    "summary": f"{symbol} announced quarterly earnings that exceeded analyst expectations, driven by strong revenue growth and improved margins.",
                    "url": "#",
                    "published_date": (datetime.now() - timedelta(days=1)).isoformat(),
                    "source": "Sample News"
                },
                {
                    "title": f"Analysts Upgrade {symbol} Price Target Following Recent Developments", 
                    "summary": f"Several Wall Street analysts have raised their price targets for {symbol} citing positive market trends and strong fundamentals.",
                    "url": "#",
                    "published_date": (datetime.now() - timedelta(days=2)).isoformat(),
                    "source": "Sample News"
                },
                {
                    "title": f"{symbol} Faces Headwinds as Market Conditions Worsen",
                    "summary": f"Industry experts express concerns about {symbol}'s ability to maintain growth amid challenging market conditions and increased competition.",
                    "url": "#", 
                    "published_date": (datetime.now() - timedelta(days=3)).isoformat(),
                    "source": "Sample News"
                }
            ]
            all_articles = sample_articles
        
        # Remove duplicates and sort by date
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            title_key = article['title'].lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        # Sort by publication date (most recent first)
        try:
            unique_articles.sort(key=lambda x: x['published_date'], reverse=True)
        except:
            pass  # Keep original order if date sorting fails
        
        return unique_articles[:max_articles]
        
    except Exception as e:
        st.error(f"Error fetching news for {symbol}: {str(e)}")
        return []

def analyze_news_sentiment(articles):
    """
    Analyze sentiment of news articles using OpenAI
    
    Args:
        articles (list): List of news articles
    
    Returns:
        list: Articles with sentiment analysis results
    """
    try:
        # Check OpenAI API key and client
        try:
            openai_client = get_openai_client()
        except ValueError as e:
            st.error(f"OpenAI API configuration error: {str(e)}")
            st.info("Please check that your OpenAI API key is correctly set in the environment variables.")
            return articles
        
        analyzed_articles = []
        
        for article in articles:
            # Combine title and summary for analysis
            text_to_analyze = f"{article['title']} {article.get('summary', '')}"
            
            # Use OpenAI for sentiment analysis
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a financial sentiment analysis expert. Analyze the sentiment of financial news articles and provide:
                        1. Overall sentiment: Positive, Negative, or Neutral
                        2. Confidence score: 0.0 to 1.0 (how confident you are in the sentiment)
                        3. Investment impact: Bullish, Bearish, or Neutral
                        4. Key themes: List of 2-3 key themes from the article
                        5. Risk factors: Any potential risks mentioned
                        
                        Respond with JSON in this exact format:
                        {
                            "sentiment": "Positive/Negative/Neutral",
                            "confidence": 0.85,
                            "investment_impact": "Bullish/Bearish/Neutral",
                            "key_themes": ["theme1", "theme2"],
                            "risk_factors": ["risk1", "risk2"],
                            "reasoning": "Brief explanation of the sentiment analysis"
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the sentiment of this financial news: {text_to_analyze}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Parse the sentiment analysis
            response_content = response.choices[0].message.content
            if response_content:
                sentiment_data = json.loads(response_content)
            else:
                # Fallback if no response content
                sentiment_data = {
                    "sentiment": "Neutral",
                    "confidence": 0.5,
                    "investment_impact": "Neutral",
                    "key_themes": [],
                    "risk_factors": [],
                    "reasoning": "Unable to analyze sentiment"
                }
            
            # Add sentiment data to article
            article_with_sentiment = article.copy()
            article_with_sentiment.update(sentiment_data)
            
            analyzed_articles.append(article_with_sentiment)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        return analyzed_articles
        
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        st.error("This could be due to:")
        st.markdown("""
        - Invalid or expired OpenAI API key
        - OpenAI API rate limits
        - Network connectivity issues
        - OpenAI service temporarily unavailable
        """)
        st.info("Please verify your OpenAI API key and try again.")
        return articles

def calculate_aggregate_sentiment(analyzed_articles):
    """
    Calculate aggregate sentiment metrics from analyzed articles
    
    Args:
        analyzed_articles (list): Articles with sentiment analysis
    
    Returns:
        dict: Aggregate sentiment metrics
    """
    if not analyzed_articles:
        return None
    
    # Count sentiments
    positive_count = sum(1 for a in analyzed_articles if a.get('sentiment') == 'Positive')
    negative_count = sum(1 for a in analyzed_articles if a.get('sentiment') == 'Negative')
    neutral_count = sum(1 for a in analyzed_articles if a.get('sentiment') == 'Neutral')
    
    total_articles = len(analyzed_articles)
    
    # Calculate weighted sentiment score (-1 to +1)
    sentiment_score = (positive_count - negative_count) / total_articles
    
    # Calculate average confidence
    avg_confidence = sum(a.get('confidence', 0) for a in analyzed_articles) / total_articles
    
    # Determine overall investment impact
    bullish_count = sum(1 for a in analyzed_articles if a.get('investment_impact') == 'Bullish')
    bearish_count = sum(1 for a in analyzed_articles if a.get('investment_impact') == 'Bearish')
    
    if bullish_count > bearish_count:
        overall_impact = "Bullish"
    elif bearish_count > bullish_count:
        overall_impact = "Bearish"
    else:
        overall_impact = "Neutral"
    
    # Collect all themes and risk factors
    all_themes = []
    all_risks = []
    
    for article in analyzed_articles:
        all_themes.extend(article.get('key_themes', []))
        all_risks.extend(article.get('risk_factors', []))
    
    # Count theme and risk frequency
    theme_counts = {}
    risk_counts = {}
    
    for theme in all_themes:
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    for risk in all_risks:
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    # Get top themes and risks
    top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_risks = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'total_articles': total_articles,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'sentiment_score': sentiment_score,
        'avg_confidence': avg_confidence,
        'overall_impact': overall_impact,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'top_themes': top_themes,
        'top_risks': top_risks
    }

def create_sentiment_charts(aggregate_metrics, analyzed_articles):
    """
    Create visualizations for sentiment analysis
    
    Args:
        aggregate_metrics (dict): Aggregate sentiment metrics
        analyzed_articles (list): Articles with sentiment analysis
    
    Returns:
        tuple: (sentiment_pie_chart, confidence_chart, timeline_chart)
    """
    if not aggregate_metrics or not analyzed_articles:
        return None, None, None
    
    # 1. Sentiment Distribution Pie Chart
    sentiment_fig = go.Figure(data=[go.Pie(
        labels=['Positive', 'Negative', 'Neutral'],
        values=[
            aggregate_metrics['positive_count'],
            aggregate_metrics['negative_count'],
            aggregate_metrics['neutral_count']
        ],
        marker_colors=['#2E8B57', '#DC143C', '#808080'],
        hole=0.4
    )])
    
    sentiment_fig.update_layout(
        title="News Sentiment Distribution",
        height=400,
        showlegend=True
    )
    
    # 2. Investment Impact Chart
    impact_fig = go.Figure(data=[go.Bar(
        x=['Bullish', 'Bearish', 'Neutral'],
        y=[
            aggregate_metrics['bullish_count'],
            aggregate_metrics['bearish_count'],
            aggregate_metrics['total_articles'] - aggregate_metrics['bullish_count'] - aggregate_metrics['bearish_count']
        ],
        marker_color=['#2E8B57', '#DC143C', '#808080']
    )])
    
    impact_fig.update_layout(
        title="Investment Impact Analysis",
        xaxis_title="Impact Type",
        yaxis_title="Number of Articles",
        height=400
    )
    
    # 3. Confidence vs Sentiment Scatter Plot
    confidence_data = []
    sentiment_numeric = []
    article_titles = []
    
    for article in analyzed_articles:
        confidence_data.append(article.get('confidence', 0))
        
        # Convert sentiment to numeric
        if article.get('sentiment') == 'Positive':
            sentiment_numeric.append(1)
        elif article.get('sentiment') == 'Negative':
            sentiment_numeric.append(-1)
        else:
            sentiment_numeric.append(0)
        
        article_titles.append(article.get('title', 'Unknown'))
    
    confidence_fig = go.Figure(data=[go.Scatter(
        x=confidence_data,
        y=sentiment_numeric,
        mode='markers',
        marker=dict(
            size=10,
            color=sentiment_numeric,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sentiment")
        ),
        text=article_titles,
        hovertemplate='<b>%{text}</b><br>Confidence: %{x:.2f}<br>Sentiment: %{y}<extra></extra>'
    )])
    
    confidence_fig.update_layout(
        title="Sentiment Confidence Analysis",
        xaxis_title="Confidence Score",
        yaxis_title="Sentiment (-1: Negative, 0: Neutral, 1: Positive)",
        height=400
    )
    
    return sentiment_fig, impact_fig, confidence_fig

def display_news_sentiment_analysis(symbol):
    """
    Main function to display news sentiment analysis for a stock
    
    Args:
        symbol (str): Stock symbol
    """
    st.subheader(f"📰 AI-Powered News Sentiment Analysis for {symbol.upper()}")
    
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        st.warning("OpenAI API key not found. Please add your OPENAI_API_KEY to use this feature.")
        st.info("This feature analyzes financial news sentiment to provide investment insights.")
        return
    
    if not OPENAI_API_KEY.startswith('sk-'):
        st.error("Invalid OpenAI API key format. Please check your API key.")
        return
    
    # News source selection
    st.markdown("### 🌐 Select News Sources")
    st.markdown("Choose which news sources to analyze for comprehensive sentiment coverage:")
    
    # Available news sources with descriptions
    news_sources = {
        "Yahoo Finance": "📈 Real-time financial news and analysis",
        "Google News": "🔍 Comprehensive news aggregation from multiple sources", 
        "NewsAPI": "📰 Professional news API (requires API key)",
        "Alpha Vantage": "💼 Financial data provider (requires API key)"
    }
    
    # Create columns for source selection
    col_src1, col_src2 = st.columns(2)
    
    selected_sources = []
    
    with col_src1:
        if st.checkbox("Yahoo Finance", value=True, help=news_sources["Yahoo Finance"]):
            selected_sources.append("Yahoo Finance")
        if st.checkbox("NewsAPI", value=False, help=news_sources["NewsAPI"]):
            # Check if API key is available
            newsapi_key = os.environ.get('NEWSAPI_API_KEY')
            if newsapi_key:
                selected_sources.append("NewsAPI")
                st.success("✅ API Key Found")
            else:
                st.warning("⚠️ API Key Missing")
                st.caption("Add NEWSAPI_API_KEY to environment to use this source")
    
    with col_src2:
        if st.checkbox("Google News", value=True, help=news_sources["Google News"]):
            selected_sources.append("Google News")
        if st.checkbox("Alpha Vantage", value=False, help=news_sources["Alpha Vantage"]):
            # Check if API key is available
            av_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
            if av_key:
                selected_sources.append("Alpha Vantage")
                st.success("✅ API Key Found")
            else:
                st.warning("⚠️ API Key Missing")
                st.caption("Add ALPHA_VANTAGE_API_KEY to environment to use this source")
    
    if not selected_sources:
        st.error("Please select at least one news source to continue.")
        return
    
    # Show API key setup instructions if premium sources are selected
    premium_sources = [s for s in selected_sources if s in ['NewsAPI', 'Alpha Vantage']]
    if premium_sources:
        with st.expander("🔑 API Key Setup Instructions"):
            st.markdown("""
            **To use premium news sources, you'll need API keys:**
            
            **NewsAPI** (Free tier: 1,000 requests/month):
            1. Visit [newsapi.org](https://newsapi.org)
            2. Sign up for a free account
            3. Copy your API key
            4. Add it to environment variables as `NEWSAPI_API_KEY`
            
            **Alpha Vantage** (Free tier: 25 requests/day):
            1. Visit [alphavantage.co](https://www.alphavantage.co/support/#api-key)
            2. Get your free API key
            3. Add it to environment variables as `ALPHA_VANTAGE_API_KEY`
            """)
    
    st.success(f"Selected sources: {', '.join(selected_sources)}")
    
    # Configuration options
    st.markdown("### ⚙️ Analysis Settings")
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        days_back = st.selectbox(
            "Analysis Period",
            options=[3, 7, 14, 30],
            index=1,
            help="Number of days to look back for news articles"
        )
    
    with col_config2:
        max_articles = st.selectbox(
            "Max Articles",
            options=[5, 10, 15, 20],
            index=1,
            help="Maximum number of articles to analyze"
        )
    
    if st.button("🔍 Analyze News Sentiment", type="primary"):
        with st.spinner(f"Fetching news from {len(selected_sources)} source(s) and analyzing sentiment..."):
            # Display progress information
            st.info(f"📡 Fetching articles from: {', '.join(selected_sources)}")
            
            # Fetch news articles from selected sources
            articles = fetch_financial_news(symbol, selected_sources, days_back, max_articles)
            
            if not articles:
                st.warning(f"No recent news articles found for {symbol}. This could be due to:")
                st.markdown("""
                - Limited news coverage for this stock
                - API rate limits or access restrictions
                - Recent market holidays or weekends
                - Stock symbol not widely covered in financial media
                """)
                return
            
            # Analyze sentiment
            analyzed_articles = analyze_news_sentiment(articles)
            
            # Calculate aggregate metrics
            aggregate_metrics = calculate_aggregate_sentiment(analyzed_articles)
            
            if aggregate_metrics:
                # Show source breakdown
                source_counts = {}
                for article in analyzed_articles:
                    source = article.get('source', 'Unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
                
                st.markdown("### 📡 Sources Used")
                col_sources = st.columns(len(source_counts))
                for i, (source, count) in enumerate(source_counts.items()):
                    with col_sources[i]:
                        st.metric(source, f"{count} articles")
                
                # Display summary metrics
                st.markdown("### 📊 Sentiment Summary")
                
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                
                with col_metric1:
                    sentiment_color = "normal"
                    if aggregate_metrics['sentiment_score'] > 0.2:
                        sentiment_color = "normal"
                    elif aggregate_metrics['sentiment_score'] < -0.2:
                        sentiment_color = "inverse"
                    
                    st.metric(
                        "Overall Sentiment",
                        f"{aggregate_metrics['sentiment_score']:+.2f}",
                        help="Range: -1.0 (Very Negative) to +1.0 (Very Positive)"
                    )
                
                with col_metric2:
                    st.metric(
                        "Confidence Level",
                        f"{aggregate_metrics['avg_confidence']:.1%}",
                        help="Average confidence in sentiment analysis"
                    )
                
                with col_metric3:
                    st.metric(
                        "Articles Analyzed",
                        aggregate_metrics['total_articles'],
                        help="Total number of articles processed"
                    )
                
                with col_metric4:
                    st.metric(
                        "Investment Outlook",
                        aggregate_metrics['overall_impact'],
                        help="Overall investment impact assessment"
                    )
                
                # Display charts
                st.markdown("### 📈 Sentiment Visualizations")
                
                sentiment_fig, impact_fig, confidence_fig = create_sentiment_charts(
                    aggregate_metrics, analyzed_articles
                )
                
                if sentiment_fig and impact_fig and confidence_fig:
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        st.plotly_chart(sentiment_fig, use_container_width=True)
                    
                    with col_chart2:
                        st.plotly_chart(impact_fig, use_container_width=True)
                    
                    st.plotly_chart(confidence_fig, use_container_width=True)
                
                # Display key themes and risks
                st.markdown("### 🔍 Key Insights")
                
                col_themes, col_risks = st.columns(2)
                
                with col_themes:
                    st.markdown("**📌 Top Themes:**")
                    if aggregate_metrics['top_themes']:
                        for theme, count in aggregate_metrics['top_themes']:
                            st.write(f"• {theme} ({count} mentions)")
                    else:
                        st.write("No major themes identified")
                
                with col_risks:
                    st.markdown("**⚠️ Risk Factors:**")
                    if aggregate_metrics['top_risks']:
                        for risk, count in aggregate_metrics['top_risks']:
                            st.write(f"• {risk} ({count} mentions)")
                    else:
                        st.write("No significant risks mentioned")
                
                # Display detailed article analysis
                st.markdown("### 📄 Detailed Article Analysis")
                
                for i, article in enumerate(analyzed_articles, 1):
                    with st.expander(f"Article {i}: {article.get('title', 'Unknown Title')}"):
                        col_art1, col_art2 = st.columns([2, 1])
                        
                        with col_art1:
                            st.write(f"**Source:** {article.get('source', 'Unknown')}")
                            st.write(f"**Published:** {article.get('published_date', 'Unknown')}")
                            st.write(f"**Summary:** {article.get('summary', 'No summary available')}")
                        
                        with col_art2:
                            sentiment = article.get('sentiment', 'Unknown')
                            confidence = article.get('confidence', 0)
                            impact = article.get('investment_impact', 'Unknown')
                            
                            # Color code sentiment
                            if sentiment == 'Positive':
                                st.success(f"**Sentiment:** {sentiment}")
                            elif sentiment == 'Negative':
                                st.error(f"**Sentiment:** {sentiment}")
                            else:
                                st.info(f"**Sentiment:** {sentiment}")
                            
                            st.metric("Confidence", f"{confidence:.1%}")
                            st.write(f"**Impact:** {impact}")
                        
                        if article.get('reasoning'):
                            st.write(f"**Analysis:** {article['reasoning']}")
            
            else:
                st.error("Failed to analyze sentiment. Please try again.")

def get_sentiment_summary_for_sharing(symbol, aggregate_metrics):
    """
    Generate a summary of sentiment analysis for social sharing
    
    Args:
        symbol (str): Stock symbol
        aggregate_metrics (dict): Aggregate sentiment metrics
    
    Returns:
        str: Formatted sentiment summary
    """
    if not aggregate_metrics:
        return f"News sentiment analysis not available for {symbol.upper()}"
    
    sentiment_score = aggregate_metrics['sentiment_score']
    total_articles = aggregate_metrics['total_articles']
    overall_impact = aggregate_metrics['overall_impact']
    confidence = aggregate_metrics['avg_confidence']
    
    # Determine sentiment description
    if sentiment_score > 0.3:
        sentiment_desc = "Very Positive"
    elif sentiment_score > 0.1:
        sentiment_desc = "Positive"
    elif sentiment_score > -0.1:
        sentiment_desc = "Neutral"
    elif sentiment_score > -0.3:
        sentiment_desc = "Negative"
    else:
        sentiment_desc = "Very Negative"
    
    summary = f"""
📰 News Sentiment Analysis for {symbol.upper()}:
• Overall Sentiment: {sentiment_desc} ({sentiment_score:+.2f})
• Investment Outlook: {overall_impact}
• Confidence Level: {confidence:.1%}
• Articles Analyzed: {total_articles}
"""
    
    # Add top themes if available
    if aggregate_metrics.get('top_themes'):
        top_theme = aggregate_metrics['top_themes'][0][0]
        summary += f"• Key Theme: {top_theme}"
    
    return summary.strip()

class NewsSentimentAnalyzer:
    """
    Main class for news sentiment analysis functionality
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.openai_client = None
        try:
            self.openai_client = get_openai_client()
        except ValueError as e:
            st.error(f"OpenAI configuration error: {str(e)}")
    
    def analyze_stock_sentiment(self, symbol, days_back=7, max_articles=10):
        """
        Analyze sentiment for a stock symbol
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days to look back
            max_articles (int): Maximum articles to analyze
        
        Returns:
            tuple: (analyzed_articles, aggregate_metrics)
        """
        if not self.openai_client:
            return None, None
        
        # Fetch news articles
        articles = fetch_financial_news(symbol, days_back, max_articles)
        
        if not articles:
            return None, None
        
        # Analyze sentiment
        analyzed_articles = analyze_news_sentiment(articles)
        
        # Calculate aggregate metrics
        aggregate_metrics = calculate_aggregate_sentiment(analyzed_articles)
        
        return analyzed_articles, aggregate_metrics
    
    def get_sentiment_summary(self, symbol, aggregate_metrics):
        """Get formatted sentiment summary for sharing"""
        return get_sentiment_summary_for_sharing(symbol, aggregate_metrics)
    
    def analyze_article_sentiment(self, title, content, symbol):
        """
        Analyze sentiment of a single article
        
        Args:
            title (str): Article title
            content (str): Article content
            symbol (str): Stock symbol
        
        Returns:
            dict: Sentiment analysis results
        """
        if not self.openai_client:
            return {
                "sentiment": "Neutral",
                "confidence": 0.5,
                "investment_impact": "Neutral",
                "key_themes": [],
                "risk_factors": [],
                "reasoning": "OpenAI client not available"
            }
        
        # Combine title and content for analysis
        text_to_analyze = f"{title} {content}"
        
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a financial sentiment analysis expert. Analyze the sentiment of financial news articles and provide:
                        1. Overall sentiment: Positive, Negative, or Neutral
                        2. Confidence score: 0.0 to 1.0 (how confident you are in the sentiment)
                        3. Investment impact: Bullish, Bearish, or Neutral
                        4. Key themes: List of 2-3 key themes from the article
                        5. Risk factors: Any potential risks mentioned
                        
                        Respond with JSON in this exact format:
                        {
                            "sentiment": "Positive/Negative/Neutral",
                            "confidence": 0.85,
                            "investment_impact": "Bullish/Bearish/Neutral",
                            "key_themes": ["theme1", "theme2"],
                            "risk_factors": ["risk1", "risk2"],
                            "reasoning": "Brief explanation of the sentiment analysis"
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the sentiment of this financial news about {symbol}: {text_to_analyze}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Parse the sentiment analysis
            response_content = response.choices[0].message.content
            if response_content:
                return json.loads(response_content)
            else:
                # Fallback if no response content
                return {
                    "sentiment": "Neutral",
                    "confidence": 0.5,
                    "investment_impact": "Neutral",
                    "key_themes": [],
                    "risk_factors": [],
                    "reasoning": "Unable to analyze sentiment"
                }
        
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "sentiment": "Neutral",
                "confidence": 0.5,
                "investment_impact": "Neutral",
                "key_themes": [],
                "risk_factors": [],
                "reasoning": f"Analysis failed: {str(e)}"
            }
    
    def display_sentiment_analysis(self, analysis_results, symbol):
        """
        Display the sentiment analysis results with social sharing
        
        Args:
            analysis_results (list): List of sentiment analysis results
            symbol (str): Stock symbol
        """
        if not analysis_results:
            st.warning("No sentiment analysis results to display.")
            return
        
        # Calculate aggregate metrics
        total_articles = len(analysis_results)
        positive_count = sum(1 for r in analysis_results if r.get('sentiment') == 'Positive')
        negative_count = sum(1 for r in analysis_results if r.get('sentiment') == 'Negative')
        neutral_count = sum(1 for r in analysis_results if r.get('sentiment') == 'Neutral')
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_articles if total_articles > 0 else 0
        
        # Calculate average confidence
        avg_confidence = sum(r.get('confidence', 0) for r in analysis_results) / total_articles if total_articles > 0 else 0
        
        # Determine overall investment impact
        bullish_count = sum(1 for r in analysis_results if r.get('investment_impact') == 'Bullish')
        bearish_count = sum(1 for r in analysis_results if r.get('investment_impact') == 'Bearish')
        
        if bullish_count > bearish_count:
            overall_impact = "Bullish"
        elif bearish_count > bullish_count:
            overall_impact = "Bearish"
        else:
            overall_impact = "Neutral"
        
        # Display summary metrics
        st.markdown("### 📊 Sentiment Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Sentiment",
                f"{sentiment_score:+.2f}",
                help="Range: -1.0 (Very Negative) to +1.0 (Very Positive)"
            )
        
        with col2:
            st.metric(
                "Confidence Level",
                f"{avg_confidence:.1%}",
                help="Average confidence in sentiment analysis"
            )
        
        with col3:
            st.metric(
                "Articles Analyzed",
                total_articles,
                help="Total number of articles processed"
            )
        
        with col4:
            st.metric(
                "Investment Outlook",
                overall_impact,
                help="Overall investment impact assessment"
            )
        
        # Display sentiment distribution chart
        st.markdown("### 📈 Sentiment Distribution")
        
        sentiment_fig = go.Figure(data=[go.Pie(
            labels=['Positive', 'Negative', 'Neutral'],
            values=[positive_count, negative_count, neutral_count],
            marker_colors=['#2E8B57', '#DC143C', '#808080'],
            hole=0.4
        )])
        
        sentiment_fig.update_layout(
            title=f"News Sentiment Distribution for {symbol}",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Display detailed article analysis
        st.markdown("### 📄 Detailed Article Analysis")
        
        for i, result in enumerate(analysis_results, 1):
            with st.expander(f"Article {i}: Analysis Results"):
                col_art1, col_art2 = st.columns([2, 1])
                
                with col_art1:
                    st.write(f"**Source:** {result.get('source', 'Unknown')}")
                    st.write(f"**Date:** {result.get('date', 'Unknown')}")
                    if result.get('reasoning'):
                        st.write(f"**Analysis:** {result['reasoning']}")
                    
                    # Display key themes
                    if result.get('key_themes'):
                        st.write("**Key Themes:**")
                        for theme in result['key_themes']:
                            st.write(f"• {theme}")
                    
                    # Display risk factors
                    if result.get('risk_factors'):
                        st.write("**Risk Factors:**")
                        for risk in result['risk_factors']:
                            st.write(f"• {risk}")
                
                with col_art2:
                    sentiment = result.get('sentiment', 'Unknown')
                    confidence = result.get('confidence', 0)
                    impact = result.get('investment_impact', 'Unknown')
                    
                    # Color code sentiment
                    if sentiment == 'Positive':
                        st.success(f"**Sentiment:** {sentiment}")
                    elif sentiment == 'Negative':
                        st.error(f"**Sentiment:** {sentiment}")
                    else:
                        st.info(f"**Sentiment:** {sentiment}")
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.write(f"**Impact:** {impact}")
        
        # Add social sharing functionality
        st.markdown("---")
        self.add_social_sharing_section(symbol, analysis_results, sentiment_score, overall_impact, avg_confidence, total_articles)
    
    def add_social_sharing_section(self, symbol, analysis_results, sentiment_score, overall_impact, avg_confidence, total_articles):
        """
        Add social sharing section for sentiment analysis
        
        Args:
            symbol (str): Stock symbol
            analysis_results (list): Analysis results
            sentiment_score (float): Overall sentiment score
            overall_impact (str): Overall investment impact
            avg_confidence (float): Average confidence
            total_articles (int): Total articles analyzed
        """
        st.subheader("📤 Share Your Sentiment Analysis")
        st.markdown("Share your AI-powered sentiment insights with customizable privacy settings")
        
        col_privacy, col_share = st.columns([1, 2])
        
        with col_privacy:
            privacy_level = st.selectbox(
                "Privacy Level:",
                ["public", "anonymized", "private"],
                format_func=lambda x: {
                    "public": "🌐 Public - Full Details",
                    "anonymized": "🔒 Anonymized - No Stock Name", 
                    "private": "🔐 Private - Limited Info"
                }.get(x, x),
                help="Choose how much information to include when sharing",
                key=f"sentiment_privacy_{symbol}"
            )
        
        with col_share:
            if st.button("🚀 Generate Shareable Sentiment Insight", type="primary", key=f"sentiment_share_{symbol}"):
                # Create sentiment insight for sharing
                insight = self.create_sentiment_insight(
                    symbol, analysis_results, sentiment_score, overall_impact, 
                    avg_confidence, total_articles, privacy_level
                )
                
                if insight:
                    # Display the generated insight
                    st.success("✅ Sentiment insight generated!")
                    
                    with st.expander("📋 Preview Your Sentiment Insight", expanded=True):
                        if privacy_level == "public":
                            st.markdown(f"**📰 {symbol} News Sentiment Analysis**")
                            st.markdown(f"**Overall Sentiment Score:** {sentiment_score:+.2f}")
                            st.markdown(f"**Investment Outlook:** {overall_impact}")
                            st.markdown(f"**Confidence Level:** {avg_confidence:.1%}")
                            st.markdown(f"**Articles Analyzed:** {total_articles}")
                            
                            # Show key themes from analysis
                            all_themes = []
                            for result in analysis_results:
                                all_themes.extend(result.get('key_themes', []))
                            
                            if all_themes:
                                theme_counts = {}
                                for theme in all_themes:
                                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
                                top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                                
                                st.markdown("**Top Themes:**")
                                for theme, count in top_themes:
                                    st.markdown(f"• {theme}")
                        
                        elif privacy_level == "anonymized":
                            st.markdown("**📰 Stock News Sentiment Analysis**")
                            st.markdown(f"**Overall Sentiment Score:** {sentiment_score:+.2f}")
                            st.markdown(f"**Investment Outlook:** {overall_impact}")
                            st.markdown(f"**Confidence Level:** {avg_confidence:.1%}")
                        
                        else:  # private
                            st.markdown("**📰 News Sentiment Analysis**")
                            st.markdown(f"**Sentiment Score:** {sentiment_score:+.2f}")
                            st.markdown(f"**Investment Outlook:** {overall_impact}")
                    
                    # Create sharing URLs
                    share_urls = self.create_sentiment_share_urls(insight, privacy_level)
                    
                    # Display sharing options with direct links
                    st.markdown("**Share your analysis:**")
                    
                    # Create sharing buttons with direct links
                    col_whatsapp, col_linkedin, col_email = st.columns(3)
                    
                    with col_whatsapp:
                        st.markdown(f"""
                        <a href="{share_urls['whatsapp']}" target="_blank" style="
                            display: inline-block; 
                            padding: 0.5rem 1rem; 
                            background-color: #25D366; 
                            color: white; 
                            text-decoration: none; 
                            border-radius: 0.25rem;
                            text-align: center;
                            width: 100%;
                        ">📱 Share on WhatsApp</a>
                        """, unsafe_allow_html=True)
                    
                    with col_linkedin:
                        st.markdown(f"""
                        <a href="{share_urls['linkedin']}" target="_blank" style="
                            display: inline-block; 
                            padding: 0.5rem 1rem; 
                            background-color: #0077B5; 
                            color: white; 
                            text-decoration: none; 
                            border-radius: 0.25rem;
                            text-align: center;
                            width: 100%;
                        ">💼 Share on LinkedIn</a>
                        """, unsafe_allow_html=True)
                    
                    with col_email:
                        st.markdown(f"""
                        <a href="{share_urls['email']}" style="
                            display: inline-block; 
                            padding: 0.5rem 1rem; 
                            background-color: #D44638; 
                            color: white; 
                            text-decoration: none; 
                            border-radius: 0.25rem;
                            text-align: center;
                            width: 100%;
                        ">📧 Email Results</a>
                        """, unsafe_allow_html=True)
                    
                    # Store insight in session state
                    if 'sentiment_insights' not in st.session_state:
                        st.session_state.sentiment_insights = []
                    st.session_state.sentiment_insights.append(insight)
                    
                    # Show insight ID for reference
                    st.caption(f"Sentiment Insight ID: {insight['id']} | Generated: {insight['timestamp'][:19]}")
    
    def create_sentiment_insight(self, symbol, analysis_results, sentiment_score, overall_impact, avg_confidence, total_articles, privacy_level):
        """
        Create a shareable sentiment insight
        
        Args:
            symbol (str): Stock symbol
            analysis_results (list): Analysis results
            sentiment_score (float): Overall sentiment score
            overall_impact (str): Overall investment impact
            avg_confidence (float): Average confidence
            total_articles (int): Total articles analyzed
            privacy_level (str): Privacy level for sharing
        
        Returns:
            dict: Formatted insight for sharing
        """
        import uuid
        from datetime import datetime
        
        # Determine sentiment description
        if sentiment_score > 0.3:
            sentiment_desc = "Very Positive"
        elif sentiment_score > 0.1:
            sentiment_desc = "Positive"
        elif sentiment_score > -0.1:
            sentiment_desc = "Neutral"
        elif sentiment_score > -0.3:
            sentiment_desc = "Negative"
        else:
            sentiment_desc = "Very Negative"
        
        # Create base insight
        insight = {
            'id': str(uuid.uuid4())[:8],
            'timestamp': datetime.now().isoformat(),
            'type': 'sentiment_analysis',
            'privacy_level': privacy_level,
            'sentiment_score': sentiment_score,
            'sentiment_description': sentiment_desc,
            'investment_outlook': overall_impact,
            'confidence_level': avg_confidence,
            'articles_analyzed': total_articles
        }
        
        # Format text based on privacy level
        if privacy_level == "public":
            insight['symbol'] = symbol
            insight['formatted_text'] = f"""📰 {symbol} News Sentiment Analysis
Overall Sentiment: {sentiment_desc} ({sentiment_score:+.2f})
Investment Outlook: {overall_impact}
Confidence Level: {avg_confidence:.1%}
Articles Analyzed: {total_articles}

Generated by AI-powered sentiment analysis"""
        
        elif privacy_level == "anonymized":
            insight['formatted_text'] = f"""📰 Stock News Sentiment Analysis
Overall Sentiment: {sentiment_desc} ({sentiment_score:+.2f})
Investment Outlook: {overall_impact}
Confidence Level: {avg_confidence:.1%}
Articles Analyzed: {total_articles}

Generated by AI-powered sentiment analysis"""
        
        else:  # private
            insight['formatted_text'] = f"""📰 News Sentiment Analysis
Sentiment Score: {sentiment_score:+.2f}
Investment Outlook: {overall_impact}
Confidence: {avg_confidence:.1%}

AI-powered analysis"""
        
        return insight
    
    def create_sentiment_share_urls(self, insight, privacy_level):
        """
        Create sharing URLs for sentiment analysis
        
        Args:
            insight (dict): Sentiment insight data
            privacy_level (str): Privacy level
        
        Returns:
            dict: Sharing URLs for different platforms
        """
        import urllib.parse
        
        # Prepare sharing text
        share_text = insight['formatted_text']
        
        # Create URLs
        whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(share_text)}"
        linkedin_url = f"https://www.linkedin.com/feed/?shareActive=true&text={urllib.parse.quote(share_text)}"
        email_subject = "News Sentiment Analysis Results"
        email_url = f"mailto:?subject={urllib.parse.quote(email_subject)}&body={urllib.parse.quote(share_text)}"
        
        return {
            'whatsapp': whatsapp_url,
            'linkedin': linkedin_url,
            'email': email_url,
            'text': share_text
        }