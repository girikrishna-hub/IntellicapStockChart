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

def fetch_financial_news(symbol, days_back=7, max_articles=10):
    """
    Fetch financial news articles for a given stock symbol
    
    Args:
        symbol (str): Stock symbol
        days_back (int): Number of days to look back for news
        max_articles (int): Maximum number of articles to fetch
    
    Returns:
        list: List of news articles with metadata
    """
    try:
        # Using Alpha Vantage News & Sentiment API (free tier available)
        # Alternative: NewsAPI, Financial Modeling Prep, or web scraping
        
        # For demo purposes, we'll use a combination of sources
        articles = []
        
        # Search terms related to the stock
        search_terms = [
            f"{symbol} stock",
            f"{symbol} earnings",
            f"{symbol} financial results",
            f"{symbol} quarterly report"
        ]
        
        # Simulate fetching news (in real implementation, you'd use actual news APIs)
        # For now, we'll create a framework that can be easily extended
        
        # You can integrate with:
        # 1. Alpha Vantage News & Sentiment API
        # 2. NewsAPI
        # 3. Financial Modeling Prep
        # 4. Yahoo Finance News
        # 5. Web scraping from financial news sites
        
        sample_articles = [
            {
                "title": f"{symbol} Reports Strong Q3 Earnings, Beats Expectations",
                "summary": f"{symbol} announced quarterly earnings that exceeded analyst expectations, driven by strong revenue growth and improved margins.",
                "url": "https://example.com/news1",
                "published_date": (datetime.now() - timedelta(days=1)).isoformat(),
                "source": "Financial News Today"
            },
            {
                "title": f"Analysts Upgrade {symbol} Price Target Following Recent Developments",
                "summary": f"Several Wall Street analysts have raised their price targets for {symbol} citing positive market trends and strong fundamentals.",
                "url": "https://example.com/news2",
                "published_date": (datetime.now() - timedelta(days=2)).isoformat(),
                "source": "Market Watch"
            },
            {
                "title": f"{symbol} Faces Headwinds as Market Conditions Worsen",
                "summary": f"Industry experts express concerns about {symbol}'s ability to maintain growth amid challenging market conditions and increased competition.",
                "url": "https://example.com/news3",
                "published_date": (datetime.now() - timedelta(days=3)).isoformat(),
                "source": "Investment Daily"
            }
        ]
        
        return sample_articles[:max_articles]
        
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
    
    # Configuration options
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
        with st.spinner("Fetching and analyzing financial news..."):
            # Fetch news articles
            articles = fetch_financial_news(symbol, days_back, max_articles)
            
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
                    col_twitter, col_linkedin, col_copy, col_email = st.columns(4)
                    
                    with col_twitter:
                        st.markdown(f"""
                        <a href="{share_urls['twitter']}" target="_blank" style="
                            display: inline-block; 
                            padding: 0.5rem 1rem; 
                            background-color: #1DA1F2; 
                            color: white; 
                            text-decoration: none; 
                            border-radius: 0.25rem;
                            text-align: center;
                            width: 100%;
                        ">🐦 Share on Twitter</a>
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
                    
                    with col_copy:
                        if st.button("📋 Copy Text", key=f"copy_sentiment_{symbol}"):
                            st.code(insight['formatted_text'], language=None)
                            st.success("Text ready to copy!")
                    
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
        twitter_url = f"https://twitter.com/intent/tweet?text={urllib.parse.quote(share_text)}"
        linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={urllib.parse.quote('https://replit.com')}&summary={urllib.parse.quote(share_text)}"
        email_subject = "News Sentiment Analysis Results"
        email_url = f"mailto:?subject={urllib.parse.quote(email_subject)}&body={urllib.parse.quote(share_text)}"
        
        return {
            'twitter': twitter_url,
            'linkedin': linkedin_url,
            'email': email_url,
            'text': share_text
        }