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