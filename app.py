import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import time
from datetime import datetime
import base64
from io import BytesIO
from streamlit_lottie import st_lottie
import requests

# Download NLTK data (with error handling for deployment)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except:
        return False

# Add this function after your other @st.cache_resource functions
@st.cache_resource
def load_lottie_url(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

@st.cache_resource
def load_header_animation():
    """Load header animation"""
    # Data analysis animation URL
    animation_url = 'https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json'
    return load_lottie_url(animation_url)

# Load the header animation
header_animation = load_header_animation()

# Initialize NLTK
download_nltk_data()

# Page config
st.set_page_config(
    page_title="🎭 Sentiment Analysis AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sentiment-positive {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .sentiment-negative {
        background: linear-gradient(45deg, #cb2d3e, #ef473a);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .sentiment-neutral {
        background: linear-gradient(45deg, #bdc3c7, #2c3e50);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load('sentiment_analysis_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please make sure 'sentiment_analysis_model.pkl' is in the app directory.")
        return None

# Preprocessing function (same as training)
@st.cache_resource
def init_preprocessing_tools():
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        return lemmatizer, stop_words
    except:
        # Fallback if NLTK data not available
        return None, set()

lemmatizer, stop_words = init_preprocessing_tools()

def preprocess_text(text):
    """Preprocess text exactly as done during training"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    
    if lemmatizer and stop_words:
        words = [lemmatizer.lemmatize(word) for word in words 
                 if word not in stop_words and len(word) > 2]
    else:
        words = [word for word in words if len(word) > 2]
    
    return ' '.join(words)

def predict_sentiment(text, model):
    """Predict sentiment with preprocessing"""
    if not model:
        return None
    
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        return {
            'sentiment': 'Neutral',
            'confidence': 0.33,
            'probabilities': {'Positive': 0.33, 'Neutral': 0.34, 'Negative': 0.33}
        }
    
    try:
        prediction = model.predict([cleaned_text])[0]
        probabilities = model.predict_proba([cleaned_text])[0]
        classes = model.classes_
        confidence = max(probabilities)
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': dict(zip(classes, probabilities)),
            'cleaned_text': cleaned_text
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment"""
    emoji_map = {
        'Positive': '😊',
        'Negative': '😞',
        'Neutral': '😐'
    }
    return emoji_map.get(sentiment, '🤔')

def get_sentiment_color(sentiment):
    """Get color for sentiment"""
    color_map = {
        'Positive': '#56ab2f',
        'Negative': '#cb2d3e',
        'Neutral': '#bdc3c7'
    }
    return color_map.get(sentiment, '#gray')

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load the model
model = load_model()

# Main header
# Enhanced header with animation
col_anim, col_title, col_spacer = st.columns([1, 4, 1])

with col_anim:
    if header_animation:
        st_lottie(
            header_animation, 
            height=120, 
            width=120, 
            key="header_animation",
            speed=1
        )
    else:
        st.markdown('<div style="font-size: 4rem; text-align: center;">📊</div>', unsafe_allow_html=True)

with col_title:
    st.markdown('<h1 class="main-header">🎭 AI Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by Advanced NLP & Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🚀 App Features")
    st.markdown("""
    - **Real-time Analysis**: Instant sentiment detection
    - **Confidence Scores**: Probability breakdown
    - **History Tracking**: View past predictions
    - **Batch Processing**: Analyze multiple texts
    - **Interactive Charts**: Beautiful visualizations
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Model Info")
    if model:
        st.success("✅ Model Loaded Successfully")
        st.info("🤖 Algorithm: Optimized ML Pipeline")
        st.info("📝 Features: TF-IDF + N-grams")
    else:
        st.error("❌ Model Not Available")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Analysis", "📋 Batch Analysis", "📈 Analytics", "🕒 History"])

with tab1:
    st.markdown("### 💬 Analyze Your Text")
    
    # Text input methods
    input_method = st.radio("Choose input method:", ["✍️ Type Text", "📎 Upload File"], horizontal=True)
    
    if input_method == "✍️ Type Text":
        user_input = st.text_area(
            "Enter your text here:",
            placeholder="Type or paste your text here... (e.g., 'I love this product!', 'This service is terrible', etc.)",
            height=120
        )
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file:
            user_input = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", user_input, height=120, disabled=True)
        else:
            user_input = ""
    
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input and model:
            with st.spinner("🤖 Analyzing sentiment..."):
                time.sleep(0.5)  # Add small delay for effect
                result = predict_sentiment(user_input, model)
                
                if result:
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    probabilities = result['probabilities']
                    
                    # Add to history
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                        'sentiment': sentiment,
                        'confidence': confidence
                    })
                    
                    # Display results
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col2:
                        emoji = get_sentiment_emoji(sentiment)
                        st.markdown(f"""
                        <div style="text-align: center; font-size: 4rem; margin: 1rem 0;">
                            {emoji}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Sentiment result with styling
                    sentiment_class = f"sentiment-{sentiment.lower()}"
                    st.markdown(f"""
                    <div class="{sentiment_class}">
                        <h2>{sentiment.upper()} SENTIMENT</h2>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.markdown("### 📊 Confidence Breakdown")
                    
                    # Create probability chart
                    prob_df = pd.DataFrame(list(probabilities.items()), columns=['Sentiment', 'Probability'])
                    
                    fig = px.bar(
                        prob_df, 
                        x='Sentiment', 
                        y='Probability',
                        color='Sentiment',
                        color_discrete_map={
                            'Positive': '#56ab2f',
                            'Negative': '#cb2d3e',
                            'Neutral': '#bdc3c7'
                        },
                        title="Sentiment Probability Distribution"
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Text analysis insights
                    with st.expander("🔍 Text Analysis Insights"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Word Count", len(user_input.split()))
                        with col2:
                            st.metric("Character Count", len(user_input))
                        with col3:
                            # Simple polarity using TextBlob as additional insight
                            try:
                                polarity = TextBlob(user_input).sentiment.polarity
                                st.metric("Polarity", f"{polarity:.2f}")
                            except:
                                st.metric("Polarity", "N/A")
                        with col4:
                            st.metric("Cleaned Words", len(result.get('cleaned_text', '').split()))
                        
                        st.markdown("**Preprocessed Text:**")
                        st.code(result.get('cleaned_text', 'N/A'))
        
        elif not user_input:
            st.warning("⚠️ Please enter some text to analyze!")
        else:
            st.error("❌ Model not available. Please check the model file.")

with tab2:
    st.markdown("### 📋 Batch Analysis")
    st.markdown("Analyze multiple texts at once")
    
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        placeholder="This product is amazing!\nI hate this service.\nIt's okay, nothing special.",
        height=200
    )
    
    if st.button("🔍 Analyze All", type="primary"):
        if batch_input and model:
            texts = [text.strip() for text in batch_input.split('\n') if text.strip()]
            
            if texts:
                results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(texts):
                    result = predict_sentiment(text, model)
                    if result:
                        results.append({
                            'Text': text[:50] + "..." if len(text) > 50 else text,
                            'Sentiment': result['sentiment'],
                            'Confidence': f"{result['confidence']:.1%}",
                            'Emoji': get_sentiment_emoji(result['sentiment'])
                        })
                    progress_bar.progress((i + 1) / len(texts))
                
                if results:
                    st.markdown("### 📊 Batch Results")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    sentiment_counts = df['Sentiment'].value_counts()
                    
                    with col1:
                        st.metric("Total Analyzed", len(results))
                    with col2:
                        st.metric("Most Common", sentiment_counts.index[0] if len(sentiment_counts) > 0 else "N/A")
                    with col3:
                        avg_conf = df['Confidence'].str.rstrip('%').astype(float).mean()
                        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                    
                    # Batch visualization
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={
                            'Positive': '#56ab2f',
                            'Negative': '#cb2d3e',
                            'Neutral': '#bdc3c7'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 📈 Analytics Dashboard")
    
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(hist_df))
        with col2:
            avg_confidence = hist_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        with col3:
            most_common = hist_df['sentiment'].mode().iloc[0] if len(hist_df) > 0 else "N/A"
            st.metric("Most Common", most_common)
        with col4:
            high_conf = (hist_df['confidence'] > 0.8).sum()
            st.metric("High Confidence", f"{high_conf}/{len(hist_df)}")
        
        # Sentiment distribution over time
        hist_df['date'] = hist_df['timestamp'].dt.date
        daily_sentiment = hist_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        fig = px.bar(
            daily_sentiment,
            x='date',
            y='count',
            color='sentiment',
            title="Sentiment Analysis Over Time",
            color_discrete_map={
                'Positive': '#56ab2f',
                'Negative': '#cb2d3e',
                'Neutral': '#bdc3c7'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        fig2 = px.histogram(
            hist_df,
            x='confidence',
            nbins=20,
            title="Confidence Score Distribution",
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.info("📊 No analysis history yet. Start analyzing some texts to see analytics!")

with tab4:
    st.markdown("### 🕒 Analysis History")
    
    if st.session_state.history:
        # Show recent analyses
        for i, item in enumerate(reversed(st.session_state.history[-10:])):  # Show last 10
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{item['text']}**")
                    st.caption(f"🕐 {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                with col2:
                    emoji = get_sentiment_emoji(item['sentiment'])
                    st.markdown(f"## {emoji}")
                    st.caption(item['sentiment'])
                
                with col3:
                    st.metric("Confidence", f"{item['confidence']:.1%}")
                
                st.divider()
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("📝 No analysis history yet. Start analyzing some texts!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚀 Built with Streamlit • 🤖 Powered by Machine Learning • 💡 Made with ❤️</p>
</div>

""", unsafe_allow_html=True)

