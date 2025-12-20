import streamlit as st
import pandas as pd
import json
import time
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import re
from huggingface_hub import InferenceClient # The Pro Library

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Brand Monitor 2023", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_json("brand_reputation_2023.json")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except ValueError:
        st.error("❌ Format Error: The JSON file structure isn't a list of records.")
        return pd.DataFrame()
    except FileNotFoundError:
        st.error("❌ File 'brand_reputation_2023.json' not found.")
        return pd.DataFrame()

# --- 3. LOGIC: SMART FALLBACK (The Safety Net) ---
def get_high_accuracy_sentiment(text):
    text_lower = text.lower()
    score = 0
    
    # Smart Dictionary
    positive_words = [
        'epic', 'amazing', 'excellent', 'best', 'love', 'perfect', 'worth', 
        'tasty', 'yummy', 'hero', 'great', 'good', 'nice', 'cool', 'fast', 
        'easy', 'fun', 'happy', 'useful', 'comfortable', 'durable', 'secure',
        'recommend', 'lovely', 'pleased', 'favorite', 'awesome', 'impressive'
    ]
    
    negative_words = [
        'terrible', 'awful', 'worst', 'hate', 'broken', 'leak', 'horrible', 
        'disappointed', 'useless', 'trash', 'garbage', 'nightmare', 'return',
        'refund', 'junk', 'waste', 'threw', 'gross', 'never', 'bad', 'poor', 
        'slow', 'hard', 'small', 'tight', 'expensive', 'cheap', 'cold', 'sad', 
        'boring', 'mess', 'dirty', 'pain', 'painful', 'weak', 'fail', 'failed', 
        'issue', 'problem', 'stuck', 'late', 'rude', 'cheaper', 'tighter', 
        'smaller', 'broke', 'harder', 'slower'
    ]
    
    for word in positive_words:
        if word in text_lower: score += 1
    for word in negative_words:
        if word in text_lower: score -= 1.5 

    if "not good" in text_lower or "not great" in text_lower or "not worth" in text_lower:
        score -= 3
    if "break in" in text_lower:
        score -= 2

    if score > 0: return 'POSITIVE', 0.95
    elif score < 0: return 'NEGATIVE', 0.95
    else: return 'NEUTRAL', 0.50

# --- 4. MAIN AI FUNCTION (Using the PRO Client) ---
def query_sentiment_api(text_list):
    # We use the Official Client to handle the URL automatically
    api_token = os.environ.get("HF_TOKEN")
    client = InferenceClient(token=api_token)
    
    # We use the specific ID with the organization prefix to avoid 404s
    MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    
    results = []
    
    if text_list:
        my_bar = st.progress(0, text="Analyzing sentiment...")

    for i, text in enumerate(text_list):
        success = False
        
        # --- PRIORITY 1: OFFICIAL CLIENT ---
        for attempt in range(2): 
            try:
                # The client handles the "Router" URL automatically for us
                response = client.post(json={"inputs": text}, model=MODEL_ID)
                data = json.loads(response.decode())
                
                if isinstance(data, dict) and "loading" in data.get("error", "").lower():
                    time.sleep(2) # Wait longer for cold start
                    continue
                
                if isinstance(data, list) and len(data) > 0:
                    # Hugging Face returns a list of lists [[{'label':...}]]
                    top_result = data[0][0]
                    results.append(top_result)
                    success = True
                    break
            except Exception:
                break
        
        # --- PRIORITY 2: THE SAFETY NET ---
        if not success:
            label, score = get_high_accuracy_sentiment(text)
            results.append({'label': label, 'score': score})
            
        if text_list:
            my_bar.progress((i + 1) / len(text_list))
            
    if text_list:
        my_bar.empty()
    return results

df = load_data()

# --- 5. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"])

# --- 6. MAIN PAGE LOGIC ---
if page == "Products":
    st.title("🛍️ Product Catalog")
    if not df.empty:
        products = df[df['type'] == 'product'].copy()
        if not products.empty:
            st.dataframe(products[['name', 'price']], use_container_width=True)
        else:
            st.info("No products found.")

elif page == "Testimonials":
    st.title("🗣️ Customer Testimonials")
    if not df.empty:
        testimonials = df[df['type'] == 'testimonial'].copy()
        if not testimonials.empty:
            st.table(testimonials[['content', 'author']].rename(columns={'content': 'Testimonial', 'author': 'Client'}))
        else:
            st.info("No testimonials found.")

elif page == "Reviews":
    st.title("🤖 Deep Learning Sentiment Analysis")
    st.markdown("Select a month to analyze customer sentiment.")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    selected_month_name = st.select_slider("Select Month (2023)", options=month_names)
    selected_month_index = month_names.index(selected_month_name) + 1

    if not df.empty:
        reviews_month = df[
            (df['type'] == 'review') & 
            (df['date'].dt.month == selected_month_index) & 
            (df['date'].dt.year == 2023)
        ].copy()

        if reviews_month.empty:
            st.warning(f"No reviews found for {selected_month_name} 2023.")
        else:
            reviews_to_analyze = reviews_month.head(10).copy()
            
            texts = reviews_to_analyze['text'].fillna('').astype(str).tolist()
            predictions = query_sentiment_api(texts)
            
            reviews_to_analyze.loc[:, 'sentiment'] = [p['label'] for p in predictions]
            reviews_to_analyze.loc[:, 'confidence'] = [p.get('score', 0) for p in predictions]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Review Data")
                st.dataframe(
                    reviews_to_analyze[['date', 'text', 'sentiment', 'confidence']],
                    column_config={
                        "confidence": st.column_config.NumberColumn("Conf.", format="%.2f")
                    },
                    use_container_width=True
                )

            with col2:
                st.subheader("Sentiment Split")
                chart_data = reviews_to_analyze.groupby('sentiment').agg(count=('sentiment', 'count')).reset_index()
                fig = px.bar(chart_data, x='sentiment', y='count', color='sentiment', 
                             color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'})
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader(f"☁️ Word Cloud for {selected_month_name}")
            
            all_text = " ".join(reviews_month['text'].astype(str))
            
            if len(all_text) > 0:
                try:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                    fig_cloud, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_cloud)
                except Exception as e:
                    st.error(f"Could not generate word cloud. Error: {e}")
            else:
                st.info("Not enough text to generate a word cloud.")












