import streamlit as st
import pandas as pd
import json
import requests
import time
import plotly.express as px
import matplotlib.pyplot as plt # <--- New import for plotting
from wordcloud import WordCloud # <--- New import for the bonus

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Brand Monitor 2023", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_json("brand_reputation_2023.json")
        
        # Robust date parsing
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        return df

    except ValueError:
        st.error("❌ Format Error: The JSON file structure isn't a list of records.")
        return pd.DataFrame()
    except FileNotFoundError:
        st.error("❌ File 'brand_reputation_2023.json' not found.")
        return pd.DataFrame()

# --- 3. AI ANALYSIS (API MODE) ---
def query_sentiment_api(text_list):
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    
    results = []
    # Only show progress bar if we actually have text to analyze
    if text_list:
        my_bar = st.progress(0, text="Analyzing with Cloud AI...")
    
    for i, text in enumerate(text_list):
        try:
            response = requests.post(API_URL, json={"inputs": text})
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                top_result = data[0][0]
                results.append(top_result)
            else:
                results.append({'label': 'NEUTRAL', 'score': 0.5})
                
        except Exception:
            results.append({'label': 'NEUTRAL', 'score': 0.0})
            
        if text_list:
            my_bar.progress((i + 1) / len(text_list))
        time.sleep(0.1) 
        
    if text_list:
        my_bar.empty()
    return results

df = load_data()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"])

# --- 5. MAIN PAGE LOGIC ---
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
        # Filter Data
        reviews_month = df[
            (df['type'] == 'review') & 
            (df['date'].dt.month == selected_month_index) & 
            (df['date'].dt.year == 2023)
        ].copy()

        if reviews_month.empty:
            st.warning(f"No reviews found for {selected_month_name} 2023.")
        else:
            # --- API LIMITATION HANDLER ---
            # Analyze only the first 10 reviews to prevent timeouts during demo
            reviews_to_analyze = reviews_month.head(10)
            
            # Run Sentiment Analysis
            texts = reviews_to_analyze['text'].fillna('').astype(str).tolist()
            predictions = query_sentiment_api(texts)
            
            reviews_to_analyze['sentiment'] = [p['label'] for p in predictions]
            reviews_to_analyze['confidence'] = [p.get('score', 0) for p in predictions]

            # --- VISUALIZATION LAYOUT ---
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

            # --- BONUS: WORD CLOUD SECTION ---
            st.divider()
            st.subheader(f"☁️ Word Cloud for {selected_month_name}")
            
            # 1. Combine all review text into one big string
            all_text = " ".join(reviews_month['text'].astype(str))
            
            if len(all_text) > 0:
                # 2. Generate the cloud
                # we use a white background because Streamlit handles dark mode differently
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                
                # 3. Display using Matplotlib
                fig_cloud, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off") # Turn off axis numbers
                st.pyplot(fig_cloud)
            else:
                st.info("Not enough text to generate a word cloud.")


