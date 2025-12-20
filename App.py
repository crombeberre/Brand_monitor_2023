import streamlit as st
import pandas as pd
import time
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from huggingface_hub import InferenceClient

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

# --- 3. PURE AI ANALYSIS (Official Client) ---
def query_sentiment_api(text_list):
    # Retrieve the Token
    api_token = os.environ.get("HF_TOKEN")
    
    if not api_token:
        st.error("⚠️ HF_TOKEN is missing in Render Environment Variables!")
        return []

    # Initialize the Official Client
    client = InferenceClient(token=api_token)
    
    # We use the standard model ID. The client handles the URL routing automatically.
    MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    
    results = []
    
    if text_list:
        my_bar = st.progress(0, text="Connecting to Hugging Face AI...")

    for i, text in enumerate(text_list):
        success = False
        
        # Retry logic for "Model Loading" (Cold Boot)
        for attempt in range(5): 
            try:
                # Use the built-in helper function for classification
                # This returns a list of scores: [{'label': 'POSITIVE', 'score': 0.99}, ...]
                response = client.text_classification(text, model=MODEL_ID)
                
                if not response:
                    continue

                # Get the top label (highest score)
                top_result = response[0]
                
                # Standardize Label
                label = top_result.label.upper()
                score = top_result.score
                
                results.append({'label': label, 'score': score})
                success = True
                break
                
            except Exception as e:
                # If the error is just "Model is loading", we wait.
                error_str = str(e).lower()
                if "loading" in error_str or "503" in error_str:
                    time.sleep(3)
                    continue
                else:
                    # If it's a REAL error (Authentication, 404), we print it on screen.
                    # st.warning(f"API Error on item {i}: {e}") # Uncomment to debug
                    time.sleep(1)
                    continue
        
        # If AI fails after 5 attempts, we DO NOT fake it. We show "ERROR".
        if not success:
            results.append({'label': 'ERROR', 'score': 0.00})
            
        if text_list:
            my_bar.progress((i + 1) / len(text_list))
            
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
            
            # CALL THE REAL AI
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
                
                # Dynamic Colors: Green for Positive, Red for Negative, Grey for Error
                fig = px.bar(chart_data, x='sentiment', y='count', color='sentiment', 
                             color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'ERROR': 'gray'})
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














