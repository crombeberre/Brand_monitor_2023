import streamlit as st
import pandas as pd
import json
import requests
import time
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Brand Monitor 2023", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load the JSON list directly into a DataFrame
        df = pd.read_json("brand_reputation_2023.json")
        
        # FIX: Handle date parsing safely (fixes the UserWarning in your logs)
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
# This method uses 0 RAM because it runs on HuggingFace's cloud, not yours.
def query_sentiment_api(text_list):
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    # Note: For heavy use, you'd need an API Token, but for a few clicks this usually works free.
    
    results = []
    my_bar = st.progress(0, text="Analyzing with Cloud AI...")
    
    for i, text in enumerate(text_list):
        try:
            # We send payload to Hugging Face
            response = requests.post(API_URL, json={"inputs": text})
            data = response.json()
            
            # Formatting the response
            # The API returns a list of lists: [[{'label': 'POSITIVE', 'score': 0.9}]]
            if isinstance(data, list) and len(data) > 0:
                top_result = data[0][0] # Get the first prediction
                results.append(top_result)
            else:
                # Fallback if API is busy/loading
                results.append({'label': 'NEUTRAL', 'score': 0.5})
                
        except Exception:
            results.append({'label': 'NEUTRAL', 'score': 0.0})
            
        # Update progress bar
        my_bar.progress((i + 1) / len(text_list))
        time.sleep(0.1) # Be nice to the free API
        
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
            st.dataframe(
                products[['name', 'price']],
                column_config={
                    "name": "Product Name",
                    "price": st.column_config.NumberColumn("Price ($)", format="$%.2f")
                },
                use_container_width=True 
            )
        else:
            st.info("No products found.")

elif page == "Testimonials":
    st.title("🗣️ Customer Testimonials")
    
    if not df.empty:
        testimonials = df[df['type'] == 'testimonial'].copy()
        if not testimonials.empty:
            st.table(testimonials[['content', 'author']].rename(columns={
                'content': 'Testimonial',
                'author': 'Client'
            }))
        else:
            st.info("No testimonials found.")

elif page == "Reviews":
    st.title("🤖 Deep Learning Sentiment Analysis")
    st.markdown("Select a month to analyze customer sentiment.")

    # A. Month Slider
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    selected_month_name = st.select_slider("Select Month (2023)", options=month_names)
    selected_month_index = month_names.index(selected_month_name) + 1

    if not df.empty:
        # B. Filter Data
        reviews_month = df[
            (df['type'] == 'review') & 
            (df['date'].dt.month == selected_month_index) & 
            (df['date'].dt.year == 2023)
        ].copy()

        if reviews_month.empty:
            st.warning(f"No reviews found for {selected_month_name} 2023.")
        else:
            # Limit to first 5 reviews to prevent API timeout during demo
            reviews_to_analyze = reviews_month.head(5)
            
            if len(reviews_month) > 5:
                st.info(f"⚡ Demo Mode: Analyzing first 5 of {len(reviews_month)} reviews to save time.")
            
            # C. Perform Sentiment Analysis via API
            texts = reviews_to_analyze['text'].fillna('').astype(str).tolist()
            predictions = query_sentiment_api(texts)
            
            reviews_to_analyze['sentiment'] = [p['label'] for p in predictions]
            reviews_to_analyze['confidence'] = [p.get('score', 0) for p in predictions]

            # D. Visualization
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Review Details")
                st.dataframe(
                    reviews_to_analyze[['date', 'text', 'sentiment', 'confidence']],
                    column_config={
                        "date": st.column_config.DateColumn("Date"),
                        "confidence": st.column_config.NumberColumn("Confidence Score", format="%.4f")
                    },
                    use_container_width=True
                )

            with col2:
                st.subheader("Sentiment Split")
                
                chart_data = reviews_to_analyze.groupby('sentiment').agg(
                    count=('sentiment', 'count'),
                    avg_confidence=('confidence', 'mean')
                ).reset_index()

                fig = px.bar(
                    chart_data,
                    x='sentiment',
                    y='count',
                    color='sentiment',
                    color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'},
                    hover_data=['avg_confidence'],
                    labels={'count': 'Count', 'avg_confidence': 'Avg Confidence'}
                )
                st.plotly_chart(fig, use_container_width=True)

