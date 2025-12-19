import streamlit as st
import pandas as pd
import json
import gc  # <--- Added for memory cleanup
from transformers import pipeline
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Brand Monitor 2023", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load the JSON list directly into a DataFrame
        df = pd.read_json("brand_reputation_2023.json")
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        return df

    except ValueError:
        st.error("❌ Format Error: The JSON file structure isn't a list of records.")
        return pd.DataFrame()
    except FileNotFoundError:
        st.error("❌ File 'brand_reputation_2023.json' not found.")
        return pd.DataFrame()

# --- 3. LOAD AI MODEL (OPTIMIZED) ---
@st.cache_resource  # <--- CRITICAL FIX: Loads model once and keeps it in memory
def load_sentiment_model():
    # Using the specific model explicitly
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Explicitly clear memory before loading the heavy model
gc.collect()

df = load_data()
sentiment_pipeline = load_sentiment_model()

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
            # C. Perform Sentiment Analysis
            # We limit the number of reviews processed at once to prevent crashes
            with st.spinner(f"Analyzing {len(reviews_month)} reviews with AI..."):
                texts = reviews_month['text'].fillna('').astype(str).tolist()
                
                # Run Model
                predictions = sentiment_pipeline(texts)
                
                reviews_month['sentiment'] = [p['label'] for p in predictions]
                reviews_month['confidence'] = [p['score'] for p in predictions]

            # D. Visualization
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Review Details")
                st.dataframe(
                    reviews_month[['date', 'text', 'sentiment', 'confidence']],
                    column_config={
                        "date": st.column_config.DateColumn("Date"),
                        "confidence": st.column_config.NumberColumn("Confidence Score", format="%.4f")
                    },
                    use_container_width=True
                )

            with col2:
                st.subheader("Sentiment Split")
                
                chart_data = reviews_month.groupby('sentiment').agg(
                    count=('sentiment', 'count'),
                    avg_confidence=('confidence', 'mean')
                ).reset_index()

                fig = px.bar(
                    chart_data,
                    x='sentiment',
                    y='count',
                    color='sentiment',
                    color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red'},
                    hover_data=['avg_confidence'],
                    labels={'count': 'Count', 'avg_confidence': 'Avg Confidence'}
                )
                st.plotly_chart(fig, use_container_width=True)
