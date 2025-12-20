import streamlit as st
import requests
import os
import json

st.set_page_config(layout="wide")
st.title("🕵️ API Diagnostic Mode")

# 1. Check the Token
api_token = os.environ.get("HF_TOKEN")
st.write("---")
st.subheader("1. Token Check")
if api_token:
    st.success(f"Token found! Length: {len(api_token)} characters")
    st.info(f"First 4 characters: {api_token[:4]}...")
else:
    st.error("❌ NO TOKEN FOUND. The app is trying to connect anonymously (which fails).")

# 2. Check the API Connection
st.write("---")
st.subheader("2. Connection Test")

# We test with the specific model you are using
API_URL = "https://router.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

test_text = "The design is epic!"

if st.button("🔴 Test Connection Now"):
    with st.spinner("Contacting Hugging Face..."):
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": test_text}, timeout=10)
            
            st.write(f"**Status Code:** `{response.status_code}`")
            
            # Try to read the error message
            try:
                data = response.json()
                st.json(data)
            except:
                st.write("Raw Text Response:")
                st.code(response.text)

            if response.status_code == 200:
                st.success("✅ SUCCESS! The API is working perfectly.")
                st.write("We can switch back to the main code without the dictionary.")
            else:
                st.error("❌ FAILURE. See the error message above.")
                
        except Exception as e:
            st.error(f"Connection Error: {e}")











