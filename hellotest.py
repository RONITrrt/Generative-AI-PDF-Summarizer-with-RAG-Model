import streamlit as st
import torch
import numpy as np
import pandas as pd
import textwrap
from sentence_transformers import util, SentenceTransformer
import google.generativeai as genai
import os

# Set up the API key for Google Gemini
os.environ["API_KEY"] = 'AIzaSyCl-zvslLoqc3E3eL2os0s287SK0tGJJi8'
genai.configure(api_key=os.environ["API_KEY"])

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    return wrapped_text

# Define the prompt template
prompt_template = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.

Example 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.

Example 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.

Example 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.

Now use the following context items to answer the user query:
{context}
Relevant passages: <extract relevant passages from the context here>
Answer:"""

def get_answer(context, query):
    prompt_one = f"Check for every data in {context} and answer specifically to {query}, short and concise."
    
    # Generate the response using Google Gemini
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt_one)
    return response.text

def main():
    st.markdown("<h1 style='text-align: center; color: darkblue;'>GenAI-powered Document Retrieval and Summarization Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Ask a question related to the provided context and get detailed answers powered by Google Gemini AI.</p>", unsafe_allow_html=True)
    
    # Input query from user
    st.subheader("Enter your Question:")
    query = st.text_input("Question related to the PDF:", "")
    
    # Submit button
    if st.button("Submit Query"):
        if query:
            st.markdown("---")
            with st.spinner("Processing your query..."):
                with open('chunked_text.txt', 'r', encoding='utf-8') as text_file:
                    top_dot_products = text_file.read()
                answer = get_answer(top_dot_products, query)
            
            st.subheader("Answer:")
            st.markdown(f"<div style='background-color: #f0f0f5; padding: 10px; border-radius: 5px;'><p style='color: black;'>{answer}</p></div>", unsafe_allow_html=True)
        else:
            st.error("Please enter a query.")
    
if __name__ == "__main__":
    main()
