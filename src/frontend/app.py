import streamlit as st
import requests
from PIL import Image
import io
import json
import pandas as pd
import plotly.express as px
import numpy as np
from typing import Dict, Any
import logging
import pytesseract
import re
from openai import OpenAI, OpenAIError
import boto3
from botocore.exceptions import ClientError
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000"
SUPPORTED_DOCUMENT_TYPES = ["invoice", "receipt", "form"]

api_key = os.getenv("PERPLEXITY_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

REGION = "us-east-1"
dynamodb = boto3.resource('dynamodb', region_name=REGION)

def extract_json_from_llm_output(llm_result):
    match = re.search(r'\{.*\}', llm_result, re.DOTALL)
    if match:
        return match.group(0)
    return None

def save_to_dynamodb(data, table_name="Receipts"):
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)
    try:
        table.put_item(Item=data)
        return True
    except ClientError as e:
        st.error(f"Failed to save to DynamoDB: {e}")
        return False

def main():
    st.set_page_config(
        page_title="FormIQ - Intelligent Document Parser",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("FormIQ: Intelligent Document Parser")
    st.markdown("""
    Upload your documents to extract and validate information using advanced AI models.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        document_type = st.selectbox(
            "Document Type",
            options=SUPPORTED_DOCUMENT_TYPES,
            index=0
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        FormIQ uses LayoutLMv3 and GPT-4 to extract and validate information from documents.
        """)
    
    # Main content
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload a document image to process"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document", width=600)

        # Process button
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Save the uploaded file to a temporary location
                    temp_path = "temp_uploaded_image.jpg"
                    image.save(temp_path)

                    # Extract fields using OCR + regex
                    fields = extract_fields(temp_path)

                    # Extract with Perplexity LLM using the provided API key
                    with st.spinner("Extracting structured data with Perplexity LLM..."):
                        try:
                            llm_result = extract_with_perplexity_llm(pytesseract.image_to_string(Image.open(temp_path)))
                            st.subheader("Structured Data (Perplexity LLM)")
                            st.code(llm_result, language="json")

                            # Extract and save JSON to DynamoDB
                            raw_json = extract_json_from_llm_output(llm_result)
                            if raw_json:
                                try:
                                    llm_data = json.loads(raw_json)
                                    if save_to_dynamodb(llm_data):
                                        st.success("Data saved to DynamoDB!")
                                except Exception as e:
                                    st.error(f"Failed to parse/save JSON: {e}")
                            else:
                                st.error("No valid JSON found in LLM output.")
                        except Exception as e:
                            st.error(f"LLM extraction failed: {e}")

                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    st.error(f"Error processing document: {str(e)}")

def display_results(results: Dict[str, Any]):
    """Display extraction and validation results."""
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Extracted Fields", "Validation", "Visualization"])
    
    with tab1:
        st.subheader("Extracted Fields")
        if "fields" in results["extraction"]:
            fields_df = pd.DataFrame(results["extraction"]["fields"])
            st.dataframe(fields_df)
        else:
            st.info("No fields extracted")
    
    with tab2:
        st.subheader("Validation Results")
        validation = results["validation"]
        
        # Display validation status
        status_color = "green" if validation["is_valid"] else "red"
        st.markdown(f"### Status: :{status_color}[{validation['is_valid']}]")
        
        # Display validation errors if any
        if validation["validation_errors"]:
            st.error("Validation Errors:")
            for error in validation["validation_errors"]:
                st.markdown(f"- {error}")
        
        # Display confidence score
        st.metric(
            "Overall Confidence",
            f"{validation['confidence_score']:.2%}"
        )
    
    with tab3:
        st.subheader("Confidence Visualization")
        if "confidence_scores" in results["extraction"]["metadata"]:
            scores = results["extraction"]["metadata"]["confidence_scores"]
            
            # Create confidence distribution plot
            fig = px.histogram(
                x=scores,
                nbins=20,
                title="Confidence Score Distribution",
                labels={"x": "Confidence Score", "y": "Count"}
            )
            st.plotly_chart(fig)
            
            # Display heatmap if available
            if "bbox" in results["extraction"]["fields"][0]:
                st.subheader("Field Location Heatmap")
                # TODO: Implement heatmap visualization
                st.info("Heatmap visualization coming soon!")

def group_tokens_by_label(tokens, labels):
    structured = {}
    current_label = None
    current_tokens = []
    for token, label in zip(tokens, labels):
        if label != current_label:
            if current_label is not None:
                structured.setdefault(current_label, []).append(' '.join(current_tokens))
            current_label = label
            current_tokens = [token]
        else:
            current_tokens.append(token)
    if current_label is not None:
        structured.setdefault(current_label, []).append(' '.join(current_tokens))
    return structured

def extract_fields(image_path):
    # OCR
    text = pytesseract.image_to_string(Image.open(image_path))
    
    # Display OCR output for debugging
    st.subheader("Raw OCR Output (for debugging)")
    st.code(text)

    # Improved Regex patterns for fields
    patterns = {
        "name": r"Mrs\s+\w+\s+\w+",
        "date": r"Date[:\s]+([\d/]+)",
        "product": r"\d+\s+\w+.*Style\s+\d+",
        "amount_paid": r"Total Paid\s+\$?([\d.,]+)",
        # Improved pattern for receipt number (handles optional dot, colon, spaces)
        "receipt_no": r"Receipt No\.?\s*:?\s*(\d+)"
    }

    results = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            results[field] = match.group(1) if match.groups() else match.group(0)
        else:
            results[field] = None

    return results

def extract_with_perplexity_llm(ocr_text):
    prompt = f"""
Extract the following fields from this receipt text:
- name
- date
- product
- amount_paid
- receipt_no

Text:
\"\"\"{ocr_text}\"\"\"

Return the result as a JSON object with those fields.
"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant. "
                "Answer user questions as concisely and directly as possible. "
                "Limit your responses to 2-3 sentences unless the user asks for more detail."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model="sonar-pro",  # Use a valid model name for your account
        messages=messages,
    )
    return response.choices[0].message.content

def interactive_chatbot_ui():
    st.header("🤖 Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history as chat bubbles
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"<div style='text-align: right; background: #262730; color: #fff; padding: 8px 12px; border-radius: 12px; margin: 4px 0 4px 40px;'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; background: #31333F; color: #fff; padding: 8px 12px; border-radius: 12px; margin: 4px 40px 4px 0;'><b>Bot:</b> {msg}</div>", unsafe_allow_html=True)

    # Input at the bottom
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...", key="chat_input_main", placeholder="Ask me anything...")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            st.session_state.chat_history.append(("You", user_input))
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": user_input}
                )
                if response.status_code == 200:
                    bot_reply = response.json()["answer"]
                else:
                    bot_reply = f"Error: Server returned status code {response.status_code}"
            except Exception as e:
                bot_reply = f"Error: {e}"
            st.session_state.chat_history.append(("Bot", bot_reply))

if __name__ == "__main__":
    
    main()
    st.markdown("---")
    interactive_chatbot_ui() 