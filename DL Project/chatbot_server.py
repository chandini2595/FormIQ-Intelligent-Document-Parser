import os
from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_API_URL = "https://api.perplexity.ai/v1/chat/completions"

# DynamoDB setup
REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
dynamodb = boto3.resource('dynamodb', region_name=REGION)
table = dynamodb.Table('Receipts')

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_endpoint(chat_request: ChatRequest):
    question = chat_request.question
    receipt_no = extract_receipt_no(question)
    if receipt_no:
        response = table.get_item(Key={'receipt_no': receipt_no})
        item = response.get('Item')
    else:
        item = None
    context = f"Database record: {item}" if item else "No matching record found."
    llm_response = query_perplexity_llm(question, context)
    return {"answer": llm_response}

def extract_receipt_no(question):
    import re
    match = re.search(r'RCPT-\d{4}-\d{4}', question)
    return match.group(0) if match else None

def query_perplexity_llm(question, context):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "pplx-7b-online",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions about receipts."},
            {"role": "user", "content": f"Question: {question}\nContext: {context}"}
        ]
    }
    response = requests.post(PERPLEXITY_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error from LLM: {response.text}" 