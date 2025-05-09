import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import boto3

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client for Perplexity
client = OpenAI(
    api_key=os.getenv('PERPLEXITY_API_KEY'),
    base_url="https://api.perplexity.ai"
)

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_endpoint(chat_request: ChatRequest):
    # Connect to DynamoDB
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('Receipts')
    
    # Get question and search DynamoDB
    question = chat_request.question
    response = table.scan()
    items = response.get('Items', [])
    
    # Format items for context with all receipt details
    context = "\n".join([
        f"Receipt {item['receipt_no']}:\n"
        f"  Name: {item['name']}\n"
        f"  Date: {item['date']}\n"
        f"  Product: {item['product']}\n"
        f"  Amount Paid: {item['amount_paid']}\n"
        for item in items
    ])
    question = f"Based on these receipts:\n{context}\n\nQuestion: {question}\nPlease provide a 2-3 line answer."
    
    # Prepare messages for the chat
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
                "Give a 2-3 line answer."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    try:
        # Get response from Perplexity
        response = client.chat.completions.create(
            model="sonar",
            messages=messages
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"error": f"Error from LLM: {str(e)}"}