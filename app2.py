import pickle
import io
import torch
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import base64
from io import BytesIO
from typing import List, Dict, Any

app = FastAPI(title="YouTube Comment Sentiment Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
loaded_model = None
loaded_tokenizer = None
chrome_driver = None

# Pydantic models
class URLRequest(BaseModel):
    url: str

class Comment(BaseModel):
    index: int
    comment: str
    sentiment: str
    confidence: float

class StatisticsResult(BaseModel):
    total_comments: int
    positive_count: int
    negative_count: int
    positive_percentage: float
    negative_percentage: float
    
class SentimentAnalysisResponse(BaseModel):
    comments: List[Comment]
    statistics: StatisticsResult
    chart_image: str  # Base64 encoded image

# Custom unpickler class to handle tensors saved on CUDA
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to predict sentiment
def predict_sentiment(text, model, tokenizer):
    # Preprocess and tokenize
    preprocessed_text = preprocess_text(text)
    inputs = tokenizer(
        preprocessed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Make prediction
    with torch.no_grad():  # This reduces memory usage
        outputs = model(**inputs)
        logits = outputs.logits
        prob = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = prob[0][prediction].item()
    
    return sentiment, confidence, prediction

# Setup Chrome driver
def setup_chrome_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(service=Service("chromedriver.exe"), options=options)
        return driver
    except Exception as e:
        print(f"Error initializing Chrome driver: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chrome driver initialization failed: {str(e)}")

# Load model function
def load_models():
    global loaded_model, loaded_tokenizer
    
    try:
        model_path = "distilbert_sentiment_model.pkl"
        print(f"Loading model from {model_path}...")
        
        # Use custom unpickler for loading the model to CPU
        with open(model_path, 'rb') as f:
            model_data = CPU_Unpickler(f).load()
        
        # Create model and load state dict
        loaded_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            state_dict=model_data['model_state_dict'],
            config=model_data['config']
        )
        loaded_tokenizer = model_data['tokenizer']
        
        # Set to evaluation mode
        loaded_model.eval()
        
        print("Model loaded successfully!")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Model file '{model_path}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Create a pie chart and return base64 encoded image
def create_pie_chart(positive_percentage, negative_percentage):
    labels = ['Positive', 'Negative']
    sizes = [positive_percentage, negative_percentage]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode the 1st slice (Positive)
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Sentiment Distribution in YouTube Comments')
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Convert to base64
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

# Initialize resources on startup
@app.on_event("startup")
async def startup_event():
    load_models()
    global chrome_driver
    chrome_driver = setup_chrome_driver()
    print("API is ready to use on port 8001!")

# Clean up resources on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    if chrome_driver:
        chrome_driver.quit()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": loaded_model is not None}

# Main endpoint for sentiment analysis
@app.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_youtube_comments(request: URLRequest):
    if not loaded_model or not loaded_tokenizer:
        raise HTTPException(status_code=500, detail="Models not loaded properly")
    
    try:
        video_url = request.url
        
        # Create new driver for each request to avoid session issues
        driver = chrome_driver
        
        print(f"Accessing {video_url}...")
        driver.get(video_url)
        
        print("Loading page content...")
        time.sleep(5)
        driver.execute_script("window.scrollBy(0, 600);")
        time.sleep(5)
        
        # Scroll to load comments
        print("Scrolling to load comments...")
        scroll_count = 0
        while scroll_count < 3:
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(2)
            scroll_count += 1
        
        # Find comments
        print("Extracting comments...")
        comment_spans = driver.find_elements(By.XPATH, '//ytd-comment-thread-renderer//ytd-expander//span')
        
        # Extract text
        comments = [span.text.strip() for span in comment_spans if span.text.strip()]
        comments = comments[:20]  # Limit to 20
        
        if not comments:
            raise HTTPException(status_code=404, detail="No comments found for this video")
            
        print(f"Found {len(comments)} comments.")
        
        # Analyze sentiments
        print("\nAnalyzing sentiments...")
        results = []
        
        for i, comment in enumerate(comments, 1):
            sentiment, confidence, prediction = predict_sentiment(comment, loaded_model, loaded_tokenizer)
            results.append({
                'index': i, 
                'comment': comment, 
                'sentiment': sentiment, 
                'confidence': confidence,
                'prediction': prediction  # 0 for negative, 1 for positive
            })
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Calculate statistics
        positive_count = sum(df['prediction'])
        negative_count = len(df) - positive_count
        positive_percentage = (positive_count / len(df)) * 100
        negative_percentage = (negative_count / len(df)) * 100
        
        # Create chart
        chart_image = create_pie_chart(positive_percentage, negative_percentage)
        
        # Prepare response
        comment_results = [
            Comment(
                index=row['index'],
                comment=row['comment'],
                sentiment=row['sentiment'],
                confidence=row['confidence']
            ) for _, row in df.iterrows()
        ]
        
        statistics = StatisticsResult(
            total_comments=len(comments),
            positive_count=positive_count,
            negative_count=negative_count,
            positive_percentage=positive_percentage,
            negative_percentage=negative_percentage
        )
        
        return SentimentAnalysisResponse(
            comments=comment_results,
            statistics=statistics,
            chart_image=chart_image
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)