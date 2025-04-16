import pickle
import io
import torch
import re
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# Custom unpickler class to handle tensors saved on CUDA
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# Function to preprocess text (same as in the training notebook)
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
def predict_sentiment(text, loaded_model, loaded_tokenizer):
    # Preprocess and tokenize
    preprocessed_text = preprocess_text(text)
    inputs = loaded_tokenizer(
        preprocessed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Make prediction
    with torch.no_grad():  # This reduces memory usage
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        prob = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = prob[0][prediction].item()
    
    return sentiment, confidence, prediction

# Main script
def main():
    print("Starting YouTube comment sentiment analysis...")
    
    # Load the model - only load once for efficiency
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
        print(f"Error: Model file '{model_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Setup Chrome options for selenium
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")
    
    try:
        driver = webdriver.Chrome(service=Service("chromedriver.exe"), options=options)
    except Exception as e:
        print(f"Error initializing Chrome driver: {str(e)}")
        print("Make sure chromedriver.exe is in the correct location and matches your Chrome version.")
        return
    
    # Get YouTube URL
    video_url = input("Enter YouTube URL: ").strip()
    
    # Scrape comments
    try:
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
        
        driver.quit()
        
        if not comments:
            print("No comments found. Try a different video or check the XPath selector.")
            return
            
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
            print(f"Comment {i}/{len(comments)} analyzed")
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Calculate statistics
        positive_count = sum(df['prediction'])
        negative_count = len(df) - positive_count
        positive_percentage = (positive_count / len(df)) * 100
        negative_percentage = (negative_count / len(df)) * 100
        
        # Display results
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS RESULTS")
        print("="*60)
        print(f"Total comments analyzed: {len(comments)}")
        print(f"Positive comments: {positive_count} ({positive_percentage:.1f}%)")
        print(f"Negative comments: {negative_count} ({negative_percentage:.1f}%)")
        print("="*60)
        
        # Display individual comment analysis
        print("\nDETAILED COMMENT ANALYSIS:")
        print("-"*60)
        for i, row in df.iterrows():
            print(f"{row['index']}. {row['comment'][:100]}{'...' if len(row['comment']) > 100 else ''}")
            print(f"   Sentiment: {row['sentiment']} (Confidence: {row['confidence']:.2f})")
            print("-"*60)
        
        # Create visualization
        labels = ['Positive', 'Negative']
        sizes = [positive_percentage, negative_percentage]
        colors = ['#66b3ff', '#ff9999']
        explode = (0.1, 0)  # explode the 1st slice (Positive)
        
        plt.figure(figsize=(10, 6))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Sentiment Distribution in YouTube Comments')
        plt.savefig('sentiment_distribution.png')
        print("\nPie chart saved as 'sentiment_distribution.png'")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    main()