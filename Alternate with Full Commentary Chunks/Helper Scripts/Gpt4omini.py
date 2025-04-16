import pandas as pd
import numpy as np
from openai import OpenAI
import os

# Configure the OpenRouter API client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # Replace with your actual API key
)

# Function to read CSV and select a random commentary
def get_random_commentary(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure 'Commentary' column exists
    if 'Commentary' not in df.columns:
        raise ValueError("CSV file must contain a 'Commentary' column")
    
    # Select a random row
    random_index = np.random.randint(0, len(df))
    random_row = df.iloc[random_index]
    
    # Get the commentary text
    commentary = random_row['Commentary']
    
    # Return both the commentary and the row index (for reference)
    return commentary, random_index

# Function to generate match report
def generate_match_report(commentary):
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://yourdomain.com",  # Replace with your site URL
                "X-Title": "Football Analysis Tool",  # Replace with your site name
            },
            model="openai/gpt-4o-mini-search-preview",
            messages=[
                {"role": "system", "content": """You are a sports analyst creating concise football match reports.
                Focus only on the key details: final score, goal scorers, red cards, potential suspensions, and overall performance."""},
                {"role": "user", "content": f"""
                Create a concise match report (about 100 words) from this football commentary.
                
                Include:
                - Who won and the score
                - Who scored the goals
                - Any red cards or players who might miss the next match due to suspension
                - Which team played better overall
                
                Commentary:
                {commentary}
                """}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

# Main execution
def main():
    # Path to your CSV file
    csv_path = "Commentaries.csv"  # Replace with your actual CSV path
    
    try:
        commentary, index = get_random_commentary(csv_path)
        print(f"\nSelected commentary #{index} with {len(commentary.split())} words")
        
        print("\nGenerating match report...")
        report = generate_match_report(commentary)
        
        if report:
            print("\n" + "="*50)
            print("MATCH REPORT:\n")
            print(report)
            print("="*50)
            
            # Save the report
            with open(f"match_report_{index}.txt", "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReport saved as match_report_{index}.txt")
        else:
            print("Failed to generate report")
            
    except Exception as e:
        print(f"Error: {e}")

# Function to process all commentaries
def process_all_commentaries(csv_path, batch_size=10, sleep_time=2):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure 'Commentary' column exists
    if 'Commentary' not in df.columns:
        raise ValueError("CSV file must contain a 'Commentary' column")
    
    total_entries = len(df)
    print(f"Processing {total_entries} commentaries in batches of {batch_size}")
    
    # Process in batches
    for start_idx in range(0, total_entries, batch_size):
        end_idx = min(start_idx + batch_size, total_entries)
        print(f"\nProcessing batch {start_idx//batch_size + 1}: entries {start_idx} to {end_idx-1}")
        
        for i in range(start_idx, end_idx):
            try:
                commentary = df.iloc[i]['Commentary']
                print(f"Processing commentary #{i} ({len(commentary.split())} words)")
                
                report = generate_match_report(commentary)
                
                # Save the report
                with open(f"match_report_{i}.txt", "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"Report saved as match_report_{i}.txt")
                
                # Brief pause between API calls
                if i < end_idx - 1:
                    import time
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Error processing commentary #{i}: {str(e)}")
        
        # Longer pause between batches
        if end_idx < total_entries:
            print(f"Batch completed. Pausing before next batch...")
            import time
            time.sleep(sleep_time * 5)
    
    print("\nAll commentaries processed!")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Process a single random commentary")
    print("2. Process all commentaries in batches")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        main()
    elif choice == "2":
        csv_path = "Commentaries.csv"  # Replace with your actual CSV path
        batch_size = int(input("Enter batch size (default 10): ") or "10")
        sleep_time = int(input("Enter seconds to wait between requests (default 2): ") or "2")
        process_all_commentaries(csv_path, batch_size, sleep_time)
    else:
        print("Invalid choice. Exiting.")