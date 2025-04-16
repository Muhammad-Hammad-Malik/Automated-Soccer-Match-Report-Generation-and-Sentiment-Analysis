import pandas as pd
import random
from openai import OpenAI

# Configure the OpenRouter API client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # Replace with your actual key
)

# Function to generate match report using the deepseekr1 model
def generate_match_report(commentary):
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://example.com",  # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "Football Analysis Tool",  # Optional. Site title for rankings on openrouter.ai.
            },
            model="nvidia/llama-3.3-nemotron-super-49b-v1:free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a football match analyst. Generate a clean, single-paragraph summary "
                        "of the match based only on the key events mentioned in the commentary. "
                        "Include goals, red cards, standout performances, and disciplinary notes "
                        "only if relevant. Do not include scores unless mentioned in the commentary. "
                        "Do not add any introductions like 'Here is a report'. Just return the paragraph."
                    )
                },
                {
                    "role": "user",
                    "content": f"Generate a concise (about 100 words) match report from the following commentary:\n\n{commentary}"
                }
            ]
        )

        # Print the full response to debug
        print("Full API Response:", completion)

        # Access and return the generated content
        return completion.choices[0].message.content if completion.choices else "No valid choices returned"

    except Exception as e:
        print(f"\n❌ Error during API call: {e}")
        return None

# Test with a random commentary
def test_random_commentary(input_csv):
    try:
        df = pd.read_csv(input_csv)

        if 'Commentary' not in df.columns:
            raise ValueError("CSV must contain a 'Commentary' column")

        # Select a random index
        random_idx = random.randint(0, len(df) - 1)
        commentary = df.at[random_idx, 'Commentary']
        
        print(f"\n▶ Testing with random commentary at index #{random_idx}:")
        print(f"Commentary: {commentary}\n")

        # Generate and print the match report
        try:
            report = generate_match_report(commentary)
            if report:
                print(f"--- Match Report ---\n{report}")
            else:
                print("No report generated due to an error.")
        except Exception as e:
            print(f"\n❌ Error generating report: {e}")

    except Exception as e:
        print(f"Error: {e}")

# Run test
if __name__ == "__main__":
    input_csv = "Commentaries.csv"
    test_random_commentary(input_csv)
