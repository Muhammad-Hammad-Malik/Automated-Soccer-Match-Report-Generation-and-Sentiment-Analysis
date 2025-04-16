import pandas as pd
from openai import OpenAI

# Configure the OpenRouter API client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Function to generate match report
def generate_match_report(commentary):
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://example.com",
            "X-Title": "Football Analysis Tool",
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
    return completion.choices[0].message.content

# Main process
def process_sequentially(input_csv, output_csv, start_index=0):
    try:
        df = pd.read_csv(input_csv)

        if 'Commentary' not in df.columns:
            raise ValueError("CSV must contain a 'Commentary' column")

        if 'MatchReport' not in df.columns:
            df['MatchReport'] = ""

        total = len(df)

        for idx in range(start_index, total):
            commentary = df.at[idx, 'Commentary']
            print(f"\n▶ Processing row #{idx} — {len(commentary.split())} words")

            try:
                report = generate_match_report(commentary)
                df.at[idx, 'MatchReport'] = report
                df.to_csv(output_csv, index=False)

                remaining = total - (idx + 1)
                print(f"✅ Done. {remaining} entries remaining.")

            except Exception as e:
                print(f"\n❌ Error at index {idx}: {e}")
                print("Halting process. You can resume from this index.")
                break

        print(f"\n🎉 Finished. Output saved to {output_csv}")

    except Exception as e:
        print(f"\n🔥 Fatal error: {e}")

# Run script
if __name__ == "__main__":
    input_csv = "Commentaries.csv"
    output_csv = "Commentaries_with_Reports.csv"
    process_sequentially(input_csv, output_csv, start_index=43)  # Change start_index as needed
