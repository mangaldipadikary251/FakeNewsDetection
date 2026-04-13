import ollama
import pandas as pd
import os

# 1. Path to your current master dataset
master_path = r"C:\dscode\project_6th_sem\master_dataset.csv"

# 2. Define topics for fake news generation
topics = [
    "5G towers affecting local weather", 
    "Secret underground tunnel in New York", 
    "AI replacing all government officials by 2027", 
    "Dangerous chemicals found in common garden plants"
]

print("Generating 40 synthetic fake news snippets... please wait.")
synthetic_rows = []

for topic in topics:
    # Generate 10 paragraphs per topic
    for _ in range(10):
        try:
            response = ollama.chat(model='llama3', messages=[
                {'role': 'system', 'content': 'You are a writer of convincing but sensationalist fake news snippets. Keep them short, about 2-3 sentences.'},
                {'role': 'user', 'content': f'Write a fake news paragraph about {topic}.'}
            ])
            text = response['message']['content']
            synthetic_rows.append({'text': text, 'label': 0}) # 0 = Fake
        except Exception as e:
            print(f"Error generating for {topic}: {e}")

# 3. Load, combine, and save
if os.path.exists(master_path):
    df_existing = pd.read_csv(master_path)
    df_synthetic = pd.DataFrame(synthetic_rows)
    df_final = pd.concat([df_existing, df_synthetic], ignore_index=True)
    df_final.to_csv(master_path, index=False)
    print(f"✅ Success! Added {len(df_synthetic)} synthetic rows. Total rows now: {len(df_final)}")
else:
    print("❌ Error: Could not find master_dataset.csv. Check your path!")