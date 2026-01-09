import requests
import json
import time
import google.generativeai as genai

# ============ CONFIGURATION ============
GEMINI_API_KEY = "AIzaSyB7WqxNXvOr8IGxd_RnsaWjyl_3vt6hSw0"  # Yaha apni key daal
YOUR_TRAINING_API = "https://peru-cat-625876.hostingersite.com/api/training/"

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)

# ============ STEP 1: Fetch Training Data from Your API ============
print("Fetching training data from your API...")
response = requests.get(YOUR_TRAINING_API)

if response.status_code != 200:
    print("Error fetching data:", response.text)
    exit()

data = response.json()
print(f"Fetched {len(data)} chat logs")

# Expected format: [{"question": "Rohu ko kitna feed dena chahiye?", "answer": "Din mein 2-3% body weight..."}]
# Agar format alag hai to adjust kar lena

# ============ STEP 2: Prepare Data for Gemini Fine-Tuning ============
training_examples = []
for item in data:
    if 'question' in item and 'answer' in item:
        training_examples.append({
            "messages": [
                {"role": "user", "content": item['question']},
                {"role": "model", "content": item['answer']}
            ]
        })

if len(training_examples) < 100:
    print("Warning: Kam data hai (<100). Fine-tuning ke liye 100+ examples best hain.")
else:
    print(f"Prepared {len(training_examples)} examples for fine-tuning")

# Save locally (optional backup)
with open('training_data.jsonl', 'w', encoding='utf-8') as f:
    for ex in training_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
print("Training data saved as training_data.jsonl")

# ============ STEP 3: Upload to Gemini & Start Fine-Tuning ============
print("Uploading data to Gemini...")

# Upload file
uploaded_file = genai.upload_file(path="training_data.jsonl", display_name="MatsyaSaarthi_Training_Data")
print(f"Uploaded file ID: {uploaded_file.name}")

# Create tuned model
print("Starting fine-tuning... (yeh 30 minutes to 4 hours lag sakta hai)")
operation = genai.create_tuned_model(
    source_model="gemini-1.5-flash",
    training_data=uploaded_file,
    id="matsyasaarthi-custom-v1",  # Tera model name
    display_name="Matsya Saarthi Custom Model",
    epoch_count=4,
    batch_size=4,
    learning_rate=0.001,
)

# Wait for completion
for status in operation.wait_bar():
    time.sleep(30)  # Thoda delay

print("Fine-tuning completed!")

tuned_model = operation.result
print(f"Tuned model name: {tuned_model.name}")

# ============ STEP 4: Use Your Custom Model in App ============
print("\nAb tere app mein yeh code use kar:")
print(f"""
model = genai.GenerativeModel('{tuned_model.name}')
response = model.generate_content("Rohu fish ko disease hai kya karu?")
print(response.text)
""")

print("\nYa PHP mein curl se:")
print(f"""
curl "https://generativelanguage.googleapis.com/v1beta/models/{tuned_model.name}:generateContent?key=$GEMINI_API_KEY" \\
-H "Content-Type: application/json" \\
-d '{{"contents":[{{"role":"user","parts":[{{"text":"pond mein oxygen kam hai kya karu"}}]}}]}}'
""")
