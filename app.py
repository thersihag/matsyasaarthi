from flask import Flask, request, jsonify
import google.generativeai as genai
import os

app = Flask(__name__)

# Gemini API Key from environment (Render pe set kar dena)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")

genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 1.5 Flash (fast + free tier mein best)
# Future mein fine-tuned model daal dena yaha
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    system_instruction="""
    You are Matsya Saarthi, a helpful and friendly AI assistant for Indian fish farmers.
    Answer in simple Hindi or Hinglish.
    Focus on: fish diseases, feeding schedule, pond management, oxygen level, water quality, subsidies, biofloc, RAS, Rohu, Catla, Tilapia, Pangasius etc.
    Be practical, short, and give actionable tips.
    """
)

@app.route('/')
def home():
    return jsonify({
        "message": "Matsya Saarthi Chatbot API Live! 🐟",
        "endpoints": {
            "POST /ask": "Send {'question': 'your message'} to get reply"
        }
    })

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing "question" in request'}), 400

        user_question = data['question'].strip()
        if not user_question:
            return jsonify({'error': 'Question cannot be empty'}), 400

        # Generate response
        response = model.generate_content(user_question)
        answer = response.text.strip()

        return jsonify({
            'question': user_question,
            'answer': answer
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For Render.com - important port binding
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
