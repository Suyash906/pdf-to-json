from flask import Flask, request, jsonify
import fitz  # PyMuPDF
from openai import OpenAI
import os
import tempfile
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Set your OpenAI API key

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def summarize_pdf_with_openai(text, pdf_filename):
    prompt = f"""
You are a legal AI assistant. Summarize the case details from the following legal judgment in this JSON format:
{{
    "pdf_file": "{pdf_filename}",
    "response": {{
        "lowerCourtName": "...",
        "currentCourtName": "...",
        "partyA": "...",
        "partyB": "...",
        "factualBackground": "...",
        "legalIssues": ["..."],
        "arguments": ["Party A Argument: ...", "Party B Argument: ..."],
        "decisions": ["..."],
        "caseLawCited": [],
        "lowerCourtFavour": "...",
        "currentCourtFavour": "...",
        "nextPlaceOfAppeal": "...",
        "precedentSearchTerms": ["..."]
    }}
}}

Text:
{text}

Only return valid JSON.
"""
    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2)

    return response.choices[0].message.content

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            file.save(tmp.name)
            text = extract_text_from_pdf(tmp.name)
            summary = summarize_pdf_with_openai(text, file.filename)
            os.unlink(tmp.name)  # Clean up
            return jsonify(json.loads(summary))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
