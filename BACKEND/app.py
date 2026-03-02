from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import PyPDF2
import re
import random
import os
import requests
import json
import anthropic
from dotenv import load_dotenv

# Load .env explicitly from the BACKEND directory so keys are found regardless of cwd
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, '.env'))

# Debug: print presence of keys (masked) to help diagnose 401s — does not reveal full secrets
def _mask_key(k):
    if not k:
        return None
    s = k.strip()
    if len(s) <= 12:
        return s[:3] + '...' + s[-3:]
    return s[:8] + '...' + s[-4:]

_openai_k = os.getenv('OPENAI_API_KEY')
_anthropic_k = os.getenv('ANTHROPIC_API_KEY')
print(f"DEBUG: OPENAI_API_KEY present={bool(_openai_k)}, value={_mask_key(_openai_k)}")
print(f"DEBUG: ANTHROPIC_API_KEY present={bool(_anthropic_k)}, value={_mask_key(_anthropic_k)}")

app = Flask(__name__)
CORS(app)

# Get the parent directory (AI-Study-assistant) to access FRONTEND folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'FRONTEND')

# 1️⃣ Root Route - Serve Homepage
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, 'homepage.html')

# 2️⃣ Serve static files from FRONTEND
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# 3️⃣ PDF Extraction Function
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# 4️⃣ Quiz Generation
@app.route("/quiz", methods=["POST"])
def generate_quiz():
    file = request.files.get("pdf")

    if not file:
        return jsonify({"error": "No file uploaded"})

    text = extract_text_from_pdf(file)

    if not text.strip():
        return jsonify({"error": "Could not extract text"})

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_URL = os.getenv("GROQ_API_URL")

    prompt_text = (
        "Create a JSON object with a top-level key 'questions' containing a list of 10 multiple-choice questions "
        "based on the following text. Each question must be an object with keys: 'question' (string), 'options' (array of 4 strings), "
        "and 'answer' (the exact string that is the correct option). Return valid JSON only, with no explanatory text.\n\nText:\n" +
        (text[:30000])
    )

    if GROQ_API_KEY and GROQ_API_URL:
        try:
            headers = {
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json'
            }
            payload = {
                'prompt': prompt_text,
                'max_tokens': 2000
            }
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()

            resp_json = None
            try:
                resp_json = resp.json()
            except Exception:
                resp_text = resp.text
                try:
                    resp_json = json.loads(resp_text)
                except Exception:
                    resp_json = {"text": resp_text}

            groq_text = None
            if isinstance(resp_json, dict):
                groq_text = resp_json.get('text') or resp_json.get('result') or resp_json.get('output')
                if not groq_text and 'outputs' in resp_json and isinstance(resp_json['outputs'], list) and len(resp_json['outputs']) > 0:
                    first = resp_json['outputs'][0]
                    if isinstance(first, dict):
                        groq_text = first.get('content') or first.get('text')
                    elif isinstance(first, str):
                        groq_text = first

            if groq_text:
                try:
                    parsed = json.loads(groq_text)
                    return jsonify(parsed)
                except Exception:
                    m = re.search(r"\{\s*\"questions\"[\s\S]*\}", groq_text)
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                            return jsonify(parsed)
                        except Exception:
                            pass
        except Exception as e:
            pass

    sentences = text.split('.')
    key_concepts = [sentence.strip() for sentence in sentences if len(sentence.split()) > 5]

    if len(key_concepts) < 10:
        return jsonify({"error": "Not enough content to generate quiz"})

    questions = []
    for concept in key_concepts[:10]:
        question_text = f"What is the significance of: {concept}?"
        options = [
            concept,
            "An unrelated concept",
            "A partially related concept",
            "None of the above"
        ]
        random.shuffle(options)
        questions.append({
            "question": question_text,
            "options": options,
            "answer": concept
        })

    return jsonify({"questions": questions})

# Helper: OpenAI chat completion fallback when Anthropic key isn't configured
def openai_chat_completion(prompt, max_tokens=1024, model="gpt-3.5-turbo"):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    # best-effort extraction
    if "choices" in j and isinstance(j["choices"], list) and len(j["choices"]) > 0:
        content = j["choices"][0].get("message", {}).get("content")
        if content:
            return content
    # fallback to other fields
    return j.get("text") or json.dumps(j)

# New helper: Hugging Face Inference API text generation
def hf_text_generation(prompt, model="google/flan-t5-large", max_tokens=1024, temperature=0.2):
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    if not HUGGINGFACE_API_KEY:
        raise RuntimeError("Hugging Face API key not configured")

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Accept": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    j = resp.json()

    # Hugging Face may return list of generations or an error dict
    if isinstance(j, dict) and j.get("error"):
        raise RuntimeError(j.get("error"))
    if isinstance(j, list) and len(j) > 0:
        first = j[0]
        if isinstance(first, dict):
            return first.get("generated_text") or json.dumps(first)
        return str(first)

    return json.dumps(j)

# New helper: very small local fallback summarizer (extractive) and elaborator (frequency-based)
def local_summarize(text, max_sentences=5):
    # Split into sentences naively
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return {"summary": ""}

    # Rank sentences by length (simple heuristic) and pick top unique ones
    ranked = sorted(sentences, key=lambda s: len(s), reverse=True)
    chosen = []
    seen = set()
    for s in ranked:
        key = s[:120]
        if key not in seen:
            chosen.append(s)
            seen.add(key)
        if len(chosen) >= max_sentences:
            break

    summary = ' '.join(chosen[:max_sentences])
    return {"summary": summary}

STOPWORDS = set(["the","and","is","in","of","to","a","for","with","on","by","an","be","that","this","as","are"])

def local_elaborate(text, max_topics=6):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1

    # pick top words as proxy topics
    topics = [w for w, _ in sorted(freq.items(), key=lambda it: it[1], reverse=True)][:max_topics]
    elaborations = []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for t in topics:
        related = [s.strip() for s in sentences if t in s.lower()]
        if related:
            elab = ' '.join(related[:3])
        else:
            elab = f"The document discusses '{t}'. Key points related to this topic appear throughout the text; review the document for context and examples." 
        elaborations.append({"topic": t, "elaboration": elab})

    return {"elaborations": elaborations}

# 5️⃣ Elaborate Topics using Claude AI (or OpenAI fallback)
@app.route("/elaborate", methods=["POST"])
def elaborate_topics():
    file = request.files.get("pdf")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    text = extract_text_from_pdf(file)
    if not text.strip():
        return jsonify({"error": "Could not extract text"}), 400

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    # Prefer Anthropic if configured
    if ANTHROPIC_API_KEY:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        prompt = (
            "Read the following document text carefully. Identify ALL important topics and concepts covered. "
            "For each topic, write a thorough explanation suitable for a student studying this subject.\n\n"
            "Return ONLY a valid JSON object with this exact format, no markdown, no extra text:\n"
            "{\"elaborations\": [{\"topic\": \"Topic Name\", \"elaboration\": \"Detailed explanation here...\"}, ...]}\n\n"
            "Document text:\n" + text[:30000]
        )

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    return jsonify({"error": "Could not parse AI response"}), 500
            else:
                return jsonify({"error": "Could not parse AI response"}), 500

        return jsonify(parsed)

    # If Anthropic not configured, try OpenAI fallback
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        prompt = (
            "Read the following document text carefully. Identify ALL important topics and concepts covered. "
            "For each topic, write a thorough explanation suitable for a student studying this subject.\n\n"
            "Return ONLY a valid JSON object with this exact format, no markdown, no extra text:\n"
            "{\"elaborations\": [{\"topic\": \"Topic Name\", \"elaboration\": \"Detailed explanation here...\"}, ...]}\n\n"
            "Document text:\n" + text[:30000]
        )

        try:
            resp_text = openai_chat_completion(prompt, max_tokens=2000, model="gpt-3.5-turbo")
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        raw = resp_text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    return jsonify({"error": "Could not parse AI response"}), 500
            else:
                return jsonify({"error": "Could not parse AI response"}), 500

        return jsonify(parsed)

    # If OpenAI not configured, try Hugging Face fallback
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    if HUGGINGFACE_API_KEY:
        prompt = (
            "Read the following document text carefully. Identify ALL important topics and concepts covered. "
            "For each topic, write a thorough explanation suitable for a student studying this subject.\n\n"
            "Return ONLY a valid JSON object with this exact format, no markdown, no extra text:\n"
            "{\"elaborations\": [{\"topic\": \"Topic Name\", \"elaboration\": \"Detailed explanation here...\"}, ...]}\n\n"
            "Document text:\n" + text[:30000]
        )

        try:
            resp_text = hf_text_generation(prompt, model="google/flan-t5-large", max_tokens=2000, temperature=0.2)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        raw = resp_text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    return jsonify({"error": "Could not parse AI response"}), 500
            else:
                return jsonify({"error": "Could not parse AI response"}), 500

        return jsonify(parsed)

    # If Hugging Face not configured, use local fallback
    return jsonify(local_elaborate(text))

# 6️⃣ Summarize PDF using Claude AI (or OpenAI fallback)
@app.route("/summarize", methods=["POST"])
def summarize_pdf():
    file = request.files.get("pdf")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    text = extract_text_from_pdf(file)
    if not text.strip():
        return jsonify({"error": "Could not extract text"}), 400

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if ANTHROPIC_API_KEY:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    "Please summarize the following document clearly and concisely. "
                    "Cover all the main topics and key points in simple language suitable for a student.\n\n"
                    "Document:\n" + text[:30000]
                )
            }]
        )

        summary = message.content[0].text.strip()
        return jsonify({"summary": summary})

    # OpenAI fallback
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        prompt = (
            "Please summarize the following document clearly and concisely. "
            "Cover all the main topics and key points in simple language suitable for a student.\n\n"
            "Document:\n" + text[:30000]
        )

        try:
            resp_text = openai_chat_completion(prompt, max_tokens=1024, model="gpt-3.5-turbo")
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        summary = resp_text.strip()
        return jsonify({"summary": summary})

    # Hugging Face fallback
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    if HUGGINGFACE_API_KEY:
        prompt = (
            "Please summarize the following document clearly and concisely. "
            "Cover all the main topics and key points in simple language suitable for a student.\n\n"
            "Document:\n" + text[:30000]
        )

        try:
            resp_text = hf_text_generation(prompt, model="google/flan-t5-large", max_tokens=1024, temperature=0.2)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        summary = resp_text.strip()
        return jsonify({"summary": summary})

    # Local fallback
    return jsonify(local_summarize(text))

# 7️⃣ Analyze Student Performance and Generate Feedback
@app.route("/analyze", methods=["POST"])
def analyze_performance():
    data = request.get_json()
    student_answers = data.get("student_answers")
    correct_answers = data.get("correct_answers")
    concepts = data.get("concepts")

    if not student_answers or not correct_answers or not concepts:
        return jsonify({"error": "Missing required data"})

    total_questions = len(correct_answers)
    score = sum(1 for i in range(total_questions) if student_answers[i] == correct_answers[i])
    performance_level = (
        "Beginner" if score / total_questions < 0.4 else
        "Intermediate" if score / total_questions <= 0.75 else
        "Advanced"
    )

    mistake_analysis = []
    weak_areas = []
    strong_areas = []

    for i, (student, correct) in enumerate(zip(student_answers, correct_answers)):
        if student != correct:
            mistake_analysis.append({
                "question_index": i + 1,
                "concept": concepts[i],
                "mistake_type": "Conceptual misunderstanding" if student in concepts else "Careless error"
            })
            weak_areas.append(concepts[i])
        else:
            strong_areas.append(concepts[i])

    improvement_plan = [
        "Revise the following concepts: " + ", ".join(set(weak_areas)),
        "Practice more questions on weak areas.",
        "Attempt quizzes of similar difficulty to strengthen understanding."
    ]

    recommended_next_level = (
        "easy" if score / total_questions < 0.4 else
        "medium" if score / total_questions <= 0.75 else
        "hard"
    )

    return jsonify({
        "score": f"{score}/{total_questions}",
        "performance_level": performance_level,
        "strong_areas": list(set(strong_areas)),
        "weak_areas": list(set(weak_areas)),
        "mistake_analysis": mistake_analysis,
        "recommended_next_level": recommended_next_level,
        "improvement_plan": improvement_plan
    })

# 8️⃣ Keep this at bottom
if __name__ == "__main__":
    app.run(debug=True)