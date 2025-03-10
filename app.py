from flask import Flask, request, jsonify, g, render_template
from flask_cors import CORS
import os
import mysql.connector
import pdfplumber
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from docx import Document

# ✅ Set Full Path for Templates
TEMPLATE_DIR = r"E:\NLP\templates"  # Full path to your template directory

app = Flask(__name__, template_folder=TEMPLATE_DIR)

CORS(app)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

faiss_index = None
id_map = None

# ✅ Load Models
search_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
summary_tokenizer = T5Tokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
summary_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

# ✅ MySQL Connection
def get_db():
    if 'db' not in g:
        g.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="knowledge_base",
            pool_name="mypool",
            pool_size=5
        )
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# ✅ Load Documents
def load_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_pdf_file(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_docx_file(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_documents(folder_path):
    """Loads all TXT, PDF, DOCX files from a folder"""
    documents, filenames = [], []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.txt'):
            content = load_txt_file(file_path)
        elif filename.endswith('.pdf'):
            content = load_pdf_file(file_path)
        elif filename.endswith('.docx'):
            content = load_docx_file(file_path)
        else:
            continue
        documents.append(content)
        filenames.append(filename)
    return documents, filenames

# ✅ Summarization
def summarize_text(text, max_length=100, min_length=30):
    """
    Generate a summary for the given text.
    If the text is too short, returns a default message.
    """
    if not text or len(text.split()) < 50:  
        return "Summary not available. Text is too short."

    # 🔹 Keep only the first 500 words for summarization (prevents errors)
    truncated_text = " ".join(text.split()[:500])

    try:
        prompt = "summarize: " + truncated_text
        inputs = summary_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = summary_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    except Exception as e:
        print(f"⚠️ Summarization Error: {e}")
        return "Error generating summary."


# ✅ FAISS Index
def get_faiss_index():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, embedding FROM documents")
    docs = cursor.fetchall()

    embeddings, ids = [], []
    for doc_id, embedding_json in docs:
        embedding = np.array(json.loads(embedding_json), dtype=np.float32)
        embeddings.append(embedding)
        ids.append(doc_id)

    if not embeddings:
        return None, {}

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    
    id_map = {i: ids[i] for i in range(len(ids))}
    return index, id_map

@app.before_request
def ensure_faiss_index():
    global faiss_index, id_map
    if faiss_index is None or id_map is None:
        print("Initializing FAISS index...")
        faiss_index, id_map = get_faiss_index()

# ✅ Search Documents
def search_documents(query, documents, filenames):
    query_embedding = search_model.encode([query], convert_to_numpy=True)
    doc_embeddings = search_model.encode(documents, convert_to_numpy=True)
    similarities = np.dot(query_embedding, doc_embeddings.T)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_documents = [documents[i] for i in ranked_indices]
    ranked_filenames = [filenames[i] for i in ranked_indices]
    ranked_scores = [similarities[i] for i in ranked_indices]
    return ranked_documents, ranked_filenames, ranked_scores

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles document uploads and stores embeddings in MySQL."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract text
    if file.filename.endswith(".pdf"):
        text = load_pdf_file(file_path)
    elif file.filename.endswith(".docx"):
        text = load_docx_file(file_path)
    elif file.filename.endswith(".txt"):
        text = load_txt_file(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # Encode and store embeddings
    embedding = search_model.encode(text).tolist()
    embedding_json = json.dumps(embedding)

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO documents (filename, content, embedding) VALUES (%s, %s, %s)",
        (file.filename, text, embedding_json)
    )
    db.commit()

    return jsonify({"message": "File uploaded and processed successfully"})

@app.route("/search", methods=["POST"])
def search():
    """Search documents using semantic similarity and summarize results."""
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, filename, content FROM documents")
    file_results = cursor.fetchall()

    if not file_results:
        return jsonify({"error": "No documents available"}), 400

    filenames, documents = [], []
    doc_map = {}
    for doc_id, filename, content in file_results:
        filenames.append(filename)
        documents.append(content)
        doc_map[filename] = content

    # Encode query & docs
    query_embedding = search_model.encode([query], convert_to_numpy=True)
    doc_embeddings = search_model.encode(documents, convert_to_numpy=True)

    # Compute similarities
    similarities = np.dot(query_embedding, doc_embeddings.T)[0]
    # Cast similarities to float64 to avoid JSON issues
    similarities = similarities.astype(np.float64)

    # Sort by highest similarity
    ranked_indices = np.argsort(similarities)[::-1]
    top_files = [filenames[i] for i in ranked_indices[:5]]
    top_scores = [similarities[i] for i in ranked_indices[:5]]
    top_summaries = [summarize_text(documents[i]) for i in ranked_indices[:5]]

    # Construct JSON response
    results = []
    for i in range(len(top_files)):
        # Convert each score to a Python float & round
        score = round(float(top_scores[i]), 4)
        results.append({
            "filename": top_files[i],
            "score": score,
            "summary": top_summaries[i]
        })

    return jsonify(results)

@app.route("/documents", methods=["GET"])
def get_documents():
    """Retrieve a list of uploaded documents."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, filename FROM documents")
    rows = cursor.fetchall()
    results = [{"id": row[0], "filename": row[1]} for row in rows]
    return jsonify(results)

if __name__ == "__main__":
    from streamlit.web import cli as stcli
    import sys
    if sys.argv[0].endswith("streamlit"):
        sys.exit(stcli.main())
    else:
        app.run(host="0.0.0.0", port=5500)