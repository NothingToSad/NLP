import os
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

# --- ✅ โหลดโมเดลสำหรับค้นหาและสรุป ---
search_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # สำหรับค้นหา
summary_tokenizer = T5Tokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")  # สำหรับสรุป
summary_model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

# --- 📥 ฟังก์ชันโหลดไฟล์เอกสาร ---
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

# --- 📂 โหลดเอกสารทั้งหมดจากโฟลเดอร์ ---
def load_documents(folder_path):
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

# --- 📝 ฟังก์ชันสรุปข้อความ ---
def summarize_text(text, max_length=100, min_length=30):
    prompt = "summarize: " + text
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

# --- 🔍 ฟังก์ชันค้นหาเอกสารที่เกี่ยวข้อง ---
def search_documents(query, documents, filenames):
    query_embedding = search_model.encode([query], convert_to_numpy=True)
    doc_embeddings = search_model.encode(documents, convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_documents = [documents[i] for i in ranked_indices]
    ranked_filenames = [filenames[i] for i in ranked_indices]
    ranked_scores = [similarities[i] for i in ranked_indices]
    return ranked_documents, ranked_filenames, ranked_scores

# --- 🚀 เริ่มทำงาน ---
folder_path = r'C:\Users\Yok\Desktop\nlp\file_test'  # 🔔 แก้เป็น path ของคุณ
documents, filenames = load_documents(folder_path)

query = "robot"  # 🔎 คำค้นหา
ranked_documents, ranked_filenames, ranked_scores = search_documents(query, documents, filenames)

# --- 📢 แสดงผลลัพธ์พร้อมสรุป ---
print("\n🔎 Top 5 เอกสารที่เกี่ยวข้อง:")
for i in range(min(5, len(ranked_filenames))):
    print(f"\n{i+1}. 📄 ไฟล์: {ranked_filenames[i]}")
    print(f"✅ ความคล้ายคลึง: {ranked_scores[i]:.2f}")
    print(f"📚 เนื้อหาบางส่วน: {ranked_documents[i][:300]}...")  # แสดง 300 ตัวอักษรแรก

    summary = summarize_text(ranked_documents[i])  # 📝 สรุปเนื้อหา
    print(f"📝 สรุปเนื้อหา: {summary}")
