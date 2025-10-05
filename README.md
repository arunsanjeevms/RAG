# <center> 📚 RAG-LLM

## 🧠 Why I Built This

While preparing for my exams, I constantly ran into the same problem — I’d search online for answers, but most of what I found didn’t match my college syllabus. It wasted time, especially when I just needed explanations based on my *official materials*.  

So, I decided to build my own **RAG (Retrieval-Augmented Generation)** system — one that could read my **college’s official PDFs** and give accurate, syllabus-specific answers. It also integrates **web search** to expand beyond the textbook when needed.  

The goal was simple: a personal AI study assistant that helps me revise smarter and faster — with answers sourced directly from my study materials, plus **10+ relevant Q&A suggestions** for deeper understanding.

---

## ⚙️ What It Does

- 📂 Upload your college PDF (textbooks, notes, syllabus, etc.)  
- 🔍 Ask any question from your syllabus  
- 📄 Retrieves answers directly from your uploaded PDF  
- 🌐 Expands with web search using RAG pipeline  
- 🧩 Suggests 10+ related questions and answers automatically  
- 💡 Uses Hugging Face Embeddings and FAISS Vector Indexing  

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Python** | Core language |
| **FAISS** | Vector storage and similarity search |
| **Hugging Face Transformers** | For text embeddings |
| **PyPDF2** | For reading and chunking PDF content |
| **Hugging Face Inference API** | For model inference (no local GPU needed) |
| **NumPy & Pickle** | For efficient vector and metadata handling |

---

## 🧰How It Works

1. **Extract PDF Text:**  
   Reads all pages using PyPDF2 and splits them into chunks (~400–800 characters).  

2. **Generate Embeddings:**  
   Uses `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face API to embed chunks.  

3. **Store in Vector DB (FAISS):**  
   Saves embeddings and metadata locally for quick retrieval.  

4. **Ask Questions:**  
   When a question is asked, the system retrieves similar chunks from the PDF and optionally queries the web.  

5. **Generate Contextual Answers:**  
   Combines retrieved knowledge and LLM reasoning to return a well-structured answer with related Q&A.  


