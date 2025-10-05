from huggingface_hub import InferenceClient
import PyPDF2
import numpy as np
import pickle
import os


USE_FAISS = True
try:
    import faiss  
except Exception as e:  # ModuleNotFoundError or other import issues
    USE_FAISS = False
    print("⚠️  FAISS not available:", str(e))
    print("➡️  Falling back to simple numpy storage. Install with 'pip install faiss-cpu' (CPU) or 'pip install faiss-gpu' (CUDA) for faster similarity search.")

# Hugging Face API key handling (avoid hard-coding secrets in production)
HF_API_TOKEN = os.getenv("HF_API_TOKEN") 
if not os.getenv("HF_API_TOKEN"):
    print("⚠️  Using fallback embedded Hugging Face token. Set HF_API_TOKEN env var to override.")

# Create a dedicated client bound to the embedding model so .feature_extraction(text) uses correct signature
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
hf_client = InferenceClient(model=EMBED_MODEL, token=HF_API_TOKEN)

def pdf_to_vectors(pdf_path):
    # Read PDF
    print(f"📄 Reading PDF: {pdf_path}")
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)

        # Extract text from each page separately
        page_texts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            page_texts.append({
                'text': page_text,
                'page_number': page_num + 1
            })

        # Combine all text for chunking
        text = ''.join([p['text'] for p in page_texts])

    print(f"📊 Total pages: {total_pages}")
    print(f"📊 Total text length: {len(text):,} characters")
    print(f"📊 Average characters per page: {len(text) // total_pages:,}")

    # Create chunks with page tracking
    chunks = []
    chunk_metadata = []

    # If there's no extractable text, bail out gracefully
    if len(text.strip()) == 0:
        print("❌ No extractable text found in this PDF (possibly scanned images or empty file).")
        print("➡️  Skipping embedding generation.")
        # Still write minimal metadata so downstream code can detect the state
        with open("chunks.pkl", "wb") as f:
            pickle.dump({
                'chunks': [],
                'metadata': [],
                'total_pages': total_pages,
                'backend': 'none'
            }, f)
        print("📁 Wrote empty metadata file: chunks.pkl")
        return np.empty((0, )), []

    for i in range(0, len(text), 400):
        chunk_text = text[i:i + 500]
        chunks.append(chunk_text)

        # Estimate which page this chunk belongs to
        estimated_page = min((i // (len(text) // total_pages)) + 1, total_pages)
        chunk_metadata.append({
            'start_pos': i,
            'estimated_page': estimated_page
        })

    print(f"✂️  Created {len(chunks)} chunks")

    # Get embeddings from Hugging Face
    print("🔄 Getting embeddings from Hugging Face...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Processing {i + 1}/{len(chunks)}")
        # Basic retry (up to 3 attempts) in case of transient HTTP errors/rate limits
        attempt = 0
        while attempt < 3:
            try:
                result = hf_client.feature_extraction(chunk)
                # Expect list (batch) -> vector, or vector directly; normalize shape
                vec = result[0] if isinstance(result, list) and isinstance(result[0], (list, tuple)) else result
                embeddings.append(vec)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 3:
                    raise
                print(f"  ⟳ Retry {attempt}/3 after error: {e}")

    # Handle case of zero chunks (should already be caught, but double guard)
    if len(embeddings) == 0:
        print("⚠️  No chunks to embed (empty text). Nothing to index.")
        with open("chunks.pkl", "wb") as f:
            pickle.dump({
                'chunks': [],
                'metadata': chunk_metadata,
                'total_pages': total_pages,
                'backend': 'none'
            }, f)
        return np.empty((0, )), []

    embeddings = np.array(embeddings).astype('float32')

    # Normalize & build index (FAISS or fallback)
    if USE_FAISS:
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        print("🗂️  Creating FAISS index...")
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product = cosine after normalization
        index.add(embeddings)
        # Save to files
        print("💾 Saving FAISS index & metadata...")
        faiss.write_index(index, "vectors.index")
    else:
        # We'll save normalized vectors manually for a simple numpy search later
        print("🗂️  Normalizing & saving embeddings (no FAISS)...")
        # manual L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        np.save("vectors.npy", embeddings)
        print("💾 Saved embeddings to vectors.npy")

    with open("chunks.pkl", "wb") as f:
        pickle.dump({
            'chunks': chunks,
            'metadata': chunk_metadata,
            'total_pages': total_pages,
            'backend': 'faiss' if USE_FAISS else 'numpy'
        }, f)

    print("✅ Vector data created successfully!")
    if USE_FAISS:
        print("📁 Files saved: vectors.index, chunks.pkl")
    else:
        print("📁 Files saved: vectors.npy, chunks.pkl")
    print(f"📊 Vector shape: {embeddings.shape}")
    if len(embeddings):
        print(f"🔢 Sample vector (first 5 dims): {embeddings[0][:5]}")

    return embeddings, chunks


# Usage
if __name__ == "__main__":
    pdf_file = "lab.pdf"  # Change to your PDF file
    embeddings, chunks = pdf_to_vectors(pdf_file)

    print("\n🎉 Setup complete! Now you can run 'ask_questions.py' to chat with your PDF!")
