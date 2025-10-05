import os
import pickle
import numpy as np
import re
from huggingface_hub import InferenceClient

# Try to import faiss if available
USE_FAISS = True
try:
    import faiss  # type: ignore
except Exception:
    USE_FAISS = False

# Shared token (reuse same env var as pdf-vector script)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not os.getenv("HF_API_TOKEN"):
    print("⚠️  Using embedded Hugging Face token (set HF_API_TOKEN to override).")

# Embedding model must match the one used in pdf-vector.py
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedding_client = InferenceClient(model=EMBED_MODEL, token=HF_API_TOKEN)

###############################################
# Generation model selection logic
###############################################

# User can supply a comma-separated list via env var; otherwise we try a curated list.
_user_models_env = os.getenv("RAG_GEN_MODELS") or os.getenv("RAG_GEN_MODEL")
if _user_models_env:
    CANDIDATE_GEN_MODELS = [m.strip() for m in _user_models_env.split(',') if m.strip()]
else:
    CANDIDATE_GEN_MODELS = [
        # Lightweight / commonly accessible instruct chat or text-gen models first
        "HuggingFaceH4/zephyr-7b-beta",
        "google/gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
        "tiiuae/falcon-7b-instruct",
        # The original one (may require conversational pipeline)
        "mistralai/Mistral-7B-Instruct-v0.2",
        # Very small fallback (not great quality, but almost always available)
        "distilbert/distilgpt2"
    ]

SELECTED_GEN_MODEL = None
GENERATION_CLIENT = None

def _try_text_gen(model: str) -> bool:
    try:
        client = InferenceClient(model=model, token=HF_API_TOKEN)
        _ = client.text_generation("Hello", max_new_tokens=5)
        return True
    except Exception as e:
        msg = str(e).lower()
        # If task unsupported we will try chat later
        if "not supported for task" in msg:
            return False
        # Some models only allow chat; treat as not text-gen capable
        return False

def _try_chat(model: str) -> bool:
    try:
        client = InferenceClient(model=model, token=HF_API_TOKEN)
        # Some hub models expose chat_completion
        _ = client.chat_completion(messages=[{"role":"user","content":"Hello"}], max_tokens=5)
        return True
    except Exception:
        return False

def select_generation_model():
    global SELECTED_GEN_MODEL, GENERATION_CLIENT
    for m in CANDIDATE_GEN_MODELS:
        if _try_text_gen(m):
            SELECTED_GEN_MODEL = m
            GENERATION_CLIENT = InferenceClient(model=m, token=HF_API_TOKEN)
            print(f"✅ Using text-generation model: {m}")
            return
    # Try chat-based
    for m in CANDIDATE_GEN_MODELS:
        if _try_chat(m):
            SELECTED_GEN_MODEL = m
            GENERATION_CLIENT = InferenceClient(model=m, token=HF_API_TOKEN)
            print(f"✅ Using chat model: {m}")
            return
    print("⚠️  No usable generation model found. Will fall back to extractive answers.")

select_generation_model()

TOP_K = 3  # number of chunks to retrieve
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT", "8000"))

def embed_text(text: str) -> np.ndarray:
    """Return L2-normalized embedding for a single text string."""
    result = embedding_client.feature_extraction(text)
    vec = result[0] if isinstance(result, list) and isinstance(result[0], (list, tuple)) else result
    arr = np.asarray(vec, dtype="float32")
    # normalize
    norm = np.linalg.norm(arr) + 1e-12
    return arr / norm

def _extractive_fallback(context: str, question: str) -> str:
    """Heuristic extractive answer if no model is available: rank sentences by overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', context)
    q_terms = {t.lower() for t in re.findall(r'\w+', question) if len(t) > 2}
    scored = []
    for s in sentences:
        s_terms = {t.lower() for t in re.findall(r'\w+', s)}
        overlap = len(q_terms & s_terms)
        if overlap:
            scored.append((overlap, s))
    if not scored:
        return "I don't have enough information in the retrieved context to answer that."
    scored.sort(reverse=True, key=lambda x: x[0])
    best = [s for _, s in scored[:3]]
    return "\n".join(best)

def generate_answer(context: str, question: str, total_pages: int) -> str:
    if len(context) > MAX_CONTEXT_CHARS:
        context_trunc = context[:MAX_CONTEXT_CHARS] + "\n... [truncated]"
    else:
        context_trunc = context

    prompt = (
        f"You are a helpful assistant answering questions about a {total_pages}-page PDF.\n"
        "Use ONLY the provided context; if insufficient, say so.\n"
        "Cite page numbers in parentheses like (p. 12).\n\n"
        f"Context:\n{context_trunc}\n\nQuestion: {question}\nAnswer:"
    )

    if GENERATION_CLIENT and SELECTED_GEN_MODEL:
        # Try text_generation first
        try:
            out = GENERATION_CLIENT.text_generation(
                prompt,
                max_new_tokens=400,
                temperature=0.2,
                repetition_penalty=1.05
            )
            return out.strip()
        except Exception as e_text:
            # Try chat if available
            try:
                out_chat = GENERATION_CLIENT.chat_completion(
                    messages=[
                        {"role": "system", "content": "You answer only from provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=380,
                    temperature=0.2
                )
                # chat_completion may return dict with 'choices'
                if isinstance(out_chat, dict) and 'choices' in out_chat:
                    choice = out_chat['choices'][0]
                    # OpenAI-like vs plain content
                    if 'message' in choice and 'content' in choice['message']:
                        return choice['message']['content'].strip()
                return str(out_chat).strip()
            except Exception as e_chat:
                print(f"⚠️  Generation failed (text + chat). Falling back to extractive. Text err: {e_text}; Chat err: {e_chat}")
                return _extractive_fallback(context_trunc, question)
    else:
        return _extractive_fallback(context_trunc, question)


def ask_question(question: str):
    # Verify metadata file
    if not os.path.exists("chunks.pkl"):
        print("❌ Error: chunks.pkl not found. Run 'pdf-vector.py' first.")
        return None

    # Determine backend
    backend = None
    has_faiss_index = os.path.exists("vectors.index")
    has_numpy_vectors = os.path.exists("vectors.npy")

    if has_faiss_index:
        backend = 'faiss'
    elif has_numpy_vectors:
        backend = 'numpy'
    else:
        print("❌ Error: No vector storage (vectors.index or vectors.npy) found.")
        return None

    try:
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)
        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']
        stored_backend = data.get('backend', backend)
        if stored_backend != backend:
            print(f"⚠️  Detected backend mismatch (metadata={stored_backend}, files={backend}). Proceeding with {backend}.")

        # Prepare query embedding
        query_vec = embed_text(question).astype('float32')
        query_vec_2d = query_vec.reshape(1, -1)

        if backend == 'faiss' and USE_FAISS:
            index = faiss.read_index("vectors.index")
            # FAISS index expects normalized vectors; query already normalized
            scores, indices = index.search(query_vec_2d, TOP_K)
            retrieved_indices = indices[0]
            retrieved_scores = scores[0]
        else:
            if backend == 'faiss' and not USE_FAISS:
                print("⚠️  FAISS index present but faiss library unavailable; cannot query.")
                return None
            # numpy backend
            doc_embeddings = np.load("vectors.npy")  # already L2-normalized
            sims = doc_embeddings @ query_vec  # cosine similarity due to normalization
            retrieved_indices = np.argsort(-sims)[:TOP_K]
            retrieved_scores = sims[retrieved_indices]

        print(f"🔍 Found {len(retrieved_indices)} relevant chunks:")
        context_parts = []
        for rank, (idx, score) in enumerate(zip(retrieved_indices, retrieved_scores), start=1):
            page_num = metadata[idx]['estimated_page']
            print(f"   Chunk {rank}: Score {score:.3f} (≈Page {page_num})")
            snippet = chunks[idx]
            context_parts.append(f"[Page {page_num}] {snippet}")

        context = "\n\n".join(context_parts)
        answer = generate_answer(context, question, total_pages)
        return answer
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        return None


def main():
    # Basic existence checks
    if not os.path.exists("chunks.pkl"):
        print("❌ Vector data not found. Run: python pdf-vector.py")
        return

    with open("chunks.pkl", "rb") as f:
        data = pickle.load(f)
    chunks = data['chunks']
    total_pages = data['total_pages']

    # Determine dim dynamically (load one embedding file if needed)
    vector_dim = None
    if os.path.exists("vectors.index") and USE_FAISS:
        try:
            idx_tmp = faiss.read_index("vectors.index")
            vector_dim = idx_tmp.d
        except Exception:
            pass
    if vector_dim is None and os.path.exists("vectors.npy"):
        arr = np.load("vectors.npy", mmap_mode='r')
        vector_dim = arr.shape[1]
    if vector_dim is None:
        vector_dim = 384  # fallback expected dim for MiniLM

    print("\n" + "=" * 60)
    print("🤖 Hugging Face RAG Ready (no OpenAI)")
    print("💡 Type 'info' for stats; 'bye' to exit")
    print("� Embedding model:", EMBED_MODEL)
    if SELECTED_GEN_MODEL:
        print("🧠 Generation model:", SELECTED_GEN_MODEL)
    else:
        print("🧠 Generation model: (none - using extractive fallback)")
    print("=" * 60)

    while True:
        try:
            question = input("\n❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break

        if question.lower() in {"bye", "quit", "exit", "q"}:
            print("👋 Goodbye! Thanks for using the RAG system!")
            break

        if question.lower() == 'info':
            print("📊 Database Info:")
            print(f"   • Total pages: {total_pages}")
            print(f"   • Total chunks: {len(chunks)}")
            print(f"   • Vector dimensions: {vector_dim}")
            print(f"   • Average chunks per page: {len(chunks)/total_pages:.1f}")
            print(f"   • Sample chunk: {chunks[0][:100]}...")
            continue

        if not question:
            print("⚠️  Please enter a question!")
            continue

        print("🔍 Retrieving & generating answer...")
        answer = ask_question(question)
        if answer:
            print(f"\n🤖 Answer:\n{answer}")
        else:
            print("❌ No answer generated.")


if __name__ == "__main__":
    main()