"""
Step 5: Lightweight RAG Pipeline
===================================
RAG = Retrieval-Augmented Generation

How RAG works:
1. INDEXING: Split resume into chunks → Create embeddings → Store in vector DB
2. RETRIEVAL: User query → Embed query → Find similar chunks
3. GENERATION: Pass relevant chunks + query to LLM → Get contextual response

Why RAG over simple prompting?
- LLMs have context limits
- RAG lets us find the MOST RELEVANT parts of a long resume
- Results are grounded in actual resume content
- Reduces hallucination

This is a LIGHTWEIGHT implementation using:
- Simple TF-IDF style similarity (no heavy ML libraries needed)
- In-memory "vector store" (no FAISS needed for small resumes)
- Falls back gracefully if sentence-transformers not available
"""

import re
import math
from typing import List, Dict, Tuple
from collections import Counter


# ─── Text Chunking ────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict]:
    """
    Split resume text into overlapping chunks.
    
    Why overlap? So context at chunk boundaries isn't lost.
    
    Args:
        text: Full resume text
        chunk_size: Target words per chunk
        overlap: Words to repeat between chunks
    Returns:
        list of dicts: [{chunk_id, text, word_count, start_word}]
    """
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    start = 0
    chunk_id = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            'chunk_id': chunk_id,
            'text': chunk_text,
            'word_count': len(chunk_words),
            'start_word': start,
            'end_word': end
        })
        
        # Move forward, keeping overlap
        start = end - overlap
        chunk_id += 1
        
        # Stop if we'd just repeat
        if end == len(words):
            break
    
    return chunks


# ─── Simple TF-IDF Embeddings ────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, remove punctuation, split."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Term Frequency: how often each word appears in this chunk."""
    if not tokens:
        return {}
    counter = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counter.items()}


def compute_idf(chunks: List[Dict]) -> Dict[str, float]:
    """
    Inverse Document Frequency: rare words are more important.
    IDF(word) = log(total_docs / docs_containing_word)
    """
    total_docs = len(chunks)
    word_doc_count = {}
    
    for chunk in chunks:
        unique_words = set(tokenize(chunk['text']))
        for word in unique_words:
            word_doc_count[word] = word_doc_count.get(word, 0) + 1
    
    idf = {}
    for word, count in word_doc_count.items():
        idf[word] = math.log(total_docs / count) if count > 0 else 0
    
    return idf


def compute_tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """TF-IDF vector = TF × IDF for each word."""
    tf = compute_tf(tokens)
    return {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}


def cosine_similarity(vec1: Dict, vec2: Dict) -> float:
    """
    Cosine similarity between two TF-IDF vectors.
    Range: 0 (no similarity) to 1 (identical).
    
    cosine_sim = (A · B) / (|A| × |B|)
    """
    # Dot product
    common_words = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[w] * vec2[w] for w in common_words)
    
    # Magnitudes
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


# ─── Vector Store (In-Memory) ─────────────────────────────────────────────────

class SimpleVectorStore:
    """
    Lightweight in-memory vector store.
    Stores TF-IDF vectors for each chunk.
    
    In production, replace this with FAISS:
        import faiss
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(embeddings))
    """
    
    def __init__(self):
        self.chunks = []
        self.vectors = []
        self.idf = {}
        self.is_indexed = False
    
    def index(self, chunks: List[Dict]):
        """
        Build the vector index from chunks.
        This is the 'indexing' phase of RAG.
        """
        self.chunks = chunks
        
        # Compute IDF across all chunks
        self.idf = compute_idf(chunks)
        
        # Compute TF-IDF vector for each chunk
        self.vectors = []
        for chunk in chunks:
            tokens = tokenize(chunk['text'])
            vector = compute_tfidf_vector(tokens, self.idf)
            self.vectors.append(vector)
        
        self.is_indexed = True
        return self
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Find most relevant chunks for a query.
        This is the 'retrieval' phase of RAG.
        
        Args:
            query: User's question or topic
            top_k: Number of chunks to return
        Returns:
            list: Top-k most relevant chunks with similarity scores
        """
        if not self.is_indexed or not self.chunks:
            return []
        
        # Embed the query using same TF-IDF approach
        query_tokens = tokenize(query)
        query_vector = compute_tfidf_vector(query_tokens, self.idf)
        
        # Calculate similarity with each chunk
        similarities = []
        for i, chunk_vector in enumerate(self.vectors):
            sim = cosine_similarity(query_vector, chunk_vector)
            similarities.append((i, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for idx, sim_score in similarities[:top_k]:
            result = dict(self.chunks[idx])  # copy
            result['similarity_score'] = round(sim_score, 4)
            results.append(result)
        
        return results


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class ResumeRAGPipeline:
    """
    Complete RAG pipeline for resume analysis.
    
    Usage:
        pipeline = ResumeRAGPipeline(resume_text)
        context = pipeline.get_context_for_query("What are the candidate's Python skills?")
    """
    
    def __init__(self, resume_text: str):
        self.resume_text = resume_text
        self.vector_store = SimpleVectorStore()
        self._build_index()
    
    def _build_index(self):
        """Chunk resume and build vector index."""
        chunks = split_into_chunks(
            self.resume_text,
            chunk_size=150,  # smaller chunks for resumes
            overlap=30
        )
        self.vector_store.index(chunks)
        self.chunk_count = len(chunks)
    
    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant resume context for a query.
        
        Args:
            query: What we're asking about
            top_k: Number of chunks to retrieve
        Returns:
            str: Concatenated relevant chunks (the 'context' for LLM)
        """
        relevant_chunks = self.vector_store.search(query, top_k=top_k)
        
        if not relevant_chunks:
            return self.resume_text[:1000]  # fallback: use start of resume
        
        # Combine chunks, removing duplicates
        seen_texts = set()
        context_parts = []
        
        for chunk in relevant_chunks:
            text = chunk['text']
            # Simple deduplication
            text_key = text[:50]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                context_parts.append(text)
        
        return '\n\n'.join(context_parts)
    
    def get_pipeline_info(self) -> Dict:
        """Return info about the pipeline for display."""
        return {
            'chunk_count': self.chunk_count,
            'resume_length': len(self.resume_text.split()),
            'index_type': 'TF-IDF Vector Store (Lightweight)',
            'similarity_metric': 'Cosine Similarity'
        }
