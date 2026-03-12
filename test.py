import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 
with open('./output/Turn_Chunks_EP01.json', 'r', encoding='utf-8') as f:
    chunks_data = json.load(f)
    embeddings_matrix = np.load('./output/Turn_MAT_01.npy')

# 2. 
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def search(query, top_k=3):
    query_vector = model.encode([query])
    
    # find cosine similarity between question and every chunk
    # result array
    similarities = cosine_similarity(query_vector, embeddings_matrix)[0]
    
    # get top k chunk
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\nคำถาม: {query}")
    print("-" * 30)
    for idx in top_indices:
        score = similarities[idx]
        text = chunks_data[idx]['text']
        print(f"[Score: {score:.4f}] Chunk ID: {chunks_data[idx]['chunk_id']}")
        print(f"เนื้อหา: {text[:200]}...\n")

# --- ทดสอบระบบ ---
search("วิธีการป้องกันโรคอัลไซเมอร์")