import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Configuration
INPUT_FILE = './output/Turn_Chunks_EP01.json'
OUTPUT_FILE = './output/Turn_Semantic_Merged_Chunks.json'
THRESHOLD = 0.70  # Adjust this: 0.7 is broad merging, 0.8 is strict

def run_semantic_merging():
    # Load your already refined chunks
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please check your path.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} refined turns. Starting semantic merging...")

    # Load the multilingual model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Pre-calculate embeddings for all individual turns to save compute
    print("Generating embeddings for turns...")
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts)

    final_merged_chunks = []
    
    # The grouping logic
    current_group_indices = [0] # Start with the first turn
    
    i = 1
    while i < len(chunks):
        # Calculate the 'Context Vector' for the current merged block
        # We take the average embedding of all turns currently in the block
        current_context_vector = np.mean(embeddings[current_group_indices], axis=0).reshape(1, -1)
        
        # Get the embedding for the next candidate turn
        next_turn_vector = embeddings[i].reshape(1, -1)
        
        # Calculate Cosine Similarity
        similarity = cosine_similarity(current_context_vector, next_turn_vector)[0][0]
        
        # LOGGING: See the math in action
        print(f"Checking Chunk {i}: Sim with current block = {similarity:.4f}")

        if similarity >= THRESHOLD:
            # Related! Add this turn to the current block
            current_group_indices.append(i)
            print(f"  -> [MERGE] Chunk {i} stays with the group.")
            i += 1
        else:
            # Not similar. Finalize the current block and start a new one.
            print(f"  -> [SPLIT] Topic change detected at Chunk {i}.")
            
            # Combine the text and metadata
            merged_text = " ".join([chunks[idx]['text'] for idx in current_group_indices])
            merged_chunk = {
                "chunk_id": f"merged_{len(final_merged_chunks):03d}",
                "text": merged_text.strip(),
                "metadata": {
                    "episode_id": chunks[0]['metadata']['episode_id'],
                    "start_time": chunks[current_group_indices[0]]['metadata']['start_time'],
                    "end_time": chunks[current_group_indices[-1]]['metadata']['end_time'],
                    "original_turn_count": len(current_group_indices),
                    "original_ids": [chunks[idx]['chunk_id'] for idx in current_group_indices]
                }
            }
            final_merged_chunks.append(merged_chunk)
            
            # Reset the block to start with the turn that caused the split
            current_group_indices = [i]
            i += 1

    # Don't forget to save the very last block
    if current_group_indices:
        merged_text = " ".join([chunks[idx]['text'] for idx in current_group_indices])
        final_merged_chunks.append({
            "chunk_id": f"merged_{len(final_merged_chunks):03d}",
            "text": merged_text.strip(),
            "metadata": {
                "episode_id": chunks[0]['metadata']['episode_id'],
                "start_time": chunks[current_group_indices[0]]['metadata']['start_time'],
                "end_time": chunks[current_group_indices[-1]]['metadata']['end_time'],
                "original_turn_count": len(current_group_indices)
            }
        })

    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_merged_chunks, f, ensure_ascii=False, indent=4)
    
    print("-" * 30)
    print(f"Success! {len(chunks)} turns reduced to {len(final_merged_chunks)} semantic chunks.")
    print(f"File saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_semantic_merging()