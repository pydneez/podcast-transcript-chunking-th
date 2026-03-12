import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def process_podcast_semantic_chunks(json_file_path, output_json_path, target_chars=1000):
    # 1. Load the raw JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get('segments', [])
    episode_name = os.path.basename(json_file_path).replace('.json', '')
    
    # 2. Setup Local LLM for Refinement
    print("Initializing Local Ollama (Qwen 2.5) for cleaning...")
    llm = ChatOllama(model="qwen2.5", temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Thai language editor. Your job is to clean up podcast transcripts.
        Strict Instructions:
        1. Correct spelling mistakes and common ASR phonetic errors (e.g., 'กระหนกวัน' -> 'กนกวรรณ', 'จุลาลงกร' -> 'จุฬาลงกรณ์').
        2. Remove filler words (เอ่อ, อ่า, แบบ) unless they add crucial context.
        3. Improve sentence flow and add appropriate spacing for readability.
        4. DO NOT summarize or rewrite the core meaning. Keep the conversational tone.
        5. DO NOT include any speaker tags, labels, or names (e.g., 'Speaker 00:') in your output.
        6. Output ONLY the cleaned text, nothing else."""),
        ("user", "Refine this transcript chunk: {raw_text}")
    ])
    cleaning_chain = prompt | llm | StrOutputParser()

    # 3. Conversational Grouping Logic
    structured_chunks = []
    current_chunk_text = ""
    current_start = segments[0]['start']
    speakers_in_chunk = set()
    chunk_index = 0

    print("Grouping segments into conversation blocks...")
    for i, seg in enumerate(segments):
        speaker = seg.get('speaker', 'UNKNOWN')
        text = seg.get('text', '').strip()
        
        # Prepend speaker tag to preserve context inside the embedding
        formatted_turn = f"[{speaker}]: {text} "

        # Check if adding this turn exceeds our target size
        if len(current_chunk_text) + len(formatted_turn) > target_chars and current_chunk_text:
            # A. Refine the whole block at once (Better context for LLM)
            print(f"Refining Chunk {chunk_index}...")
            cleaned_text = cleaning_chain.invoke({"raw_text": current_chunk_text.strip()})

            # B. Save the chunk
            chunk_data = {
                "chunk_id": f"{episode_name}_sem_{chunk_index:03d}",
                "text": cleaned_text,
                "metadata": {
                    "episode_id": episode_name,
                    "speakers": list(speakers_in_chunk),
                    "start_time": current_start,
                    "end_time": segments[i-1]['end'],
                    "char_count": len(cleaned_text)
                }
            }
            structured_chunks.append(chunk_data)
            
            # C. Reset for next block
            chunk_index += 1
            current_chunk_text = formatted_turn
            current_start = seg['start']
            speakers_in_chunk = {speaker}
        else:
            current_chunk_text += formatted_turn

    # Handle the final remaining segments
    if current_chunk_text:
        cleaned_text = cleaning_chain.invoke({"raw_text": current_chunk_text.strip()})
        structured_chunks.append({
            "chunk_id": f"{episode_name}_sem_{chunk_index:03d}",
            "text": cleaned_text,
            "metadata": {
                "episode_id": episode_name,
                "speakers": list(speakers_in_chunk),
                "start_time": current_start,
                "end_time": segments[-1]['end']
            }
        })

    # 4. Save Structured JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(structured_chunks, f, ensure_ascii=False, indent=4)
    
    # 5. Generate Semantic Embeddings
    print("Generating embeddings for conversational blocks...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode([c['text'] for c in structured_chunks])
    
    return structured_chunks, embeddings

if __name__ == "__main__":
    file_path = './transcript/Podcast_01.json'
    output_path = './output/Semantic_Chunks_EP01.json'
    mat_path = './output/Semantic_MAT_01.npy'
    
    os.makedirs('./transcript', exist_ok=True)
    chunks, embs = process_podcast_semantic_chunks(file_path, output_path)
    np.save(mat_path, embs)
    print(f"Finished! Matrix shape: {embs.shape}")