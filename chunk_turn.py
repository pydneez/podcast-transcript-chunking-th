import json
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def process_podcast_embeddings(json_file_path, output_json_path):
    # 1. Load data using Path for reliability
    p = Path(json_file_path)
    data = json.loads(p.read_text(encoding='utf-8'))
    
    segments = data.get('segments', [])
    episode_name = p.stem
    
    # 2. Refined Turn-Based Chunking (Matching your dialog logic)
    structured_chunks = []
    current_speaker = None
    current_text = []
    current_start = 0.0
    chunk_index = 0

    for seg in segments:
        speaker = seg.get('speaker', 'UNKNOWN')
        text = seg.get('text', '').strip()
        
        if not text:
            continue

        if speaker != current_speaker:
            # If there was a previous speaker, save their completed turn
            if current_speaker is not None:
                combined_text = " ".join(current_text)
                structured_chunks.append({
                    "chunk_id": f"{episode_name}_chunk_{chunk_index:03d}",
                    "text": combined_text,
                    "metadata": {
                        "episode_id": episode_name,
                        "speaker": current_speaker,
                        "start_time": current_start,
                        "end_time": segments[segments.index(seg)-1]['end'],
                        "turn_index": chunk_index
                    }
                })
                chunk_index += 1
            
            # Reset for the new speaker
            current_speaker = speaker
            current_start = seg.get('start', 0.0)
            current_text = [text]
        else:
            # Same speaker: append the segment text
            current_text.append(text)

    # Add the final turn
    if current_speaker is not None:
        combined_text = " ".join(current_text)
        structured_chunks.append({
            "chunk_id": f"{episode_name}_chunk_{chunk_index:03d}",
            "text": combined_text,
            "metadata": {
                "episode_id": episode_name,
                "speaker": current_speaker,
                "start_time": current_start,
                "end_time": segments[-1]['end'],
                "turn_index": chunk_index
            }
        })

    print(f"Created {len(structured_chunks)} turn-based chunks.")

    # 3. Local LLM Refinement Pipeline
    print("Initializing Ollama (Qwen 2.5)...")
    llm = ChatOllama(model="qwen2.5", temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Thai language editor. Your job is to clean up podcast transcripts.
        Strict Instructions:
        1. Correct spelling mistakes and common ASR phonetic errors based on the surrounding context (e.g., 'กระหนกวัน' -> 'กนกวรรณ', 'จุลาลงกร' -> 'จุฬาลงกรณ์').
        2. Remove filler words (เอ่อ, อ่า, แบบ) unless they add crucial context.
        3. Improve sentence flow and add appropriate spacing for readability.
        4. DO NOT summarize or rewrite the core meaning. Keep the conversational tone of the podcast.
        5. DO NOT include any speaker tags, labels, or names (e.g., 'Speaker 00:') in your output.
        6. Output ONLY the cleaned text, nothing else."""),
        ("user", "{raw_text}")
    ])
    cleaning_chain = prompt | llm | StrOutputParser()

    for i, chunk in enumerate(structured_chunks):
        print(f"Refining chunk {i+1}/{len(structured_chunks)}...")
        cleaned = cleaning_chain.invoke({"raw_text": chunk['text']})
        chunk["text"] = cleaned.strip()
        chunk["metadata"]["char_count"] = len(chunk["text"])

    # 4. Save and Embed
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(structured_chunks, f, ensure_ascii=False, indent=4)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode([c['text'] for c in structured_chunks])
    
    return structured_chunks, embeddings

if __name__ == "__main__":
    file_path = './transcript/Podcast_01.json'
    out_json = './output/Turn_Chunks_EP01.json'
    out_mat = './output/Turn_MAT_01.npy'
    
    chunks, matrix = process_podcast_embeddings(file_path, out_json)
    np.save(out_mat, matrix)
    print(f"Matrix saved with shape: {matrix.shape}")