import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SemanticChunk:
    text: str
    interviewee_id: str
    interview_id: str
    product_name: str
    start_turn: int
    end_turn: int
    speaker_distribution: Dict[str, int]
    embedding: Optional[np.ndarray] = None
    

class TextProcessor:
    
    @staticmethod
    def clean_unicode(text: str) -> str:
        replacements = {
            '\u2019': "'",
            '\u2018': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u2013': '-',
            '\u2014': '--',
            '\u2026': '...',
            '\u00a0': ' ',
            '\u2022': '•',
        }
        
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
        
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = TextProcessor.clean_unicode(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        
        return text.strip()


class SemanticChunker:
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", similarity_threshold: float = 0.7):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.text_processor = TextProcessor()
    
    def extract_turns_text(self, transcript: List[Dict]) -> List[str]:
        turns = []
        for turn in transcript:
            speaker = turn.get('speaker', '')
            text = turn.get('text', '')
            
            speaker = self.text_processor.clean_text(speaker)
            text = self.text_processor.clean_text(text)
            
            turn_text = f"{speaker}: {text}"
            turns.append(turn_text)
        
        return turns
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def create_semantic_chunks(self, transcript_data: Dict[str, Any]) -> List[SemanticChunk]:
        transcript = transcript_data.get('transcript', [])
        metadata = transcript_data.get('metadata', {})
        
        if not transcript:
            return []
        
        turns_text = self.extract_turns_text(transcript)
        
        if len(turns_text) < 2:
            return []
        
        chunks = []
        current_chunk_turns = [0]
        
        for i in range(1, len(turns_text)):
            last_turn_idx = current_chunk_turns[-1]
            similarity = self.calculate_semantic_similarity(turns_text[last_turn_idx], turns_text[i])
            
            if similarity >= self.similarity_threshold:
                current_chunk_turns.append(i)
            else:
                if len(current_chunk_turns) >= 1:
                    chunk = self._create_chunk_from_turns(
                        current_chunk_turns, turns_text, transcript, metadata
                    )
                    if chunk:
                        chunks.append(chunk)
                
                current_chunk_turns = [i]
        
        if current_chunk_turns:
            chunk = self._create_chunk_from_turns(
                current_chunk_turns, turns_text, transcript, metadata
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_turns(self, turn_indices: List[int], turns_text: List[str], 
                                transcript: List[Dict], metadata: Dict) -> Optional[SemanticChunk]:
        if not turn_indices:
            return None
        
        chunk_text = " ".join(turns_text[i] for i in turn_indices)
        
        speaker_count = {}
        for i in turn_indices:
            speaker = transcript[i].get('speaker', 'Unknown')
            speaker = self.text_processor.clean_text(speaker)
            speaker_count[speaker] = speaker_count.get(speaker, 0) + 1
        
        interviewee_info = metadata.get('interviewee', '')
        interviewee_id = interviewee_info.split('(')[-1].replace(')', '') if '(' in interviewee_info else 'Unknown'
        
        chunk = SemanticChunk(
            text=chunk_text,
            interviewee_id=interviewee_id,
            interview_id=metadata.get('interviewId', ''),
            product_name=metadata.get('product', ''),
            start_turn=min(turn_indices) + 1,
            end_turn=max(turn_indices) + 1,
            speaker_distribution=speaker_count
        )
        
        return chunk


class QdrantRAG:
    
    def __init__(self, collection_name: str = "interview_transcripts", 
                 model_name: str = "BAAI/bge-base-en-v1.5",
                 qdrant_host: str = "localhost", qdrant_port: int = 6333):
        
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.chunker = SemanticChunker(model_name)
        
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        self._create_collection()
    
    def _create_collection(self):
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def load_and_process_transcripts(self, transcript_dir: str = "transcripts") -> List[SemanticChunk]:
        transcript_path = Path(transcript_dir)
        all_chunks = []
        
        if not transcript_path.exists():
            print(f"Transcript directory not found: {transcript_dir}")
            return []
        
        json_files = list(transcript_path.glob("*.json"))
        print(f"Found {len(json_files)} transcript files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                
                chunks = self.chunker.create_semantic_chunks(transcript_data)
                all_chunks.extend(chunks)
                print(f"Processed {json_file.name}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def embed_and_store_chunks(self, chunks: List[SemanticChunk]):
        if not chunks:
            print("No chunks to store")
            return
        
        print("Generating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk.text,
                    "interviewee_id": chunk.interviewee_id,
                    "interview_id": chunk.interview_id,
                    "product_name": chunk.product_name,
                    "start_turn": chunk.start_turn,
                    "end_turn": chunk.end_turn,
                    "speaker_distribution": chunk.speaker_distribution
                }
            )
            points.append(point)
        
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"Stored {len(points)} chunks in Qdrant")
    
    def search_relevant_transcripts(self, query: str, top_k: int = 10) -> List[str]:
        query_embedding = self.model.encode([query])[0]
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=0.3
        )
        
        seen_interviewees = set()
        relevant_interviewees = []
        
        for result in search_results:
            interviewee_id = result.payload.get("interviewee_id")
            if interviewee_id and interviewee_id not in seen_interviewees:
                seen_interviewees.add(interviewee_id)
                relevant_interviewees.append(interviewee_id)
                
                if len(relevant_interviewees) >= 3:
                    break
        
        return relevant_interviewees
    
    def get_detailed_results(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query])[0]
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=0.3
        )
        
        detailed_results = []
        for result in search_results:
            detailed_results.append({
                "score": result.score,
                "interviewee_id": result.payload.get("interviewee_id"),
                "product_name": result.payload.get("product_name"),
                "text_snippet": result.payload.get("text")[:200] + "...",
                "interview_id": result.payload.get("interview_id"),
                "turns": f"{result.payload.get('start_turn')}-{result.payload.get('end_turn')}"
            })
        
        return detailed_results
    
    def build_index(self, transcript_dir: str = "transcripts"):
        print("Starting RAG index building...")
        
        chunks = self.load_and_process_transcripts(transcript_dir)
        
        if not chunks:
            print("No chunks created. Check your transcript directory and files.")
            return
        
        self.embed_and_store_chunks(chunks)
        
        print("RAG index building completed!")
    
    def query(self, user_query: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        interviewee_ids = self.search_relevant_transcripts(user_query)
        detailed_results = self.get_detailed_results(user_query)
        
        return interviewee_ids, detailed_results


if __name__ == "__main__":
    rag = QdrantRAG()
    
    rag.build_index("transcripts")
    
    test_queries = [
        "What are some favorites in the headphones category and what makes them successful",
        "What do users think of my airfryer lineup of the brand COSORI",
        "What features do popular non-analog watches on the market have",
        "How does battery life play into consumer appeal",
        "Why are electric toothbrushes popular"
    ]
    
    print("\n" + "="*50)
    print("RAG SYSTEM QUERY EXAMPLES")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            interviewee_ids, detailed_results = rag.query(query)
            
            print("Top 3 Relevant Interviewee IDs:")
            for i, interviewee_id in enumerate(interviewee_ids, 1):
                print(f"  {i}. {interviewee_id}")
            
            print("\nDetailed Results:")
            for i, result in enumerate(detailed_results[:3], 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Interviewee: {result['interviewee_id']}")
                print(f"     Product: {result['product_name']}")
                print(f"     Snippet: {result['text_snippet']}")
                print()
        
        except Exception as e:
            print(f"Error processing query: {e}")