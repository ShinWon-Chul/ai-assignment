"""Retriever for similar legal questions with hybrid search (Semantic + BM25)."""

import json
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi


class QuestionRetriever:
    """Retrieve similar questions from vector store using hybrid search."""

    def __init__(
        self,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_hybrid: bool = True,
        category_boost: float = 0.0,
    ) -> None:
        """
        Initialize the retriever with hybrid search.

        Args:
            semantic_weight: Weight for semantic search (default: 0.7)
            bm25_weight: Weight for BM25 search (default: 0.3)
            use_hybrid: Whether to use hybrid search or semantic only (default: True)
            category_boost: Boost score for same-category documents (default: 0.0, range: 0.0-0.3)
        """
        # Store hybrid search settings
        self.use_hybrid = use_hybrid
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.category_boost = category_boost

        # Get project root
        root = Path(__file__).parent.parent.parent
        vector_store_path = root / "vector_store" / "chroma"
        documents_path = root / "data" / "processed" / "documents.json"

        if not vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {vector_store_path}. "
                "Run generate_embeddings.py first."
            )

        # Initialize clients
        self.openai_client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=str(vector_store_path))

        try:
            self.collection = self.chroma_client.get_collection("legal_questions")
        except Exception as e:
            raise RuntimeError(
                "Failed to load 'legal_questions' collection. "
                "Run generate_embeddings.py first."
            ) from e

        # Initialize BM25 for hybrid search
        self.documents = []
        self.bm25 = None

        if use_hybrid:
            # Load documents for BM25
            if not documents_path.exists():
                raise FileNotFoundError(
                    f"Documents not found at {documents_path}. "
                    "Run process_data.py first."
                )

            with open(documents_path, encoding="utf-8") as f:
                self.documents = json.load(f)

            # Build BM25 index
            print(f"Building BM25 index from {len(self.documents)} documents...")
            tokenized_corpus = [doc["text"].split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("✓ BM25 index built successfully")

    async def retrieve(
        self, query: str, top_k: int = 5, category: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve similar questions using hybrid search (Semantic + BM25).

        Args:
            query: The question to search for
            top_k: Number of similar questions to retrieve
            category: Optional category filter (e.g., "Law", "Criminal Law")

        Returns:
            List of similar questions with metadata, ranked by hybrid score
        """
        if not self.use_hybrid:
            # Use semantic search only (original behavior)
            return await self._semantic_search(query, top_k, category)

        # Hybrid search: Semantic + BM25
        # Step 1: Get more candidates from both methods
        candidates_k = top_k * 3  # Get 3x candidates for better reranking

        # Semantic search
        semantic_results = await self._semantic_search(query, candidates_k, category)

        # BM25 search
        bm25_results = self._bm25_search(query, candidates_k, category)

        # Step 2: Combine scores with linear combination
        combined_scores = {}

        # Add semantic scores
        for result in semantic_results:
            doc_id = result["id"]
            # Normalize semantic similarity to [0, 1]
            semantic_score = result["similarity"]
            combined_scores[doc_id] = {
                "semantic_score": semantic_score,
                "bm25_score": 0.0,
                "doc": result,
            }

        # Add BM25 scores
        for result in bm25_results:
            doc_id = result["id"]
            bm25_score = result["bm25_score"]

            if doc_id in combined_scores:
                combined_scores[doc_id]["bm25_score"] = bm25_score
            else:
                # Add document info if not in semantic results
                combined_scores[doc_id] = {
                    "semantic_score": 0.0,
                    "bm25_score": bm25_score,
                    "doc": result,
                }

        # Step 3: Calculate final hybrid scores with category boost
        for doc_id in combined_scores:
            semantic_score = combined_scores[doc_id]["semantic_score"]
            bm25_score = combined_scores[doc_id]["bm25_score"]

            # Linear combination
            hybrid_score = (
                self.semantic_weight * semantic_score + self.bm25_weight * bm25_score
            )

            # Apply category boost if same category
            if category and self.category_boost > 0:
                doc_category = combined_scores[doc_id]["doc"].get("category", "")
                if doc_category == category:
                    hybrid_score += self.category_boost
                    combined_scores[doc_id]["category_boosted"] = True
                else:
                    combined_scores[doc_id]["category_boosted"] = False
            else:
                combined_scores[doc_id]["category_boosted"] = False

            combined_scores[doc_id]["hybrid_score"] = hybrid_score

        # Step 4: Sort by hybrid score and return top_k
        ranked_docs = sorted(
            combined_scores.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:top_k]

        # Step 5: Format results
        results = []
        for item in ranked_docs:
            doc = item["doc"]
            doc["hybrid_score"] = item["hybrid_score"]
            doc["semantic_score"] = item["semantic_score"]
            doc["bm25_score"] = item["bm25_score"]
            doc["category_boosted"] = item["category_boosted"]
            results.append(doc)

        return results

    async def _semantic_search(
        self, query: str, top_k: int, category: str | None = None
    ) -> list[dict[str, Any]]:
        """Perform semantic search using embeddings."""
        # Generate query embedding
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small", input=query
        )
        query_embedding = response.data[0].embedding

        # Build filter
        where = {"category": category} if category else None

        # Search in vector store
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )

        # Format results
        retrieved = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                retrieved.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "question": metadata.get("question", ""),
                        "answer": metadata["answer"],
                        "category": metadata["category"],
                        "difficulty": metadata.get("difficulty", 0.0),
                        "distance": results["distances"][0][i],
                        "similarity": 1 - results["distances"][0][i],
                    }
                )

        return retrieved

    def _bm25_search(
        self, query: str, top_k: int, category: str | None = None
    ) -> list[dict[str, Any]]:
        """Perform BM25 keyword-based search."""
        if self.bm25 is None:
            return []

        # Tokenize query
        tokenized_query = query.split()

        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[: top_k * 2]  # Get more candidates for category filtering

        # Format results with category filtering
        retrieved = []
        for idx in top_indices:
            doc = self.documents[idx]

            # Apply category filter if specified
            if category and doc.get("category") != category:
                continue

            # Normalize BM25 score to [0, 1] range
            # Using sigmoid-like normalization
            score = bm25_scores[idx]
            normalized_score = score / (score + 1.0)  # Simple normalization

            retrieved.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer_letter", ""),
                    "category": doc.get("category", ""),
                    "difficulty": doc.get("difficulty", 0.0),
                    "bm25_score": normalized_score,
                    "similarity": normalized_score,  # For compatibility
                }
            )

            if len(retrieved) >= top_k:
                break

        return retrieved

    def set_weights(
        self, semantic_weight: float, bm25_weight: float, category_boost: float = None
    ) -> None:
        """
        Update the weights for hybrid search.

        Args:
            semantic_weight: Weight for semantic search
            bm25_weight: Weight for BM25 search
            category_boost: Optional category boost (if None, keeps current value)
        """
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        if category_boost is not None:
            self.category_boost = category_boost
        print(
            f"✓ Updated weights: semantic={semantic_weight:.2f}, bm25={bm25_weight:.2f}, category_boost={self.category_boost:.2f}"
        )
