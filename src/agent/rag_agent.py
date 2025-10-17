"""RAG Agent for legal question answering."""

import os
from typing import Literal

from openai import AsyncOpenAI

from src.agent.prompts import SYSTEM_PROMPT, create_rag_prompt, create_zero_shot_prompt
from src.agent.retriever import QuestionRetriever


class RAGAgent:
    """RAG-based agent for answering legal multiple-choice questions."""

    def __init__(
        self,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_hybrid: bool = True,
        category_boost: float = 0.0,
        retriever: QuestionRetriever | None = None,
    ) -> None:
        """
        Initialize the RAG agent.

        Args:
            top_k: Number of documents to retrieve
            semantic_weight: Weight for semantic search
            bm25_weight: Weight for BM25 search
            use_hybrid: Whether to use hybrid search
            category_boost: Boost score for same-category documents (0.0-0.3)
            retriever: Optional pre-initialized retriever (for efficiency in batch experiments)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.top_k = top_k
        self.category_boost = category_boost

        # Use provided retriever or create new one
        if retriever is not None:
            self.retriever = retriever
            self.use_rag = True
            # Update weights if provided
            if self.retriever.use_hybrid:
                self.retriever.set_weights(semantic_weight, bm25_weight, category_boost)
            print(
                f"✓ RAG Agent initialized with existing retriever (top_k={top_k}, category_boost={category_boost})"
            )
        else:
            # Initialize new retriever
            try:
                self.retriever = QuestionRetriever(
                    semantic_weight=semantic_weight,
                    bm25_weight=bm25_weight,
                    use_hybrid=use_hybrid,
                    category_boost=category_boost,
                )
                self.use_rag = True
                hybrid_status = (
                    f"Hybrid (semantic={semantic_weight}, bm25={bm25_weight})"
                    if use_hybrid
                    else "Semantic only"
                )
                category_status = (
                    f", category_boost={category_boost}" if category_boost > 0 else ""
                )
                print(
                    f"✓ RAG Agent initialized with vector store ({hybrid_status}{category_status}, top_k={top_k})"
                )
            except Exception as e:
                print(f"⚠ Warning: Could not initialize retriever: {e}")
                print("⚠ Falling back to zero-shot mode")
                self.retriever = None
                self.use_rag = False

    async def predict(
        self, query: str, category: str | None = None
    ) -> Literal["A", "B", "C", "D"]:
        """
        Predict the answer for a given question using RAG.

        Args:
            query: The multiple-choice question with options
            category: Optional category for category-aware retrieval

        Returns:
            The predicted answer (A, B, C, or D)
        """
        try:
            if self.use_rag and self.retriever:
                return await self._predict_with_rag(query, category)
            else:
                return await self._predict_zero_shot(query)
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Fallback to zero-shot
            return await self._predict_zero_shot(query)

    async def _predict_with_rag(
        self, query: str, category: str | None = None
    ) -> Literal["A", "B", "C", "D"]:
        """
        Predict using RAG (Retrieval-Augmented Generation).

        Args:
            query: The question
            category: Optional category for category-aware retrieval

        Returns:
            The predicted answer
        """
        # 1. Retrieve similar questions (with category-aware boost if provided)
        retrieved = await self.retriever.retrieve(
            query=query, top_k=self.top_k, category=category
        )  # type: ignore[union-attr]

        if not retrieved:
            print("No similar questions found, using zero-shot")
            return await self._predict_zero_shot(query)

        # 2. Create RAG prompt
        prompt = create_rag_prompt(query, retrieved)

        # 3. Get LLM response
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        # 4. Parse answer
        answer_text = response.choices[0].message.content
        if answer_text:
            answer = self._extract_answer(answer_text)
            if answer:
                return answer

        # Fallback: use most similar question's answer
        print("Could not parse LLM answer, using most similar question's answer")
        return retrieved[0]["answer"]  # type: ignore[return-value]

    async def _predict_zero_shot(self, query: str) -> Literal["A", "B", "C", "D"]:
        """
        Predict using zero-shot (no retrieval).

        Args:
            query: The question

        Returns:
            The predicted answer
        """
        prompt = create_zero_shot_prompt(query)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        answer_text = response.choices[0].message.content
        if answer_text:
            answer = self._extract_answer(answer_text)
            if answer:
                return answer

        # Last resort fallback
        return "A"

    def _extract_answer(self, text: str) -> Literal["A", "B", "C", "D"] | None:
        """
        Extract answer letter from LLM response.

        Args:
            text: LLM response text

        Returns:
            Extracted answer or None
        """
        text = text.strip()

        # Direct match
        if text in ["A", "B", "C", "D"]:
            return text  # type: ignore[return-value]

        # Find first occurrence of A, B, C, or D
        for letter in ["A", "B", "C", "D"]:
            if letter in text:
                return letter  # type: ignore[return-value]

        return None
