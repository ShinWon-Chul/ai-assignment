"""Prompt templates for RAG agent."""

from typing import Any

SYSTEM_PROMPT = """당신은 한국 법률 전문가입니다.
객관식 법률 문제를 분석하고 정답을 선택하는 것이 당신의 임무입니다.

주어진 유사 문제들을 참고하여, 법률 원칙과 논리적 추론을 바탕으로 정답을 도출하세요.
반드시 A, B, C, D 중 하나만 선택해야 합니다.

답변 형식:
- 첫 줄에 반드시 정답(A, B, C, D 중 하나)만 출력하세요.
- 추가 설명은 필요하지 않습니다."""


def create_rag_prompt(query: str, retrieved_docs: list[dict[str, Any]]) -> str:
    """
    Create RAG prompt with retrieved context.

    Args:
        query: The current question to answer
        retrieved_docs: List of retrieved similar questions

    Returns:
        Formatted prompt string
    """
    # Build context from retrieved documents
    context = "=== 참고할 유사 문제들 ===\n\n"

    for i, doc in enumerate(retrieved_docs, 1):
        similarity_pct = doc["similarity"] * 100
        context += f"[유사 문제 {i}] (유사도: {similarity_pct:.1f}%)\n"
        context += doc["text"]
        context += "\n\n"

    # Build full prompt
    prompt = f"""{context}
{'=' * 50}

=== 현재 풀어야 할 문제 ===

{query}

위 유사 문제들의 패턴과 법률 원칙을 참고하여 정답을 선택하세요.
반드시 A, B, C, 또는 D 중 하나의 문자만 출력하세요."""

    return prompt


def create_zero_shot_prompt(query: str) -> str:
    """
    Create zero-shot prompt without retrieval (fallback).

    Args:
        query: The question to answer

    Returns:
        Formatted prompt string
    """
    prompt = f"""다음 법률 문제를 풀어주세요.

{query}

반드시 A, B, C, 또는 D 중 하나의 문자만 출력하세요."""

    return prompt

