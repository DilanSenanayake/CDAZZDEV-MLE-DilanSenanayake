"""
Pydantic schemas for Task 1B structured LLM outputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HeadlineSentiment(BaseModel):
    headline: str = Field(..., min_length=1)
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    brief_reason: str = Field(..., min_length=1)

    @field_validator("brief_reason")
    @classmethod
    def trim_reason(cls, value: str) -> str:
        return value.strip()


class SentimentBatchResult(BaseModel):
    items: list[HeadlineSentiment]
    overall_sentiment_score: float = Field(
        ...,
        description="Aggregate score in [-1, 1] where -1=bearish, 1=bullish",
    )
    overall_label: Literal["positive", "negative", "neutral"]


class TradeSignal(BaseModel):
    signal: Literal["Buy", "Hold", "Sell"]
    justification: str = Field(..., min_length=20)
    key_factors: list[str] = Field(default_factory=list)

    @field_validator("justification")
    @classmethod
    def trim_justification(cls, value: str) -> str:
        text = value.strip()
        if len(text.split()) < 3:
            raise ValueError("justification must be several sentences of reasoning")
        return text
