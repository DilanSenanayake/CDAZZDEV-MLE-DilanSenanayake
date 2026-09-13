"""Pydantic schemas for multi-agent handoffs (Task 3B)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RiskItem(BaseModel):
    title: str
    evidence: str
    horizon_days: int = 90


class DataAnalystBrief(BaseModel):
    ticker: str
    financial_health_summary: str
    price_snapshot: dict
    volatility: dict
    sentiment: dict
    top_risks_draft: list[RiskItem] = Field(default_factory=list)
    notes_for_writer: str = ""


class ClarificationRequest(BaseModel):
    question: str
    needed_fields: list[str] = Field(default_factory=list)


class ClarificationResponse(BaseModel):
    question: str
    answer: str
    data: dict = Field(default_factory=dict)


class FinalResearchReport(BaseModel):
    ticker: str
    financial_health_summary: str
    top_three_risks: list[RiskItem]
    hedge_strategy: str
    sources_note: str = ""


class AgentMessage(BaseModel):
    role: Literal["analyst", "writer", "system"]
    content: str
    payload: dict | None = None
