from pydantic import BaseModel, Field
from typing import List

class Source(BaseModel):
    title: str
    url: str
    snippet: str = ""

class ResearchReport(BaseModel):
    topic: str
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    review_notes: List[str] = Field(default_factory=list)
    mode: str = "offline"

    def to_markdown(self) -> str:
        source_lines = "\n".join(
            f"- [{s.title}]({s.url}) — {s.snippet}" for s in self.sources
        ) or "- No external sources were retrieved in this run."
        findings = "\n".join(f"{i}. {item}" for i, item in enumerate(self.key_findings, 1))
        limits = "\n".join(f"- {item}" for item in self.limitations)
        review = "\n".join(f"- {item}" for item in self.review_notes)
        return f"""# Research Report: {self.topic}

> Run mode: **{self.mode}**

## Executive summary
{self.executive_summary}

## Key findings
{findings}

## Sources
{source_lines}

## Limitations
{limits}

## Reviewer notes
{review}
"""
