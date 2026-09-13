"""Slide edit agent 的请求 / 结果数据模型。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from .html_safety import SlideEditValidationResult

AgentEditMode = Literal["slide", "element"]
RunStatus = Literal["completed", "max_iterations", "cancelled", "failed"]

AGENT_MIN_ITERATIONS = 2
AGENT_MAX_ITERATIONS = 100
AGENT_DEFAULT_ITERATIONS = 12


class SlideEditAgentRequest(BaseModel):
    projectId: str
    slideIndex: int
    userRequest: str
    chatHistory: Optional[List[Dict[str, Any]]] = None
    mode: AgentEditMode = "slide"
    slideTitle: Optional[str] = None
    slideContent: Optional[str] = None
    slideOutline: Optional[Dict[str, Any]] = None
    projectInfo: Optional[Dict[str, Any]] = None
    selectedElementHtml: Optional[str] = None
    selectedElementId: Optional[str] = None
    slideScreenshot: Optional[str] = None
    elementScreenshot: Optional[str] = None
    images: Optional[List[Dict[str, Any]]] = None
    visionEnabled: bool = False
    maxIterations: Optional[int] = None
    runId: Optional[str] = None
    conversation_id: Optional[str] = None


class SlideEditAgentApplyRequest(BaseModel):
    proposalId: str
    projectId: str
    slideIndex: int
    expectedBaseHash: str
    htmlContent: str
    slideData: Optional[Dict[str, Any]] = None


class SlideEditAgentCancelRequest(BaseModel):
    runId: str


def coerce_agent_max_iterations(raw_value: Any) -> int:
    try:
        if raw_value is not None:
            return max(
                AGENT_MIN_ITERATIONS,
                min(AGENT_MAX_ITERATIONS, int(raw_value)),
            )
    except (TypeError, ValueError):
        pass
    return AGENT_DEFAULT_ITERATIONS


@dataclass
class SlideEditProposal:
    proposal_id: str
    base_hash: str
    summary: str
    changed_slide_indices: List[int]
    html_content: str
    validation: SlideEditValidationResult
    tool_transcript: List[Dict[str, Any]]
    slide_data: Dict[str, Any]
    created_at: float
    revision: int = 0
    changed: bool = False
    diff: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "proposalId": self.proposal_id,
            "baseHash": self.base_hash,
            "summary": self.summary,
            "changedSlideIndices": self.changed_slide_indices,
            "htmlContent": self.html_content,
            "revision": self.revision,
            "changed": self.changed,
            "diff": self.diff,
            "validation": {
                "valid": self.validation.valid,
                "errors": self.validation.errors,
                "warnings": self.validation.warnings,
            },
            "toolTranscript": self.tool_transcript,
            "slideData": self.slide_data,
            "createdAt": self.created_at,
        }


@dataclass
class SlideEditRunResult:
    run_id: str
    status: RunStatus
    summary: str
    proposal: Optional[SlideEditProposal] = None
    iterations_used: int = 0
    error: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "runId": self.run_id,
            "status": self.status,
            "summary": self.summary,
            "iterationsUsed": self.iterations_used,
        }
        if self.proposal is not None:
            payload["proposal"] = self.proposal.to_public_dict()
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class SlideEditAgentContext:
    """一次 run 的不变输入。草稿状态不放这里，见 SlideDraft。"""

    request: SlideEditAgentRequest
    run_id: str
    project_id: str
    slide_index: int
    mode: AgentEditMode
    base_html: str
    slide_data: Dict[str, Any]
    project_info: Dict[str, Any] = field(default_factory=dict)
    slide_outline: Dict[str, Any] = field(default_factory=dict)
    selected_element_id: Optional[str] = None
    selected_element_html: Optional[str] = None

    @classmethod
    def from_request(cls, request: SlideEditAgentRequest) -> "SlideEditAgentContext":
        base_html = request.slideContent or ""
        outline = request.slideOutline or {}
        slide_data = {
            "page_number": request.slideIndex,
            "title": request.slideTitle
            or outline.get("title")
            or f"Slide {request.slideIndex}",
            "html_content": base_html,
            "slide_type": outline.get("slide_type") or outline.get("type") or "content",
            "content_points": outline.get("content_points") or [],
            "metadata": {},
            "is_user_edited": True,
        }
        return cls(
            request=request,
            run_id=(request.runId or "").strip() or new_run_id(),
            project_id=request.projectId,
            slide_index=request.slideIndex,
            mode=request.mode,
            base_html=base_html,
            slide_data=slide_data,
            project_info=request.projectInfo or {},
            slide_outline=outline,
            selected_element_id=request.selectedElementId,
            selected_element_html=request.selectedElementHtml,
        )


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex}"


def new_proposal_id() -> str:
    return f"slide-edit-{uuid.uuid4().hex}"


def now() -> float:
    return time.time()
