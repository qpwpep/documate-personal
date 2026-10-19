from __future__ import annotations

import hashlib
import json
import keyword
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.request_contracts import WireRequestContract

PlannerRouteName = Literal["docs", "upload"]
PlannedDocsLibrary = Literal[
    "Python", "Git", "LangChain", "matplotlib", "numpy", "pandas", "pytorch",
    "Hugging Face", "fastapi", "BeautifulSoup", "streamlit", "gradio",
    "scikit-learn", "pydantic", "Bare", "unsupported",
]
DocsEvidenceNeed = Literal["signature", "parameters", "examples", "explanation"]
PLANNER_ROUTES: tuple[PlannerRouteName, ...] = ("docs", "upload")
# Historical benchmark diagnostics still use this value; new plans preserve tasks.
PLANNER_WARNING_DUPLICATE_ROUTE_MERGED = "duplicate_route_merged"


class RetrievalRequirement(BaseModel):
    """What evidence must establish, independently of a provider search string."""

    model_config = ConfigDict(extra="forbid")
    library: str | None = Field(default=None, description="The one library owning this official source; null for upload or an unknown library.")
    symbols: list[str] = Field(default_factory=list, description="Exact API/code symbols to find, preferably fully qualified official names.")
    version: str | None = Field(default=None, description="Explicit requested version; null when unconstrained. Never substitute another version.")
    aspects: list[str] = Field(default_factory=list, description="Literal parameter/code anchors explicitly requested in the user's dialogue. For docs setting-meaning questions, keep the parameter name but omit ordinary data and arbitrary counts, capacities, sizes, thresholds or seeds; keep the complete setting in query and the body request. A value changing the result is not sufficient to make it a literal anchor. Retain categorical modes, enum members, sentinels and distinct behavior-branch values alongside their parameter names, including numeric and Boolean choices. Preserve exact source/code matching requests. Natural-language properties and questions belong in query, for upload as well as docs. Do not guess the identifier or value answering such a question. General documentation needs belong in evidence_needs, not aspects.")
    evidence_needs: list[DocsEvidenceNeed] = Field(default_factory=list, description="Docs-only evidence kinds requested by the user: signature for call syntax, parameters for option/argument descriptions, examples for usage examples, explanation for conceptual or behavioral explanation. Explaining a named parameter setting requests BOTH parameters and explanation, even when phrased as its behavior or effect; that value needs the parameter's own description. Keep empty for upload. Do not infer unrelated kinds from model knowledge.")
    file_ids: list[str] = Field(default_factory=list, description="Upload file IDs from the active catalog whose evidence is explicitly requested. Empty searches all active uploads without requiring every file in the answer. For a comparison or inspection of all attachments, include every requested file ID in one task. Empty for docs.")
    match: Literal["topic", "symbol", "definition"] = Field(default="topic", description="topic for broad explanation; symbol for API/usage references, including an uploaded function's implementation together with its actual calls/examples; definition when only the named uploaded function/class implementation is needed. A definition alone excludes calls outside its body.")

    @field_validator("symbols", "aspects")
    @classmethod
    def clean_terms(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @field_validator("evidence_needs")
    @classmethod
    def clean_evidence_needs(cls, values: list[DocsEvidenceNeed]) -> list[DocsEvidenceNeed]:
        # Evidence kinds are a set of requirements, not a preference order.
        return sorted(set(values))

    @field_validator("file_ids")
    @classmethod
    def clean_file_ids(cls, values: list[str]) -> list[str]:
        return sorted({value.strip() for value in values if value.strip()})

    @field_validator("library", "version")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return (value.strip() or None) if value is not None else None

    @property
    def specified(self) -> bool:
        return bool(self.library or self.symbols or self.version or self.aspects or self.evidence_needs or self.file_ids or self.match != "topic")


class PlannedRequirement(RetrievalRequirement):
    """An independently planned subject has one match mode, never mixed symbol roles."""

    library: PlannedDocsLibrary | None = Field(
        default=None,
        description="Exact canonical owning-library ID for official docs. Use unsupported for an explicitly requested library outside the enum; preserve its actual name in query/symbols. Python owns standard-library modules such as pathlib. Never append module names, versions or prose to this ID. Null for upload or an unscoped docs subject.",
    )
    version: str | None = Field(
        default=None, pattern=r"^\S+$",
        description="One exact requested version token, for example 3.12, 1.26, v2.4 or latest. For 'Python 3.12', return only '3.12'. Never include a library name, spaces, prose or an inferred default version. Null when the user did not constrain the version.",
    )
    aspects: list[str] = Field(
        default_factory=list,
        description="Literal parameter names, option/flag names, requested value tokens or code fragments that must appear in returned source, such as axis, --force, reflect or random_state. For docs setting-meaning questions, omit ordinary data and arbitrary counts, capacities, sizes, thresholds or seeds; keep their parameter names here and complete settings in query and the body request. A value changing the result is not sufficient to make it a literal anchor. Keep categorical modes, enum members, sentinels and distinct behavior-branch values separately alongside their parameter names, including numeric or Boolean choices. Preserve an explicitly requested exact source/code match. Do not repeat the primary API already in symbols. For upload and docs, never put natural-language properties or conditions here; preserve their meaning in query. For docs additionally request the relevant evidence_needs. Only anchors explicitly requested in the user's dialogue belong here; never guess an identifier from the user's description of a property.",
    )
    symbols: list[str] = Field(default_factory=list, max_length=1,
                              description="At most one primary target. Use separate tasks for independent symbols; calls inspected inside a function belong in its aspects, not additional definitions.")

    @model_validator(mode="after")
    def validate_named_target(self) -> "PlannedRequirement":
        if self.match != "topic" and not self.symbols:
            raise ValueError("symbol and definition match require a nonempty primary symbol")
        return self


class RetrievalQuery(BaseModel):
    """Validated search text and result budget shared by plans and rewrites."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=1)
    k: int = Field(..., ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


class RetrievalTask(RetrievalQuery):
    route: PlannerRouteName
    requirement: PlannedRequirement = Field(default_factory=PlannedRequirement)
    requirement_id: str = Field(default="", description="A stable short ID for this independent requirement; preserve it during retries.")

    @model_validator(mode="after")
    def assign_requirement_identity(self) -> "RetrievalTask":
        if self.route != "upload" and self.requirement.file_ids:
            raise ValueError("file_ids apply only to uploaded-file retrieval")
        if self.route != "docs" and self.requirement.evidence_needs:
            raise ValueError("evidence_needs apply only to official-doc retrieval")
        if self.route == "upload" and self.requirement.match != "topic":
            for aspect in self.requirement.aspects:
                # Reject only single-line runs of ordinary names. Keywords (including
                # soft keywords) can form incomplete source fragments such as
                # "is None" or "async def" that cannot parse as a module.
                # Actual occurrence is checked against uploaded source later.
                words = aspect.split()
                if len(words) > 1 and not any(newline in aspect for newline in ("\r", "\n")) and all(
                    word.isidentifier() and not keyword.iskeyword(word) and not keyword.issoftkeyword(word)
                    for word in words
                ):
                    raise ValueError(
                        f"Upload aspect {aspect!r} is a natural-language condition, not a literal code anchor. "
                        "Preserve the question in query and remove it from aspects; never guess its code name."
                    )
        if not self.requirement_id.strip():
            identity = {"route": self.route, "requirement": self.requirement.model_dump()}
            if not self.requirement.specified:
                identity["query"] = self.query
            self.requirement_id = "req_" + hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        else:
            self.requirement_id = self.requirement_id.strip()
        return self


class RetrievalPlanOutput(BaseModel):
    """Initial model-owned evidence requirements for independently planned subjects."""

    model_config = ConfigDict(extra="forbid")
    use_retrieval: bool = Field(description="Whether a sufficiently resolved request needs retrieved evidence.")
    tasks: list[RetrievalTask] = Field(description="One task per independent source/subject/version requirement. Multiple tasks may share a route. Include upload even if the file is missing; omit excluded sources.")

    @model_validator(mode="after")
    def validate_rules(self) -> "RetrievalPlanOutput":
        if not self.use_retrieval and self.tasks:
            raise ValueError("tasks must be empty when use_retrieval is false")
        if self.use_retrieval and not self.tasks:
            raise ValueError("tasks must contain at least one requirement when use_retrieval is true")
        identities = [task.requirement_id for task in self.tasks]
        if len(set(identities)) != len(identities):
            raise ValueError("each independent requirement must have a unique ID")
        return self


class RetrievalRewrite(RetrievalQuery):
    """The only model-owned changes allowed during retrieval recovery."""

    requirement_id: str = Field(min_length=1, description="An exact failed requirement ID supplied by the server.")


class RetrievalRewriteOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rewrites: list[RetrievalRewrite] = Field(description="Exactly one query/k rewrite for every failed requirement ID. Omit successful requirements.")

    @model_validator(mode="after")
    def unique_requirement_ids(self) -> "RetrievalRewriteOutput":
        identities = [rewrite.requirement_id for rewrite in self.rewrites]
        if len(set(identities)) != len(identities):
            raise ValueError("each failed requirement must have exactly one rewrite")
        return self


class PlannerOutput(RetrievalPlanOutput):
    """Internal plan with the server-preserved interpretation available to later nodes."""

    request_contract: WireRequestContract | None = None

    @classmethod
    def fallback(cls, request_contract: WireRequestContract | None = None) -> "PlannerOutput":
        return cls(use_retrieval=False, tasks=[], request_contract=request_contract)


class AspectReference(BaseModel):
    """A current user reference selecting one literal anchor from the previous answer."""

    model_config = ConfigDict(extra="forbid")
    requirement_id: str = Field(min_length=1)
    aspect: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1, description="Local request_contract evidence with interpretation=reference, current turn_id, and scope covering retrieval.<requirement_id>.aspects: that path, a parent, or current_request.")
    answer_quote: str = Field(min_length=1, description="Exact, minimal selected item from previous_answer_for_aspects containing this aspect. Never quote the whole answer or unselected items.")


class InitialPlannerOutput(PlannerOutput):
    """Initial model output must include the request facts before any plan can execute."""

    request_contract: WireRequestContract = Field(description="Interpret the current user request once, with locally defined evidence and offered body references. Do not invent server identities, hashes or readiness.")

    aspect_references: list[AspectReference] = Field(default_factory=list, description="Required for EACH aspect absent from user_turn_ledger but selected from previous_answer_for_aspects by the current user. Never leave empty for a selected numbered/referred parameter. Cite current user evidence with interpretation=reference and scope covering retrieval.<requirement_id>.aspects (that path, a parent, or current_request), plus the exact selected answer item. Empty only for direct user anchors or no selected indirect aspects.")
