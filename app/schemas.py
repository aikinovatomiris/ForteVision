import base64
import io
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator, field_validator


class EvidenceFrame(BaseModel):
    timestamp_sec: float = Field(ge=0, allow_inf_nan=False)
    image_base64: str = Field(min_length=4, max_length=8*1024*1024)

    @field_validator('image_base64')
    @classmethod
    def valid_jpeg(cls, value):
        from PIL import Image
        try:
            raw = base64.b64decode(value, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'JPEG' or max(image.size) > 4096:
                    raise ValueError('Требуется JPEG размером до 4096 px')
                image.verify()
        except Exception as exc:
            raise ValueError('Evidence должен содержать корректный JPEG в base64') from exc
        return value


class RuleReference(BaseModel):
    source_title: str = Field(min_length=1)
    source_url: str
    section: str = Field(min_length=1)
    verified_at: str
    applicability: str | None = None
    reference_role: str = "validated_binding"

    @field_validator('source_url')
    @classmethod
    def official_source(cls, value):
        parsed = urlparse(value)
        if parsed.scheme != 'https' or parsed.hostname not in ('adilet.zan.kz', 'old.adilet.zan.kz', 'nationalbank.kz', 'www.nationalbank.kz'):
            raise ValueError('Нужен официальный HTTPS источник РК')
        return value


class Violation(BaseModel):
    type: str
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)
    context_start_sec: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    context_end_sec: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    event_type: str | None = None
    rule_references: list[RuleReference] | None = None


class ReviewCandidate(Violation):
    review_reason: str
    review_priority: Literal['high', 'normal'] = 'normal'


class JobAccepted(BaseModel):
    job_id: str


class ProcessingJob(BaseModel):
    job_id: str
    status: Literal['processing']


class FailedJob(BaseModel):
    job_id: str
    status: Literal['failed']
    error: str


class DoneJob(BaseModel):
    job_id: str
    status: Literal['done']
    verdict: Literal['violation', 'no_violation']
    violations: list[Violation]
    review_candidates: list[ReviewCandidate] = Field(default_factory=list)
    review_required: bool = False
    assessment_status: Literal['needs_review', 'incomplete'] | None = None
    candidate_count: int | None = Field(default=None, ge=0)
    workflow_checks: list[dict] | None = None
    observations: list[dict] | None = None
    analysis_limitations: list[str] | None = None
    analysis_complete: bool | None = None
    analysis_backend: str | None = None
    sampled_frames: int | None = None
    # Optional additions; the jury's required fields remain unchanged.
    elapsed_seconds: float | None = None
    within_five_minutes: bool | None = None
    queue_seconds: float | None = None
    processing_seconds: float | None = None

    @model_validator(mode='after')
    def consistent_verdict(self):
        if bool(self.violations) != (self.verdict == 'violation'):
            raise ValueError('verdict и violations не согласованы')
        if self.review_candidates and (not self.review_required or self.analysis_complete is not False or self.assessment_status != 'needs_review'):
            raise ValueError('Кандидаты проверки требуют явного needs_review и неполного анализа')
        return self


JobResponse = Annotated[ProcessingJob | DoneJob | FailedJob, Field(discriminator='status')]
