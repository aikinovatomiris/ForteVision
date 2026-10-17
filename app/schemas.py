from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceFrame(BaseModel):
    timestamp_sec: float = Field(ge=0, allow_inf_nan=False)
    image_base64: str = Field(min_length=4)


class Violation(BaseModel):
    type: str
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


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
    # Optional additions; the jury's required fields remain unchanged.
    elapsed_seconds: float | None = None
    within_five_minutes: bool | None = None

    @model_validator(mode='after')
    def consistent_verdict(self):
        if bool(self.violations) != (self.verdict == 'violation'):
            raise ValueError('verdict и violations не согласованы')
        return self


JobResponse = Annotated[ProcessingJob | DoneJob | FailedJob, Field(discriminator='status')]
