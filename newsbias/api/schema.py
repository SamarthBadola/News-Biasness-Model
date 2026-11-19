"""Request and response schema of the classifier API (version 1).

This is the contract between the platform backend (FastAPI on Render) and the model service. The backend sends
the headline and the full article body it fetched; the service derives the lede itself with the dataset's lede
rule, so the model always reads the same kind of input it was trained and evaluated on. The response echoes the
headline and lede it used, and highlight offsets point into those echoed strings.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

API_VERSION = "1"

PoliticalLabel = Literal["Left", "Center", "Right"]
GenderLabel = Literal["Misogynistic", "Misandrist", "Neutral"]


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=1, max_length=1000, description="The article headline as published.")
    body: str = Field(
        min_length=1, max_length=200_000,
        description="The full article body as extracted by the page fetcher, one paragraph per line "
                    "(paragraphs separated by newlines). Bylines, datelines and boilerplate may be left in: the "
                    "service removes them with the same rule the dataset used.")
    explain: bool = Field(True, description="Compute highlighted phrases for biased labels (never for Center "
                                            "or Neutral). Set false when only labels are needed; it saves "
                                            "several seconds on a CPU host.")
    request_id: str | None = Field(None, max_length=100, description="Echoed back, for logs.")


class Highlight(BaseModel):
    field: Literal["headline", "lede"] = Field(description="Which echoed string the offsets refer to.")
    start: int = Field(ge=0, description="Start character offset in that string.")
    end: int = Field(ge=0, description="End character offset (exclusive).")
    text: str = Field(description="The highlighted phrase, equal to string[start:end].")
    weight: float = Field(ge=0, le=1, description="Share of the positive evidence for the predicted label that "
                                                   "this phrase carries (Integrated Gradients).")


class PoliticalResult(BaseModel):
    label: PoliticalLabel
    confidence: float = Field(ge=0, le=1, description="Calibrated probability of the label.")
    probabilities: dict[PoliticalLabel, float]
    highlights: list[Highlight] = Field(description="Phrases that pushed the model towards the label, strongest "
                                                    "first. Empty for Center, or when explain is false.")


class GenderResult(BaseModel):
    label: GenderLabel
    confidence: float = Field(ge=0, le=1, description="Calibrated probability of the label.")
    probabilities: dict[GenderLabel, float]
    highlights: list[Highlight] = Field(description="Phrases that pushed the model towards the label, strongest "
                                                    "first. Empty for Neutral, or when explain is false.")


class ModelInput(BaseModel):
    headline: str = Field(description="Headline after whitespace normalisation.")
    lede: str = Field(description="The lede the model read: first two body paragraphs after boilerplate and "
                                  "dateline removal, cut to 120 words.")
    lede_words: int
    truncated: bool = Field(description="True if headline plus lede exceeded the model's token limit.")
    scope: Literal["headline+lede"] = Field("headline+lede", description="What the labels describe.")


class ClassifyResponse(BaseModel):
    api_version: str = API_VERSION
    model_version: str
    request_id: str | None = None
    input: ModelInput
    political: PoliticalResult
    gender: GenderResult


class Health(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_version: str | None = None
    api_version: str = API_VERSION


class ErrorResponse(BaseModel):
    detail: str
