"""Gemini-backed interview logic and response validation."""

import json
import logging
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st
from google import genai
from google.genai import types

from prompts import (
    answer_evaluation_prompt,
    final_report_prompt,
    get_question_category,
    question_generation_prompt,
)


PRIMARY_MODEL = "gemini-3.5-flash-lite"
MAX_TRANSIENT_RETRIES = 2
MAX_QUESTION_RETRIES = 2
QUESTION_HISTORY_PATH = (
    Path(__file__).resolve().parent / "data" / "question_history.json"
)
INTERVIEW_HISTORY_PATH = (
    Path(__file__).resolve().parent / "data" / "interview_history.json"
)

logger = logging.getLogger("intervai.gemini")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass(frozen=True)
class GeminiCallResult:
    """Explicit outcome of a single logical Gemini operation."""

    success: bool
    text: str = ""
    error_type: str = ""
    message: str = ""
    model: str = ""


class GeminiCallError(RuntimeError):
    """Safe, classified Gemini failure for the application layer."""

    def __init__(self, result):
        self.error_type = result.error_type
        self.model = result.model
        super().__init__(result.message)


class InterviewHistoryError(RuntimeError):
    """Safe user-facing failure while accessing completed interview history."""


def _write_json_atomically(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temp_path = Path(temporary_file.name)
            json.dump(value, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def load_interview_history():
    """Load completed interviews, creating an empty history file if needed."""
    path = Path(INTERVIEW_HISTORY_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            _write_json_atomically(path, [])
            return []
        with path.open("r", encoding="utf-8") as history_file:
            history = json.load(history_file)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        logger.error(
            "[IntervAI] Could not load interview history (%s): %s",
            type(error).__name__,
            _safe_error_detail(error),
        )
        raise InterviewHistoryError(
            "Unable to load interview history right now."
        ) from error

    if not isinstance(history, list) or any(
        not isinstance(record, dict)
        or not isinstance(record.get("id"), str)
        or not record["id"]
        for record in history
    ):
        logger.error("[IntervAI] Interview history JSON has an invalid structure.")
        raise InterviewHistoryError(
            "Unable to load interview history right now."
        )
    return history


def save_completed_interview(record):
    """Insert or replace one fully completed interview by its stable ID."""
    if not isinstance(record, dict) or not isinstance(record.get("id"), str):
        raise ValueError("A completed interview record needs a unique ID.")
    history = load_interview_history()
    history = [item for item in history if item.get("id") != record["id"]]
    history.append(record)
    _write_json_atomically(INTERVIEW_HISTORY_PATH, history)


def create_interview_record(
    interview_id,
    role,
    experience,
    interview_type,
    questions,
    answers,
    evaluations,
    final_report,
):
    """Create a serializable record from a fully completed interview."""
    if not (
        isinstance(questions, list)
        and isinstance(answers, list)
        and isinstance(evaluations, list)
        and questions
        and len(questions) == len(answers) == len(evaluations)
        and isinstance(final_report, dict)
    ):
        raise ValueError("Only fully completed interviews can be archived.")
    if not interview_id:
        interview_id = uuid4().hex
    return {
        "id": interview_id,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "role": role,
        "experience": experience,
        "interview_type": interview_type,
        "question_count": len(questions),
        "overall_score": final_report.get("overall_score"),
        "questions": questions,
        "answers": answers,
        "evaluations": evaluations,
        "final_report": final_report,
    }


def delete_interview_record(interview_id):
    """Delete one archived interview by ID."""
    history = load_interview_history()
    remaining = [record for record in history if record.get("id") != interview_id]
    _write_json_atomically(INTERVIEW_HISTORY_PATH, remaining)


def clear_interview_history():
    """Clear completed interviews without touching question history."""
    _write_json_atomically(INTERVIEW_HISTORY_PATH, [])


def normalize_question(question):
    """Normalize a question for conservative case/spacing/punctuation matching."""
    if not isinstance(question, str):
        return ""
    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    normalized = normalized.strip("\"'“”‘’")
    normalized = re.sub(r"[!?.,;:]+$", "", normalized).strip()
    return normalized


def get_history_key(role, experience, interview_type):
    """Build the distinct history bucket for an interview configuration."""
    return f"{role.strip()}|{experience.strip()}|{interview_type.strip()}"


def save_question_history(history):
    """Persist normalized question strings to the local JSON history file."""
    cleaned_history = {}
    if isinstance(history, dict):
        for key, questions in history.items():
            if not isinstance(key, str) or not isinstance(questions, (list, tuple)):
                continue
            normalized_questions = []
            seen = set()
            for question in questions:
                normalized = normalize_question(question)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    normalized_questions.append(normalized)
            if normalized_questions:
                cleaned_history[key] = normalized_questions

    history_path = Path(QUESTION_HISTORY_PATH)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=history_path.parent,
            prefix=f"{history_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temp_path = Path(temporary_file.name)
            json.dump(cleaned_history, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temp_path, history_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def load_question_history():
    """Load and normalize local history, safely recovering missing/bad JSON."""
    history_path = Path(QUESTION_HISTORY_PATH)
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.warning(
            "[IntervAI] Could not access question history directory: %s",
            error,
        )
        return {}
    if not history_path.exists():
        save_question_history({})
        return {}

    try:
        with history_path.open("r", encoding="utf-8") as history_file:
            raw_history = json.load(history_file)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        logger.warning(
            "[IntervAI] Could not load question history; using an empty history: %s",
            error,
        )
        return {}

    if not isinstance(raw_history, dict):
        logger.warning(
            "[IntervAI] Question history has an invalid structure; using an empty history."
        )
        return {}

    normalized_history = {}
    for key, questions in raw_history.items():
        if not isinstance(key, str) or not isinstance(questions, list):
            continue
        normalized_questions = []
        seen = set()
        for question in questions:
            normalized = normalize_question(question)
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_questions.append(normalized)
        if normalized_questions:
            normalized_history[key] = normalized_questions
    return normalized_history


def is_question_repeated(question, previous_questions):
    """Return whether a normalized question appears in supplied history."""
    normalized = normalize_question(question)
    if not normalized:
        return False
    if not isinstance(previous_questions, (list, tuple, set)):
        return False
    return any(
        normalized == normalize_question(previous)
        for previous in previous_questions
        if isinstance(previous, str)
    )


def add_questions_to_history(role, experience, interview_type, questions):
    """Add successfully used questions to the selected history bucket."""
    history = load_question_history()
    history_key = get_history_key(role, experience, interview_type)
    existing = history.get(history_key, [])
    seen = {normalize_question(question) for question in existing}
    for question in questions if isinstance(questions, (list, tuple)) else []:
        normalized = normalize_question(question)
        if normalized and normalized not in seen:
            existing.append(normalized)
            seen.add(normalized)
    if existing:
        history[history_key] = existing
    save_question_history(history)


def clear_question_history():
    """Clear local question history without affecting the active interview."""
    save_question_history({})


def get_client():
    """Create a Gemini client using the key in Streamlit secrets."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception as error:
        raise GeminiCallError(
            GeminiCallResult(
                success=False,
                error_type="authentication_error",
                message=(
                    "GEMINI_API_KEY is not configured. Add it to "
                    ".streamlit/secrets.toml."
                ),
            )
        ) from error

    if not isinstance(api_key, str) or not api_key.strip():
        raise GeminiCallError(
            GeminiCallResult(
                success=False,
                error_type="authentication_error",
                message="GEMINI_API_KEY is empty.",
            )
        )

    try:
        return genai.Client(api_key=api_key.strip())
    except Exception as error:
        raise GeminiCallError(
            GeminiCallResult(
                success=False,
                error_type=_classify_api_error(error),
                message=_safe_error_detail(error),
            )
        ) from error


def _classify_api_error(error):
    message = str(error).lower()
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    status = str(status or "").lower()

    daily_quota_terms = (
        "generate_requests_per_day",
        "requests_per_day",
        "perday",
        "per_day",
        "daily quota",
        "quota will reset",
    )
    if any(term in message for term in daily_quota_terms):
        return "quota_exhausted"
    if (
        "quota exceeded" in message
        or "quota_exhausted" in message
        or "resource_exhausted" in message
    ) and ("429" in message or "429" in status or "rate limit" in message):
        return "quota_exhausted"
    if (
        "401" in message
        or "403" in message
        or status in ("401", "403", "unauthenticated", "permission_denied")
        or "unauthenticated" in message
        or "permission_denied" in message
        or "api key not valid" in message
        or "invalid api key" in message
    ):
        return "authentication_error"
    if (
        "model not found" in message
        or "model_not_found" in message
        or "unknown model" in message
        or "is not found for api version" in message
        or ("404" in message and "model" in message)
        or "model is unavailable" in message
    ):
        return "model_unavailable"
    if (
        "400" in message
        or status == "400"
        or "invalid_argument" in message
        or "bad request" in message
    ):
        return "invalid_request"
    if any(
        term in message or term in status
        for term in (
            "503",
            "unavailable",
            "deadline_exceeded",
            "deadline exceeded",
            "timeout",
            "temporarily",
            "502",
            "500",
        )
    ):
        return "temporary_error"
    if "429" in message or "429" in status:
        return "temporary_error"
    return "unknown_error"


def _safe_error_detail(error):
    message = str(error)
    message = re.sub(
        r"(?i)(api[_ -]?key\s*[=:]\s*)[^\s&,;]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"\bAIza[0-9A-Za-z_-]{20,}\b",
        "[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)(authorization\s*:\s*(?:bearer|key)\s+)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    return message[:1000]


def _call_model(client, model, prompt, config):
    """Call one model with retries only for temporary service errors."""
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            logger.info("[Gemini] Trying model: %s", model)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                return GeminiCallResult(
                    success=False,
                    error_type="unknown_error",
                    message="Gemini returned an empty response.",
                    model=model,
                )
            return GeminiCallResult(
                success=True,
                text=text,
                model=model,
            )
        except GeminiCallError as error:
            return GeminiCallResult(
                success=False,
                error_type=error.error_type,
                message=str(error),
                model=model,
            )
        except Exception as error:
            error_type = _classify_api_error(error)
            logger.warning(
                "[Gemini] %s failed (%s): %s",
                model,
                error_type,
                _safe_error_detail(error),
            )
            if (
                error_type == "temporary_error"
                and attempt < MAX_TRANSIENT_RETRIES
            ):
                time.sleep(2**attempt)
                continue
            return GeminiCallResult(
                success=False,
                error_type=error_type,
                message=_safe_error_detail(error),
                model=model,
            )

    return GeminiCallResult(
        success=False,
        error_type="temporary_error",
        message="Gemini is temporarily unavailable.",
        model=model,
    )


def call_gemini(prompt, json_mode=False):
    """Call the configured Gemini model with bounded transient retries."""
    try:
        client = get_client()
    except GeminiCallError as error:
        logger.error("[Gemini] Client setup failed: %s", error.error_type)
        return GeminiCallResult(
            success=False,
            error_type=error.error_type,
            message=str(error),
        )

    config = (
        types.GenerateContentConfig(response_mime_type="application/json")
        if json_mode
        else None
    )
    primary = _call_model(client, PRIMARY_MODEL, prompt, config)
    return primary


def _response_text(result):
    if result.success:
        return result.text
    raise GeminiCallError(result)


def _json_object(response_text):
    """Parse a JSON object, including one surrounded by markdown fences."""
    if not response_text:
        return None

    candidate = response_text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        result = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            result = json.loads(candidate[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return None
    return result if isinstance(result, dict) else None


def _score(value, default=None):
    """Convert a model-provided score to a bounded integer or None."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number) or number < 0 or number > 10:
        return default
    return int(round(number))


def _text_list(value, fallback):
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    result = [str(item).strip() for item in items if str(item).strip()]
    return result or list(fallback)


def _normalize_evaluation(payload):
    payload = payload or {}
    score = _score(payload.get("score"))
    dimensions = {
        key: _score(payload.get(key), default=score)
        for key in (
            "technical_accuracy",
            "completeness",
            "clarity",
            "relevance",
        )
    }
    ideal_answer = payload.get("ideal_answer")
    if not isinstance(ideal_answer, str) or not ideal_answer.strip():
        ideal_answer = (
            "A strong answer should address the question directly, explain "
            "the reasoning clearly, and include a relevant example."
        )

    return {
        "score": score,
        **dimensions,
        "strengths": _text_list(
            payload.get("strengths"),
            ["No specific strength could be identified from this response."],
        ),
        "improvements": _text_list(
            payload.get("improvements"),
            ["Add more specific detail and explain your reasoning clearly."],
        ),
        "ideal_answer": ideal_answer.strip(),
    }


def generate_question(
    role,
    experience,
    interview_type,
    question_number,
    total_questions,
    previous_questions=None,
    previous_evaluation=None,
):
    """Generate one role-aware, adaptive question without exact repeats."""
    if isinstance(previous_questions, (list, tuple)):
        previous_questions = [
            question.strip()
            for question in previous_questions
            if isinstance(question, str) and question.strip()
        ]
    else:
        previous_questions = []

    history = load_question_history()
    history_key = get_history_key(role, experience, interview_type)
    history_questions = history.get(history_key, [])
    excluded_questions = list(
        dict.fromkeys(
            [
                *history_questions,
                *(
                    normalize_question(question)
                    for question in previous_questions
                ),
            ]
        )
    )

    for attempt in range(MAX_QUESTION_RETRIES):
        prompt = question_generation_prompt(
            role=role,
            experience=experience,
            interview_type=interview_type,
            question_number=question_number,
            total_questions=total_questions,
            previous_questions=previous_questions,
            previous_evaluation=previous_evaluation,
            excluded_questions=excluded_questions,
        )

        response_text = _response_text(call_gemini(prompt))
        question = re.sub(
            r"^(?:question\s*\d*\s*:\s*)",
            "",
            response_text.strip(),
            flags=re.IGNORECASE,
        ).strip().strip('"')
        if not question:
            continue
        if not is_question_repeated(question, excluded_questions):
            return question
        excluded_questions.append(question)

    raise GeminiCallError(
        GeminiCallResult(
            success=False,
            error_type="question_not_unique",
            message=(
                "Could not generate a new unique question right now. "
                "Please try again."
            ),
        )
    )


def evaluate_answer(
    role,
    experience,
    interview_type,
    question,
    answer,
    question_category=None,
):
    """Evaluate an answer and normalize malformed or incomplete model JSON."""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Please provide an answer before submitting.")

    prompt = answer_evaluation_prompt(
        role=role,
        experience=experience,
        interview_type=interview_type,
        question=question,
        answer=answer.strip(),
        question_category=question_category,
    )
    response_text = _response_text(call_gemini(prompt, json_mode=True))
    evaluation = _normalize_evaluation(_json_object(response_text))
    evaluation["question_category"] = question_category or interview_type
    return evaluation


def _valid_evaluation_score(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(score) or not 0 <= score <= 10:
        return None
    return score


def calculate_overall_score(evaluations):
    """Calculate a percentage from valid saved evaluation scores only."""
    if not isinstance(evaluations, (list, tuple)):
        return None

    scores = [
        score
        for evaluation in evaluations
        if isinstance(evaluation, dict)
        for score in [_valid_evaluation_score(evaluation.get("score"))]
        if score is not None
    ]
    if not scores:
        return None
    return round((sum(scores) / len(scores)) * 10)


def _average_evaluation_field(evaluations, field):
    values = [
        value
        for evaluation in evaluations
        if isinstance(evaluation, dict)
        for value in [_valid_evaluation_score(evaluation.get(field))]
        if value is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values))


def _evaluation_texts(evaluation, field):
    value = evaluation.get(field) if isinstance(evaluation, dict) else None
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = []
    return [
        item.strip()
        for item in values
        if isinstance(item, str) and item.strip()
    ]


def _unique_texts(items):
    return list(dict.fromkeys(item for item in items if item))[:6]


def _local_final_report(
    interview_type,
    questions,
    answers,
    evaluations,
):
    """Build a complete scorecard and question analysis from saved state."""
    question_analysis = []
    strengths = []
    weaknesses = []

    for index, (question, answer, evaluation) in enumerate(
        zip(questions, answers, evaluations),
        start=1,
    ):
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        question_strengths = _evaluation_texts(evaluation, "strengths")
        question_weaknesses = _evaluation_texts(evaluation, "improvements")
        strengths.extend(question_strengths)
        weaknesses.extend(question_weaknesses)
        question_analysis.append(
            {
                "question_number": index,
                "category": evaluation.get(
                    "question_category",
                    get_question_category(interview_type, index),
                ),
                "question": question,
                "answer": answer,
                "score": _valid_evaluation_score(evaluation.get("score")),
                "strengths": question_strengths,
                "weaknesses": question_weaknesses,
            }
        )

    overall_score = calculate_overall_score(evaluations)
    valid_score_count = sum(
        _valid_evaluation_score(evaluation.get("score")) is not None
        for evaluation in evaluations
        if isinstance(evaluation, dict)
    )
    if overall_score is None:
        performance_summary = (
            f"The interview covered {len(questions)} question(s). "
            "An overall score could not be calculated because no valid "
            "evaluation scores were available."
        )
    else:
        performance_summary = (
            f"The interview covered {len(questions)} question(s). "
            f"Your overall score is {overall_score}/100, calculated locally "
            f"from {valid_score_count} valid saved evaluation score(s)."
        )

    strengths = _unique_texts(strengths)
    weaknesses = _unique_texts(weaknesses)
    recommendations = [
        f"Practice improving: {weakness.rstrip('.')}."
        for weakness in weaknesses[:5]
    ]
    if not recommendations:
        recommendations = [
            "Review each question and answer, and continue practicing "
            "role-related interview responses."
        ]

    return {
        "overall_score": overall_score,
        "technical_knowledge": _average_evaluation_field(
            evaluations, "technical_accuracy"
        ),
        "completeness": _average_evaluation_field(evaluations, "completeness"),
        "clarity": _average_evaluation_field(evaluations, "clarity"),
        "relevance": _average_evaluation_field(evaluations, "relevance"),
        "performance_summary": performance_summary,
        "interview_summary": performance_summary,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "areas_to_improve": weaknesses,
        "recommendations": recommendations,
        "question_analysis": question_analysis,
        "final_tip": recommendations[0],
        "qualitative_analysis_source": "saved evaluations",
        "analysis_notice": "",
    }


def _qualitative_list(payload, field, fallback):
    value = payload.get(field) if isinstance(payload, dict) else None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return fallback
    cleaned = _unique_texts(
        [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
    )
    return cleaned or fallback


def _merge_qualitative_analysis(report, payload):
    """Merge usable qualitative fields; keep local values for invalid fields."""
    if not isinstance(payload, dict):
        report["analysis_notice"] = (
            "AI qualitative analysis was unavailable; this report uses your "
            "saved answer evaluations."
        )
        return report

    summary = payload.get("performance_summary")
    valid_summary = isinstance(summary, str) and bool(summary.strip())
    valid_lists = any(
        isinstance(payload.get(field), list)
        and any(
            isinstance(item, str) and item.strip()
            for item in payload[field]
        )
        for field in ("strengths", "weaknesses", "recommendations")
    )
    if not valid_summary and not valid_lists:
        report["analysis_notice"] = (
            "AI qualitative analysis was unavailable; this report uses your "
            "saved answer evaluations."
        )
        return report

    if valid_summary:
        report["performance_summary"] = summary.strip()
        report["interview_summary"] = summary.strip()

    report["strengths"] = _qualitative_list(
        payload, "strengths", report["strengths"]
    )
    report["weaknesses"] = _qualitative_list(
        payload, "weaknesses", report["weaknesses"]
    )
    report["areas_to_improve"] = report["weaknesses"]
    report["recommendations"] = _qualitative_list(
        payload, "recommendations", report["recommendations"]
    )
    if report["recommendations"]:
        report["final_tip"] = report["recommendations"][0]
    report["qualitative_analysis_source"] = "Gemini"
    return report


def generate_final_report(
    role,
    experience,
    interview_type,
    questions,
    answers,
    evaluations,
):
    """Build scores locally and add Gemini qualitative analysis when available."""
    if not questions or not answers or not evaluations:
        raise ValueError("Complete at least one evaluated question first.")
    if not (len(questions) == len(answers) == len(evaluations)):
        raise ValueError(
            "Interview data is incomplete. Questions, answers, and evaluations "
            "must have the same length."
        )

    report = _local_final_report(
        interview_type,
        questions,
        answers,
        evaluations,
    )
    prompt = final_report_prompt(
        role=role,
        experience=experience,
        interview_type=interview_type,
        questions=questions,
        answers=answers,
        evaluations=evaluations,
    )
    try:
        response_text = _response_text(call_gemini(prompt, json_mode=True))
    except Exception as error:
        logger.warning(
            "[Gemini] Qualitative final-report analysis unavailable; "
            "using saved evaluations: %s",
            _safe_error_detail(error),
        )
        report["analysis_notice"] = (
            "AI qualitative analysis was unavailable; this report uses your "
            "saved answer evaluations."
        )
        return report

    return _merge_qualitative_analysis(
        report,
        _json_object(response_text),
    )
