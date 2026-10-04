"""Prompt construction for IntervAI's Gemini requests."""

import json
import math


FULL_INTERVIEW_SCHEDULE = (
    "HR",
    "Technical",
    "Coding",
    "Technical",
    "HR",
    "Coding",
)


def _normalize_previous_questions(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [
        question.strip()
        for question in value
        if isinstance(question, str) and question.strip()
    ]


def _normalize_previous_evaluation(value):
    if not isinstance(value, dict):
        value = {}

    raw_score = value.get("score")
    if isinstance(raw_score, bool):
        score = None
    else:
        try:
            numeric_score = float(raw_score)
        except (TypeError, ValueError, OverflowError):
            score = None
        else:
            score = (
                int(numeric_score)
                if math.isfinite(numeric_score)
                and numeric_score.is_integer()
                and 0 <= numeric_score <= 10
                else None
            )

    def clean_items(items):
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, (list, tuple)):
            return []
        return [
            item.strip()
            for item in items
            if isinstance(item, str) and item.strip()
        ]

    answer = value.get("previous_answer")
    return {
        "score": score,
        "strengths": clean_items(value.get("strengths")),
        "improvements": clean_items(value.get("improvements")),
        "previous_answer": (
            answer.strip()[:3000]
            if isinstance(answer, str) and answer.strip()
            else ""
        ),
    }


def get_question_category(interview_type, question_number):
    """Return the category assigned to a question in a Full Interview."""
    if interview_type != "Full Interview":
        return interview_type
    if question_number <= len(FULL_INTERVIEW_SCHEDULE):
        return FULL_INTERVIEW_SCHEDULE[question_number - 1]
    cycle = ("HR", "Technical", "Coding")
    return cycle[(question_number - len(FULL_INTERVIEW_SCHEDULE) - 1) % len(cycle)]


def question_generation_prompt(
    role,
    experience,
    interview_type,
    question_number,
    total_questions,
    previous_questions=None,
    previous_evaluation=None,
    excluded_questions=None,
):
    """Create a role-aware, adaptive prompt for one interview question."""
    previous_questions = _normalize_previous_questions(previous_questions)
    excluded_questions = _normalize_previous_questions(
        excluded_questions if excluded_questions is not None else previous_questions
    )
    previous_evaluation = _normalize_previous_evaluation(previous_evaluation)
    category = get_question_category(interview_type, question_number)

    previous_questions_text = (
        "\n".join(
            f"{index}. {question}"
            for index, question in enumerate(previous_questions, start=1)
        )
        if previous_questions
        else "No previous questions."
    )
    excluded_questions_text = (
        "\n".join(
            f"{index}. {question}"
            for index, question in enumerate(excluded_questions, start=1)
        )
        if excluded_questions
        else "No excluded questions."
    )

    if any(
        (
            previous_evaluation["score"] is not None,
            previous_evaluation["strengths"],
            previous_evaluation["improvements"],
            previous_evaluation["previous_answer"],
        )
    ):
        score_text = (
            f'{previous_evaluation["score"]}/10'
            if previous_evaluation["score"] is not None
            else "Unavailable"
        )
        strengths_text = (
            "; ".join(previous_evaluation["strengths"])
            or "Not provided"
        )
        improvements_text = (
            "; ".join(previous_evaluation["improvements"])
            or "Not provided"
        )
        answer_text = (
            previous_evaluation["previous_answer"] or "Not provided"
        )
        adaptive_context = f"""
Previous score: {score_text}
Strengths: {strengths_text}
Areas to improve: {improvements_text}
Previous answer: {answer_text}

Adapt the difficulty based on the previous overall score:
- 8–10: increase difficulty with a deeper, practical, or advanced question.
- 6–7: keep approximately the same difficulty and explore another concept.
- 0–5: reinforce fundamentals with a clearer, simpler question targeting an area to improve.
"""
    else:
        adaptive_context = "This is the first question; use the candidate's experience level to set the difficulty."

    category_guidance = {
        "Technical": (
            "Ask one role-specific technical concept, architecture, debugging, "
            "or practical knowledge question. Avoid generic questions when a "
            "specific technology or responsibility fits the role."
        ),
        "HR": (
            "Ask one behavioral or motivation question. Tie it to the selected "
            "role where natural; assess communication and professional judgment, "
            "not technical trivia."
        ),
        "Coding": (
            "Ask one programming, debugging, algorithm, or implementation "
            "question suited to the role. For a non-programming role, ask a "
            "role-relevant data or technical problem-solving exercise."
        ),
    }[category]

    return f"""
You are an experienced interviewer conducting a realistic practice interview.

Candidate:
- Role: {role}
- Experience level: {experience}
- Selected interview type: {interview_type}
- Category for this question: {category}
- Question: {question_number} of {total_questions}

Generate exactly ONE clear, natural interview question that is specifically
relevant to the candidate's role and appropriate for their experience.

Question-type requirements:
{category_guidance}
The role must materially influence the topic and expected answer. Prefer
technologies, responsibilities, and scenarios normally associated with
"{role}". For example, ask a Python Developer about Python when suitable, a
React Developer about React and frontend behavior, a MERN developer about the
MongoDB/Express/React/Node stack, and a Java Developer about Java and its
ecosystem. Adapt this guidance to the actual selected role.

{adaptive_context}

STRICT QUESTION EXCLUSION:
The following questions have already been used for this role, experience
level, and interview type. You MUST NOT generate any of these questions or a
trivially reworded version of them. Generate a genuinely different question.
Do not ask any question that appears in the excluded question list. Generate
a different question that tests the same skill if necessary.

Excluded questions:
{excluded_questions_text}

Never repeat or merely reword a previous question. Ask only one question.
Do not include an answer, explanation, preamble, category label, or markdown.

Previous questions:
{previous_questions_text}

Return only the question text.
"""


def answer_evaluation_prompt(
    role,
    experience,
    interview_type,
    question,
    answer,
    question_category=None,
):
    """Create a category-aware prompt for evaluating one submitted answer."""
    category = question_category or interview_type
    if interview_type == "Full Interview":
        evaluation_focus = {
            "HR": (
                "Prioritize communication, relevance, clarity, confidence, "
                "professionalism, and completeness. Do not judge an HR answer "
                "primarily as a technical answer. Use technical_accuracy to "
                "represent soundness and role-awareness of the response, not "
                "knowledge of technical facts."
            ),
            "Coding": (
                "Prioritize correctness, logic, edge cases, complexity, and "
                "the candidate's explanation. If no executable code is needed, "
                "judge the proposed solution and reasoning."
            ),
            "Technical": (
                "Prioritize technical accuracy, understanding, completeness, "
                "clarity, and relevance to the selected role."
            ),
        }[category]
    elif category == "HR":
        evaluation_focus = (
            "Prioritize communication, relevance, clarity, confidence, "
            "professionalism, and completeness. Do not judge an HR answer "
            "primarily as a technical answer. Use technical_accuracy to "
            "represent soundness and role-awareness, not technical facts."
        )
    elif category == "Coding":
        evaluation_focus = (
            "Prioritize correctness, logic, edge cases, complexity, and "
            "explanation. Evaluate code or a clear solution explanation fairly."
        )
    else:
        evaluation_focus = (
            "Prioritize technical accuracy, understanding, completeness, "
            "clarity, and relevance to the selected role."
        )

    return f"""
You are an experienced interviewer evaluating one practice interview answer.

Candidate role: {role}
Experience level: {experience}
Selected interview type: {interview_type}
Question category: {category}

Question:
{question}

Candidate answer:
{answer}

Evaluate only the provided answer, at a standard appropriate to the candidate's
experience. {evaluation_focus}
Score each category from 0 to 10, using integer scores. "score" is the overall
quality of this answer for its question category.
Give concise, specific, constructive feedback. Do not invent candidate facts.
The ideal answer should answer this exact question and be role- and experience-
appropriate.

Return only a JSON object with exactly these fields:
{{
  "score": 0,
  "technical_accuracy": 0,
  "completeness": 0,
  "clarity": 0,
  "relevance": 0,
  "strengths": ["specific demonstrated strength"],
  "improvements": ["specific actionable improvement"],
  "ideal_answer": "A concise example of a strong answer."
}}
"""


def final_report_prompt(
    role,
    experience,
    interview_type,
    questions,
    answers,
    evaluations,
):
    """Create a qualitative-only report prompt from the complete interview."""
    interview_data = []
    for index, (question, answer, evaluation) in enumerate(
        zip(questions, answers, evaluations),
        start=1,
    ):
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        category = evaluation.get(
            "question_category",
            get_question_category(interview_type, index),
        )
        if not isinstance(category, str):
            category = get_question_category(interview_type, index)

        def safe_score(value):
            if isinstance(value, bool) or not isinstance(
                value, (int, float, str, type(None))
            ):
                return None
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        def safe_text_list(value):
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple)):
                return []
            return [
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            ]

        interview_data.append(
            {
                "question_number": index,
                "category": category,
                "question": question if isinstance(question, str) else "",
                "answer": answer if isinstance(answer, str) else "",
                "evaluation": {
                    "score": safe_score(evaluation.get("score")),
                    "technical_accuracy": safe_score(
                        evaluation.get("technical_accuracy")
                    ),
                    "completeness": safe_score(
                        evaluation.get("completeness")
                    ),
                    "clarity": safe_score(evaluation.get("clarity")),
                    "relevance": safe_score(evaluation.get("relevance")),
                    "strengths": safe_text_list(evaluation.get("strengths")),
                    "improvements": safe_text_list(
                        evaluation.get("improvements")
                    ),
                },
            }
        )

    return f"""
You are a senior interviewer writing a qualitative report based on the
candidate's COMPLETE practice interview.

Role: {role}
Experience: {experience}
Interview Type: {interview_type}

For every question, the record contains the question, candidate answer, and
existing evaluation. These evaluations are the source of truth for numeric
scores. Do not calculate, change, or return any numeric scores.

Complete interview record:
{json.dumps(interview_data, ensure_ascii=False, indent=2)}

Analyze patterns across all questions. Make strengths and weaknesses specific
to evidence in the answers and evaluations. Recommendations should address
recurring or important improvement areas. Do not invent candidate facts.

Return only valid JSON, either plain JSON or a JSON markdown code block, with
these qualitative fields:
{{
  "performance_summary": "A concise summary of performance across the entire interview.",
  "strengths": ["specific demonstrated strength"],
  "weaknesses": ["specific area to improve"],
  "recommendations": ["practical preparation recommendation"]
}}
"""
