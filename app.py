"""Streamlit user interface for the IntervAI interview simulator."""

import math
import re
from datetime import datetime
from uuid import uuid4

import streamlit as st

from interview_engine import (
    GeminiCallError,
    InterviewHistoryError,
    add_questions_to_history,
    clear_question_history,
    clear_interview_history,
    create_interview_record,
    delete_interview_record,
    evaluate_answer,
    generate_final_report,
    generate_question,
    load_interview_history,
    load_question_history,
    save_completed_interview,
)
from prompts import get_question_category


st.set_page_config(
    page_title="IntervAI",
    page_icon="🤖",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f9fc;
        color: #172033;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1 {
        color: #172033;
        letter-spacing: -0.035em;
        font-weight: 750;
    }
    h2, h3, h4 {
        color: #202c43;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] {
        background: #f0f4fa;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e9f2;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 2px 8px rgba(29, 47, 76, 0.04);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e2e8f0;
        border-radius: 14px;
        background: #ffffff;
        box-shadow: 0 3px 12px rgba(29, 47, 76, 0.035);
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-radius: 9px;
    }
    [data-testid="stTextArea"] label {
        font-size: 1.05rem;
        font-weight: 650;
        color: #202c43;
    }
    [data-testid="stAlert"] {
        border-radius: 10px;
    }
    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 2.65rem;
        transition: border-color 120ms ease, background-color 120ms ease;
    }
    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }
        [data-testid="stMetric"] {
            padding: 0.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SESSION_DEFAULTS = {
    "interview_started": False,
    "interview_completed": False,
    "current_question": None,
    "current_question_number": 0,
    "current_question_category": None,
    "questions": [],
    "answers": [],
    "evaluations": [],
    "answer_submitted": False,
    "last_evaluation": None,
    "final_report": None,
    "role": None,
    "experience": None,
    "interview_type": None,
    "number_of_questions": 5,
    "confirm_clear_question_history": False,
    "current_page": "New Interview",
    "selected_history_id": None,
    "history_saved": False,
    "history_save_error": False,
    "interview_id": None,
    "delete_interview_id": None,
    "confirm_clear_interview_history": False,
}

for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default.copy() if isinstance(default, list) else default

for key in ("questions", "answers", "evaluations"):
    if not isinstance(st.session_state[key], list):
        st.session_state[key] = []

for key in ("interview_started", "interview_completed", "answer_submitted"):
    if not isinstance(st.session_state[key], bool):
        st.session_state[key] = False

if (
    isinstance(st.session_state.number_of_questions, bool)
    or not isinstance(st.session_state.number_of_questions, int)
    or not 3 <= st.session_state.number_of_questions <= 10
):
    st.session_state.number_of_questions = 5

if (
    isinstance(st.session_state.current_question_number, bool)
    or not isinstance(st.session_state.current_question_number, int)
):
    st.session_state.current_question_number = 0

if not isinstance(st.session_state.final_report, dict):
    st.session_state.final_report = None
    st.session_state.interview_completed = False

if not isinstance(st.session_state.last_evaluation, dict):
    st.session_state.last_evaluation = None
    st.session_state.answer_submitted = False


def reset_interview():
    interview_keys = (
        "interview_started",
        "interview_completed",
        "current_question",
        "current_question_number",
        "current_question_category",
        "questions",
        "answers",
        "evaluations",
        "answer_submitted",
        "last_evaluation",
        "final_report",
        "role",
        "experience",
        "interview_type",
        "number_of_questions",
        "history_saved",
        "history_save_error",
        "interview_id",
    )
    for key in interview_keys:
        default = SESSION_DEFAULTS[key]
        st.session_state[key] = default.copy() if isinstance(default, list) else default


def show_list(title, entries, empty_message):
    st.markdown(title)
    if entries:
        for entry in entries:
            st.markdown(f"- {entry}")
    else:
        st.caption(empty_message)


def log_failure(context, error):
    """Write technical details to the terminal without exposing API keys."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
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
        print(
            f"[IntervAI] {context}: "
            f"{type(current).__name__}: {message}"
        )
        current = current.__cause__ or current.__context__


def failure_message(error, action):
    if not isinstance(error, GeminiCallError):
        return f"Could not {action}. Please try again."

    messages = {
        "quota_exhausted": (
            "Gemini's free-tier quota is currently exhausted for this project. "
            "Please try again after the quota resets, or configure another "
            "Gemini project/API key."
        ),
        "temporary_error": "Gemini is temporarily unavailable. Please try again in a moment.",
        "authentication_error": "Gemini authentication failed. Please check your GEMINI_API_KEY.",
        "model_unavailable": (
            "Gemini is temporarily unavailable. Please try again later."
        ),
        "invalid_request": "Gemini could not process this request. Please try again.",
        "question_not_unique": (
            "Could not generate a new unique question right now. Please try again."
        ),
        "unknown_error": f"Could not {action}. Please try again.",
    }
    return messages.get(
        error.error_type,
        f"Could not {action}. Please try again.",
    )


def _valid_report_score(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 100
    )


def _chart_score(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(score):
        return None
    return min(100.0, max(0.0, score))


def _format_history_date(value):
    if not isinstance(value, str):
        return "Date unavailable"
    try:
        date = datetime.fromisoformat(value)
    except ValueError:
        return value
    return f"{date.strftime('%b')} {date.day}, {date.year} {date.strftime('%H:%M')}"


def _history_timestamp(record):
    value = record.get("date")
    if not isinstance(value, str):
        return 0
    try:
        return datetime.fromisoformat(value).timestamp()
    except (OSError, OverflowError, ValueError):
        return 0


def _report_entries(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [
        entry.strip()
        for entry in value
        if isinstance(entry, str) and entry.strip()
    ]


def render_saved_report(record):
    report = record.get("final_report")
    if not isinstance(report, dict):
        st.error("This saved interview does not contain a valid report.")
        return

    st.header("🏆 Interview Performance Report")
    st.write(
        f'**Role:** {record.get("role", "Unknown")}  \n'
        f'**Experience:** {record.get("experience", "Unknown")}  \n'
        f'**Interview Type:** {record.get("interview_type", "Unknown")}  \n'
        f'**Date:** {_format_history_date(record.get("date"))}  \n'
        f'**Question Count:** {record.get("question_count", 0)}'
    )

    score = report.get("overall_score")
    st.metric(
        "Overall Score",
        f"{score}/100" if _valid_report_score(score) else "N/A",
    )
    if _valid_report_score(score):
        st.progress(score / 100)
    if report.get("analysis_notice"):
        st.info(report["analysis_notice"])

    st.markdown("### Performance Summary")
    summary = report.get(
        "performance_summary",
        report.get("interview_summary", "No summary saved."),
    )
    st.write(summary if isinstance(summary, str) else "No summary saved.")
    show_list(
        "### Strengths",
        _report_entries(report.get("strengths")),
        "No strengths saved.",
    )
    show_list(
        "### Areas for Improvement",
        _report_entries(
            report.get("weaknesses", report.get("areas_to_improve"))
        ),
        "No weaknesses saved.",
    )
    show_list(
        "### Recommendations",
        _report_entries(report.get("recommendations")),
        "No recommendations saved.",
    )
    st.markdown("### Final Tip")
    final_tip = report.get("final_tip", "No final tip saved.")
    st.info(final_tip if isinstance(final_tip, str) else "No final tip saved.")

    st.markdown("### Question Analysis")
    analysis = report.get("question_analysis", [])
    questions = record.get("questions", [])
    answers = record.get("answers", [])
    evaluations = record.get("evaluations", [])
    questions = questions if isinstance(questions, list) else []
    answers = answers if isinstance(answers, list) else []
    evaluations = evaluations if isinstance(evaluations, list) else []
    if not isinstance(analysis, list) or not analysis:
        analysis = []
        for index, question in enumerate(questions):
            answer = answers[index] if index < len(answers) else ""
            evaluation = (
                evaluations[index]
                if index < len(evaluations) and isinstance(evaluations[index], dict)
                else {}
            )
            analysis.append(
                {
                    "question_number": index + 1,
                    "category": evaluation.get("question_category", ""),
                    "question": question,
                    "answer": answer,
                    "score": evaluation.get("score"),
                    "strengths": evaluation.get("strengths", []),
                    "weaknesses": evaluation.get("improvements", []),
                }
            )

    analysis = [item for item in analysis if isinstance(item, dict)]
    for index, item in enumerate(analysis):
        question_number = item.get("question_number", "")
        st.markdown(f'#### Question {question_number}')
        st.markdown(f'**Question:** {item.get("question", "")}')
        st.markdown(f'**Your Answer:** {item.get("answer", "")}')
        item_score = item.get("score")
        score_text = (
            f"{item_score}/10"
            if isinstance(item_score, (int, float))
            and not isinstance(item_score, bool)
            else "Not scored"
        )
        st.markdown(f"**Score:** {score_text}")
        feedback = []
        feedback.extend(
            f"Strength: {value}"
            for value in _report_entries(item.get("strengths"))
        )
        feedback.extend(
            f"Improvement: {value}"
            for value in _report_entries(item.get("weaknesses"))
        )
        if feedback:
            st.markdown("**Feedback:** " + " · ".join(feedback))
        if index < len(analysis) - 1:
            st.divider()


st.title("🤖 IntervAI")
st.subheader("Your AI-powered personal interviewer")
st.write(
    "Practice realistic interviews, get instant feedback, and track your progress."
)
st.divider()

with st.sidebar:
    st.markdown("## 🤖 IntervAI")
    st.caption("Your AI-powered personal interviewer")
    st.markdown("### Navigation")
    if st.button(
        "🏠 New Interview",
        key="navigate_new_interview",
        use_container_width=True,
    ):
        reset_interview()
        st.session_state.current_page = "New Interview"
        st.session_state.selected_history_id = None
        st.rerun()
    if st.button(
        "📚 Interview History",
        key="navigate_interview_history",
        use_container_width=True,
    ):
        st.session_state.current_page = "Interview History"
        st.session_state.selected_history_id = None
        st.rerun()
    if st.session_state.interview_started or st.session_state.interview_completed:
        if st.button(
            "Return to Current Interview",
            key="return_current_interview",
        ):
            st.session_state.current_page = "New Interview"
            st.session_state.selected_history_id = None
            st.rerun()

    st.divider()
    st.markdown("### Practice")
    question_history = load_question_history()
    practiced_count = sum(
        len(questions) for questions in question_history.values()
    )
    st.caption(f"Questions Practiced: {practiced_count}")
    if st.button("Clear Question History", key="clear_question_history"):
        st.session_state.confirm_clear_question_history = True

    if st.session_state.confirm_clear_question_history:
        st.warning("Are you sure you want to clear your previous question history?")
        confirm_column, cancel_column = st.columns(2)
        if confirm_column.button("Confirm", key="confirm_clear_history"):
            try:
                clear_question_history()
            except OSError as error:
                log_failure("Question history clearing failed", error)
                st.error("Could not clear question history. Please try again.")
            else:
                st.session_state.confirm_clear_question_history = False
                st.success("Question history cleared.")
        if cancel_column.button("Cancel", key="cancel_clear_history"):
            st.session_state.confirm_clear_question_history = False


if st.session_state.current_page == "Interview History":
    st.header("📚 Interview History")
    try:
        interview_history = load_interview_history()
    except InterviewHistoryError as error:
        log_failure("Interview history load failed", error)
        st.error(
            "Interview history is currently unavailable. Your saved data has "
            "not been changed."
        )
    else:
        interview_history.sort(
            key=_history_timestamp,
            reverse=True,
        )
        valid_scores = [
            score
            for record in interview_history
            for score in [
                record.get(
                    "overall_score",
                    (record.get("final_report") or {}).get("overall_score")
                    if isinstance(record.get("final_report"), dict)
                    else None,
                )
            ]
            if _valid_report_score(score)
        ]

        stat_columns = st.columns(3)
        stat_columns[0].metric("Completed Interviews", len(interview_history))
        stat_columns[1].metric(
            "Average Score",
            f"{sum(valid_scores) / len(valid_scores):.1f}/100"
            if valid_scores
            else "N/A",
        )
        stat_columns[2].metric(
            "Best Score",
            f"{max(valid_scores):g}/100" if valid_scores else "N/A",
        )

        if interview_history:
            chart_records = []
            for record in sorted(interview_history, key=_history_timestamp):
                report = record.get("final_report")
                saved_score = record.get("overall_score")
                if saved_score is None and isinstance(report, dict):
                    saved_score = report.get("overall_score")
                score = _chart_score(saved_score)
                if score is not None:
                    chart_records.append(
                        {
                            "Interview": f"Interview {len(chart_records) + 1}",
                            "Score": score,
                            "Date": _format_history_date(record.get("date")),
                        }
                    )

            if chart_records:
                st.markdown("### 📈 Score Progress")
                st.caption("Your interview performance over time")
                st.vega_lite_chart(
                    spec={
                        "data": {"values": chart_records},
                        "mark": {
                            "type": "line",
                            "point": {"filled": True, "size": 100},
                        },
                        "encoding": {
                            "x": {
                                "field": "Interview",
                                "type": "ordinal",
                                "sort": [
                                    record["Interview"] for record in chart_records
                                ],
                                "axis": {"title": "Interview"},
                            },
                            "y": {
                                "field": "Score",
                                "type": "quantitative",
                                "scale": {"domain": [0, 100]},
                                "axis": {
                                    "title": "Score (0–100)",
                                    "values": [0, 20, 40, 60, 80, 100],
                                },
                            },
                            "tooltip": [
                                {"field": "Interview", "type": "ordinal"},
                                {
                                    "field": "Date",
                                    "type": "nominal",
                                    "title": "Interview date",
                                },
                                {
                                    "field": "Score",
                                    "type": "quantitative",
                                    "title": "Overall score",
                                },
                            ],
                        },
                    },
                    height=320,
                )

            for record in interview_history:
                interview_id = record.get("id")
                with st.container(border=True):
                    score = record.get(
                        "overall_score",
                        (record.get("final_report") or {}).get("overall_score")
                        if isinstance(record.get("final_report"), dict)
                        else None,
                    )
                    st.markdown(f"### {record.get('role', 'Unknown role')}")
                    st.caption(
                        f'{record.get("interview_type", "Unknown type")} · '
                        f'{record.get("experience", "Experience unavailable")} · '
                        f'{record.get("question_count", 0)} questions'
                    )
                    score_column, date_column = st.columns((1, 2))
                    score_column.metric(
                        "Overall Score",
                        f"{score:g}/100" if _valid_report_score(score) else "N/A",
                    )
                    date_column.caption("Completed")
                    date_column.write(_format_history_date(record.get("date")))
                    view_column, delete_column = st.columns(2)
                    if view_column.button(
                        "View Report",
                        key=f"view_interview_{interview_id}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_history_id = interview_id
                        st.session_state.current_page = "Saved Report"
                        st.rerun()
                    if delete_column.button(
                        "Delete",
                        key=f"delete_interview_{interview_id}",
                    ):
                        st.session_state.delete_interview_id = interview_id
                        st.rerun()

                    if st.session_state.delete_interview_id == interview_id:
                        st.warning("Delete this saved interview and its report?")
                        confirm_column, cancel_column = st.columns(2)
                        if confirm_column.button(
                            "Confirm Delete",
                            key=f"confirm_delete_interview_{interview_id}",
                        ):
                            try:
                                delete_interview_record(interview_id)
                            except (InterviewHistoryError, OSError) as error:
                                log_failure("Interview deletion failed", error)
                                st.error(
                                    "Could not delete this interview. Please try again."
                                )
                            else:
                                st.session_state.delete_interview_id = None
                                st.rerun()
                        if cancel_column.button(
                            "Cancel",
                            key=f"cancel_delete_interview_{interview_id}",
                        ):
                            st.session_state.delete_interview_id = None
                            st.rerun()

            if st.button("Clear All Interview History"):
                st.session_state.confirm_clear_interview_history = True
            if st.session_state.confirm_clear_interview_history:
                st.warning(
                    "Are you sure you want to permanently delete all saved interviews?"
                )
                confirm_column, cancel_column = st.columns(2)
                if confirm_column.button(
                    "Confirm Clear All",
                    key="confirm_clear_all_interview_history_button",
                ):
                    try:
                        clear_interview_history()
                    except OSError as error:
                        log_failure("Interview history clearing failed", error)
                        st.error(
                            "Could not clear interview history. Please try again."
                        )
                    else:
                        st.session_state.confirm_clear_interview_history = False
                        st.session_state.delete_interview_id = None
                        st.rerun()
                if cancel_column.button(
                    "Cancel",
                    key="cancel_clear_interview_history",
                ):
                    st.session_state.confirm_clear_interview_history = False
                    st.rerun()
        else:
            st.info("No completed interviews have been saved yet.")

elif st.session_state.current_page == "Saved Report":
    try:
        saved_interviews = load_interview_history()
    except InterviewHistoryError as error:
        log_failure("Saved report load failed", error)
        st.error("This saved report is currently unavailable.")
    else:
        saved_record = next(
            (
                record
                for record in saved_interviews
                if record.get("id") == st.session_state.selected_history_id
            ),
            None,
        )
        if saved_record is None:
            st.warning("This interview is no longer in your saved history.")
        else:
            render_saved_report(saved_record)
        if st.button("← Back to Interview History"):
            st.session_state.current_page = "Interview History"
            st.rerun()

elif st.session_state.interview_completed:
    report = st.session_state.final_report
    st.header("🏆 Interview Performance Report")
    st.success("Your interview assessment is ready.")

    if st.session_state.history_save_error:
        st.warning(
            "Your report is ready, but it could not be saved to Interview "
            "History."
        )
        if st.button("Retry Saving to Interview History"):
            try:
                record = create_interview_record(
                    interview_id=st.session_state.interview_id,
                    role=st.session_state.role,
                    experience=st.session_state.experience,
                    interview_type=st.session_state.interview_type,
                    questions=st.session_state.questions,
                    answers=st.session_state.answers,
                    evaluations=st.session_state.evaluations,
                    final_report=st.session_state.final_report,
                )
                save_completed_interview(record)
            except Exception as error:
                log_failure("Interview history retry failed", error)
                st.error(
                    "Could not save the interview to history. Please try again."
                )
            else:
                st.session_state.history_saved = True
                st.session_state.history_save_error = False
                st.success("Interview saved to history.")

    st.metric(
        "Overall Score",
        (
            f'{report["overall_score"]}/100'
            if report["overall_score"] is not None
            else "N/A"
        ),
    )
    score_columns = st.columns(4)
    score_cards = (
        ("Technical Knowledge", report["technical_knowledge"], 10),
        ("Completeness", report["completeness"], 10),
        ("Clarity", report["clarity"], 10),
        ("Relevance", report["relevance"], 10),
    )
    for column, (label, value, maximum) in zip(score_columns, score_cards):
        display_value = f"{value}/{maximum}" if value is not None else "N/A"
        column.metric(label, display_value)

    if report["overall_score"] is not None:
        st.progress(report["overall_score"] / 100)
    if report.get("analysis_notice"):
        st.info(report["analysis_notice"])

    st.markdown("### Performance Summary")
    st.write(report["performance_summary"])
    show_list("### Strengths", report["strengths"], "No strengths were returned.")
    show_list(
        "### Areas for Improvement",
        report["weaknesses"],
        "No weaknesses were identified in the saved evaluations.",
    )
    show_list(
        "### Recommendations",
        report["recommendations"],
        "No recommendations were returned.",
    )
    st.markdown("### Final Tip")
    st.info(report["final_tip"])
    with st.expander("Question Analysis"):
        for item in report["question_analysis"]:
            score = item["score"]
            score_text = f"{score:g}/10" if score is not None else "Not scored"
            st.markdown(
                f'**Question {item["question_number"]} · '
                f'{item["category"]} · {score_text}**'
            )
            st.write(item["question"])
            st.caption(f'Your answer: {item["answer"]}')
            if item["strengths"]:
                st.markdown(
                    "**Strengths:** " + "; ".join(item["strengths"])
                )
            if item["weaknesses"]:
                st.markdown(
                    "**Areas to improve:** " + "; ".join(item["weaknesses"])
                )
            if item["question_number"] < len(report["question_analysis"]):
                st.divider()
    st.caption(
        f'{st.session_state.role} · {st.session_state.experience} · '
        f'{st.session_state.interview_type} · '
        f'{st.session_state.number_of_questions} questions'
    )

    if st.button("🔄 Practice Again", use_container_width=True):
        reset_interview()
        st.session_state.current_page = "New Interview"
        st.rerun()

elif not st.session_state.interview_started:
    st.header("Start Your Interview")
    st.write(
        "Customize your interview and let AI adapt the questions to your role "
        "and experience."
    )

    roles = [
        "Python Developer",
        "Java Developer",
        "JavaScript Developer",
        "React Developer",
        "MERN Stack Developer",
        "Frontend Developer",
        "Backend Developer",
        "Full Stack Developer",
        "Data Analyst",
        "Data Scientist",
        "AI/ML Engineer",
        "DevOps Engineer",
        "Software Engineer",
    ]
    role = st.selectbox("What role are you preparing for?", roles)
    experience = st.selectbox(
        "Experience Level",
        ["Fresher", "0–2 Years", "2–5 Years", "5+ Years"],
    )
    interview_type = st.radio(
        "Interview Type",
        ["Technical", "HR", "Coding", "Full Interview"],
        horizontal=True,
    )
    number_of_questions = st.slider(
        "Number of Questions",
        min_value=3,
        max_value=10,
        value=5,
    )

    if st.button("🚀 Start Interview", use_container_width=True):
        if not role.strip() or not experience or not interview_type:
            st.error("Choose a role, experience level, and interview type.")
        elif not 3 <= number_of_questions <= 10:
            st.error("Choose between 3 and 10 questions.")
        else:
            category = get_question_category(interview_type, 1)
            try:
                with st.spinner("Preparing your first interview question..."):
                    question = generate_question(
                        role=role,
                        experience=experience,
                        interview_type=interview_type,
                        question_number=1,
                        total_questions=number_of_questions,
                        previous_questions=[],
                        previous_evaluation=None,
                    )
            except Exception as error:
                log_failure("Interview start failed", error)
                st.error(
                    failure_message(error, "start the interview")
                )
            else:
                reset_interview()
                st.session_state.role = role
                st.session_state.experience = experience
                st.session_state.interview_type = interview_type
                st.session_state.number_of_questions = number_of_questions
                st.session_state.current_question_number = 1
                st.session_state.current_question = question
                st.session_state.current_question_category = category
                st.session_state.questions = [question]
                st.session_state.interview_started = True
                st.session_state.interview_id = uuid4().hex
                st.session_state.current_page = "New Interview"
                try:
                    add_questions_to_history(
                        role,
                        experience,
                        interview_type,
                        [question],
                    )
                except OSError as error:
                    log_failure("Question history save failed", error)
                st.rerun()

    st.divider()
    st.markdown("### Prepare with purpose")
    feature_columns = st.columns(4)
    feature_cards = (
        ("🎯 Adaptive Questions", "Questions adjust based on your responses."),
        ("🧠 AI Feedback", "Get detailed feedback after every answer."),
        ("📊 Performance Report", "Understand strengths and improvement areas."),
        ("📚 Practice History", "Track your interview performance over time."),
    )
    for column, (title, description) in zip(feature_columns, feature_cards):
        with column:
            st.markdown(f"**{title}**")
            st.caption(description)

else:
    total_questions = st.session_state.number_of_questions
    current_number = st.session_state.current_question_number

    if (
        not st.session_state.current_question
        or not isinstance(current_number, int)
        or not 1 <= current_number <= total_questions
    ):
        st.error("The interview state is incomplete. Please start a new interview.")
        if st.button("Start a new interview"):
            reset_interview()
            st.rerun()
    else:
        st.header("Interview in Progress")
        st.markdown(f"### Question {current_number} of {total_questions}")
        st.progress(current_number / total_questions)
        st.caption(
            f"{round(current_number / total_questions * 100)}% complete"
        )
        with st.container(border=True):
            st.markdown("#### Interview Question")
            st.markdown(st.session_state.current_question)
        if not st.session_state.answer_submitted:
            answer = st.text_area(
                "Your Answer",
                height=180,
                placeholder="Explain your answer as you would in a real interview...",
                key=f"answer_{current_number}",
            )
            if st.button("📤 Submit Answer", use_container_width=True):
                if not answer.strip():
                    st.warning("Please enter an answer before submitting.")
                else:
                    try:
                        with st.spinner("AI is evaluating your answer..."):
                            evaluation = evaluate_answer(
                                role=st.session_state.role,
                                experience=st.session_state.experience,
                                interview_type=st.session_state.interview_type,
                                question=st.session_state.current_question,
                                answer=answer,
                                question_category=st.session_state.current_question_category,
                            )
                    except Exception as error:
                        log_failure("Answer evaluation failed", error)
                        st.error(
                            failure_message(error, "evaluate your answer")
                        )
                    else:
                        st.session_state.answers.append(answer.strip())
                        st.session_state.evaluations.append(evaluation)
                        st.session_state.last_evaluation = evaluation
                        st.session_state.answer_submitted = True
                        st.rerun()
        else:
            evaluation = st.session_state.last_evaluation
            if not isinstance(evaluation, dict):
                st.error("The saved evaluation is unavailable. Please restart the interview.")
            else:
                category = st.session_state.current_question_category
                st.markdown("### Answer Evaluation")
                score_value = evaluation["score"]
                score_label = (
                    f"{score_value}/10"
                    if score_value is not None
                    else "Not available"
                )
                st.metric("Score", score_label)
                metric_columns = st.columns(4)
                metric_labels = (
                    ("Technical Accuracy", "technical_accuracy"),
                    ("Completeness", "completeness"),
                    ("Clarity", "clarity"),
                    ("Relevance", "relevance"),
                )
                if category == "HR":
                    metric_labels = (
                        ("Role Awareness", "technical_accuracy"),
                        ("Completeness", "completeness"),
                        ("Clarity", "clarity"),
                        ("Relevance", "relevance"),
                    )
                for column, (label, key) in zip(metric_columns, metric_labels):
                    value = evaluation[key]
                    column.metric(
                        label,
                        f"{value}/10" if value is not None else "N/A",
                    )

                show_list(
                    "### What You Did Well",
                    evaluation["strengths"],
                    "No specific strengths were returned.",
                )
                show_list(
                    "### How to Improve",
                    evaluation["improvements"],
                    "No specific improvements were returned.",
                )
                st.markdown("### Ideal Answer")
                st.write(evaluation["ideal_answer"])

                if current_number < total_questions:
                    if st.button("➡️ Next Question", use_container_width=True):
                        if (
                            not st.session_state.answer_submitted
                            or not isinstance(
                                st.session_state.last_evaluation, dict
                            )
                            or len(st.session_state.questions) != current_number
                            or len(st.session_state.answers) != current_number
                            or len(st.session_state.evaluations) != current_number
                        ):
                            st.error(
                                "Submit and evaluate this answer before "
                                "requesting the next question."
                            )
                            st.stop()

                        previous_evaluation = dict(evaluation)
                        previous_evaluation["previous_answer"] = (
                            st.session_state.answers[-1]
                        )
                        next_number = current_number + 1
                        next_category = get_question_category(
                            st.session_state.interview_type,
                            next_number,
                        )
                        try:
                            with st.spinner("Preparing your next question..."):
                                next_question = generate_question(
                                    role=st.session_state.role,
                                    experience=st.session_state.experience,
                                    interview_type=st.session_state.interview_type,
                                    question_number=next_number,
                                    total_questions=total_questions,
                                    previous_questions=st.session_state.questions,
                                    previous_evaluation=previous_evaluation,
                                )
                        except Exception as error:
                            log_failure("Next question generation failed", error)
                            st.error(
                                failure_message(
                                    error, "generate the next question"
                                )
                            )
                        else:
                            st.session_state.current_question = next_question
                            st.session_state.current_question_category = next_category
                            st.session_state.current_question_number = next_number
                            st.session_state.questions.append(next_question)
                            try:
                                add_questions_to_history(
                                    st.session_state.role,
                                    st.session_state.experience,
                                    st.session_state.interview_type,
                                    [next_question],
                                )
                            except OSError as error:
                                log_failure("Question history save failed", error)
                            st.session_state.answer_submitted = False
                            st.session_state.last_evaluation = None
                            st.rerun()
                else:
                    if st.button(
                        "🏆 Generate Final Report",
                        use_container_width=True,
                    ):
                        try:
                            with st.spinner(
                                "Analyzing your complete interview..."
                            ):
                                final_report = generate_final_report(
                                    role=st.session_state.role,
                                    experience=st.session_state.experience,
                                    interview_type=st.session_state.interview_type,
                                    questions=st.session_state.questions,
                                    answers=st.session_state.answers,
                                    evaluations=st.session_state.evaluations,
                                )
                        except Exception as error:
                            log_failure("Final report generation failed", error)
                            st.error(
                                failure_message(
                                    error, "generate the final report"
                                )
                            )
                        else:
                            st.session_state.final_report = final_report
                            try:
                                record = create_interview_record(
                                    interview_id=st.session_state.interview_id,
                                    role=st.session_state.role,
                                    experience=st.session_state.experience,
                                    interview_type=st.session_state.interview_type,
                                    questions=st.session_state.questions,
                                    answers=st.session_state.answers,
                                    evaluations=st.session_state.evaluations,
                                    final_report=final_report,
                                )
                                save_completed_interview(record)
                            except Exception as error:
                                log_failure(
                                    "Completed interview archive failed",
                                    error,
                                )
                                st.session_state.history_saved = False
                                st.session_state.history_save_error = True
                            else:
                                st.session_state.history_saved = True
                                st.session_state.history_save_error = False
                            st.session_state.interview_completed = True
                            st.session_state.current_page = "New Interview"
                            st.rerun()
