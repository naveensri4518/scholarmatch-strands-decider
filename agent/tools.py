"""ScholarMatch tools: read a student's application from sample data, and record the verdict.

The sample data is mirrored from agent/data/applications.json (hand-authored, deterministic). The
agent calls the plain helpers (_get_application, _record_decision) directly; the @tool wrappers
expose the same actions to a Strands Agent if you wire one in. Keep a clear one-line docstring on
each tool: the model reads it.
"""

from __future__ import annotations

import json
from pathlib import Path

from strands import tool

_DATA = Path(__file__).resolve().parent / "data" / "applications.json"
APPLICATIONS: dict[str, dict] = {}
if _DATA.exists():
    for _row in json.loads(_DATA.read_text()):
        APPLICATIONS[_row["student_id"]] = _row
DECISIONS: list[dict] = []


def _get_application(student_id: str) -> dict:
    sid = (student_id or "").strip().upper()
    row = APPLICATIONS.get(sid)
    if not row:
        return {"error": f"no application for {student_id!r}"}
    return {
        "student_id": row["student_id"],
        "name": row.get("name", ""),
        "scholarship": row.get("scholarship", "Need-Based Financial Aid Scholarship"),
        "reason": row.get("reason", ""),
        "cgpa": row.get("cgpa"),
        "credits": row.get("credits"),
        "income_bracket": row.get("income_bracket", ""),
    }


@tool
def get_application(student_id: str) -> dict:
    """Look up a student's scholarship application (reason, scholarship, basic facts) by student id."""
    return _get_application(student_id)


def _record_decision(student_id: str, verdict: str, note: str = "") -> dict:
    entry = {"student_id": student_id, "verdict": verdict, "note": note}
    DECISIONS.append(entry)
    return {"recorded": True, **entry}


@tool
def record_decision(student_id: str, verdict: str, note: str = "") -> dict:
    """Record the eligibility verdict (eligible/needs_review/ineligible) for an application."""
    return _record_decision(student_id, verdict, note)


TOOLS = [get_application, record_decision]
