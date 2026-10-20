"""ScholarMatch: a scholarship-review agent (local only, not deployed).

Strands Decider 2B reads a student's application reason and decides whether it matches the
scholarship policy (eligible / needs_review / ineligible, #policy). A Strands Agent then writes a
short explanation of that verdict.

    from agent.agent import HackathonAgent
    out = HackathonAgent().chat("My father lost his job and we can no longer afford my tuition.")
    out["reply"], out["decisions"]
"""

from __future__ import annotations

import re
import time
from typing import Any

from strands import Agent

from decider_kit import Decider
from decider_kit.llm import agent_model, default_model_id, strip_thinking

from .decision import POLICY, SCHOLARSHIP, ScholarshipPolicyCheck
from .tools import _get_application, _record_decision

NAME = "ScholarMatch"
SYSTEM_PROMPT = (
    "You are ScholarMatch, a scholarship reviewer for a university scholarship office. "
    "You are given a student's application reason, the scholarship policy, and a verdict "
    "(eligible, needs_review, or ineligible) that Strands Decider 2B has already reached. "
    "Write a short, plain explanation (one or two sentences) for the applicant that states the "
    "verdict and why, referring to the scholarship policy. Do not re-decide; explain the given "
    "verdict. Be brief and kind. Respond in clear English only."
)

_STUDENT_ID = re.compile(r"\bSTU-\d+\b", re.IGNORECASE)


class HackathonAgent:
    """One policy check + fresh Strands Agent (explainer) per chat turn."""

    def __init__(self, decider: Decider | None = None, model_id: str | None = None, eager: bool | None = None):
        import os

        self.decider = decider or Decider()
        self.eager = eager if eager is not None else os.environ.get("DEMO_EAGER") == "1"
        self.model_id = model_id or default_model_id()  # LLM_PROVIDER in .env: bedrock, local, ugmdu
        self.policy_check = ScholarshipPolicyCheck(self.decider)

    def _fresh_agent(self) -> Agent:
        """Create a new Strands Agent for this turn.

        A fresh Agent per chat() call means no accumulated conversation history from prior
        applications can bleed into the current explanation.  The explainer has no tools and
        is stateless between turns, so there is no cost to recreating it.
        """
        return Agent(model=agent_model(self.model_id), system_prompt=SYSTEM_PROMPT,
                     tools=[], callback_handler=None)

    # Deterministic, policy-grounded explanation templates keyed on action.
    # These are the primary response path. They are always consistent with the verdict
    # and require no LLM call — eliminating the entire category of LLM-vs-guard conflicts.
    _EXPLAIN_ELIGIBLE = (
        "Your application has been found eligible for the Need-Based Financial Aid Scholarship. "
        "Your reason describes a specific financial difficulty that affects your ability to pay "
        "tuition, which matches the scholarship policy."
    )
    _EXPLAIN_NEEDS_REVIEW = (
        "Your application has been held for human review. "
        "The reason provided could not be assessed with sufficient confidence — it may be too "
        "vague, borderline, or ambiguous for an automatic decision. A reviewer will assess it."
    )
    _EXPLAIN_INELIGIBLE = (
        "Your application has been found ineligible for the Need-Based Financial Aid Scholarship. "
        "The reason provided does not demonstrate a genuine financial hardship affecting tuition "
        "payment, as required by the scholarship policy."
    )

    def _explain(self, reason: str, verdict) -> str:  # noqa: ARG002  (reason reserved for future use)
        """Return a deterministic, policy-grounded explanation for the verdict.

        This is the primary response-generation path. It always agrees with verdict.action
        because it is keyed directly on it — no LLM, no guard needed for this path.
        The guard in _guard_reply() remains as a belt-and-suspenders check.
        """
        return {
            "eligible": self._EXPLAIN_ELIGIBLE,
            "needs_review": self._EXPLAIN_NEEDS_REVIEW,
            "ineligible": self._EXPLAIN_INELIGIBLE,
        }.get(verdict.action, self._EXPLAIN_NEEDS_REVIEW)

    def _llm_explain(self, reason: str, verdict) -> str:
        """LLM-generated explanation (available for richer wording if needed in future).

        Not called from chat() by default — the deterministic path is used instead.
        Kept here so it can be wired back in if required.
        """
        return strip_thinking(str(self._fresh_agent()(self._prompt(reason, verdict))))

    def _resolve(self, message: str) -> tuple[str, str, str]:
        """Return (reason, scholarship, student_id). Look up by id if one is mentioned, else use the message."""
        m = _STUDENT_ID.search(message or "")
        if m:
            sid = m.group(0).upper()
            app = _get_application(sid)
            if "reason" in app:
                return app["reason"], app.get("scholarship", SCHOLARSHIP), sid
        return message.strip(), SCHOLARSHIP, ""

    def chat(self, message: str, session_id: str = "default", images: list[Any] | None = None) -> dict[str, Any]:
        """One turn: resolve the application, let Strands Decider 2B decide (#policy), then explain."""
        reason, _scholarship, sid = self._resolve(message)
        t0 = time.perf_counter()
        verdict = self.policy_check.check(reason)  # Strands Decider 2B decides
        if sid:
            _record_decision(sid, verdict.action, verdict.eligibility_label)
        # Deterministic explanation: build the reply directly from verdict.action.
        # This replaces the LLM call as the primary path so the response always matches
        # the panel verdict, regardless of what the LLM would have written.
        # The LLM is still available via _llm_explain() for richer wording when needed,
        # but the guard always enforces consistency as a final backstop.
        raw_reply = self._explain(reason, verdict)
        reply = self._guard_reply(raw_reply, verdict.action)
        return {"reply": reply, "decisions": [verdict.as_log()], "model_id": self.model_id,
                "latency_ms": round((time.perf_counter() - t0) * 1000), "agent": NAME, "eager": self.eager}

    # --- safe fallback templates (used only when the LLM contradicts the decider) -----
    _FALLBACK: dict[str, str] = {
        "eligible": (
            "Your application has been found eligible. "
            "Your reason describes a specific financial difficulty that matches the "
            "Need-Based Financial Aid Scholarship policy."
        ),
        "needs_review": (
            "Your application has been held for human review. "
            "The reason provided needs further assessment before a decision can be made."
        ),
        "ineligible": (
            "Your application has been found ineligible. "
            "The reason provided does not match the financial-need criteria of the "
            "Need-Based Financial Aid Scholarship policy."
        ),
    }

    # At least one of these words/phrases MUST appear in the reply for the given action.
    # If none are present, the reply did not even mention the verdict — use the fallback.
    _REQUIRES: dict[str, list[str]] = {
        "eligible":     ["eligible", "qualif"],   # "eligible", "qualifies", "qualify"
        "needs_review": ["review", "reviewer", "assessed", "assessment", "further",
                         "held", "referred", "pending", "cannot be determined"],
        "ineligible":   ["ineligible", "not eligible", "does not qualify",
                         "does not meet", "not qualify"],
    }

    # Any of these phrases in the reply contradicts the given action → use the fallback.
    # Covers both exact substrings and common LLM paraphrases of the opposite verdict.
    _CONTRADICTS: dict[str, list[str]] = {
        "eligible": [
            "ineligible", "not eligible", "does not qualify", "does not meet",
            "needs review", "needs_review", "held for review", "held for human",
            "human review", "further review", "further assessment",
        ],
        "needs_review": [
            # All the ways a model says "eligible" -------------------------
            "is eligible", "are eligible", "has been found eligible",
            "found to be eligible", "found eligible", "been approved",
            "has been approved", "been found to be eligible",
            "application is eligible", "application qualifies",
            "your application qualifies", "qualifies for the scholarship",
            "qualifies for financial", "you qualify",
            "congratulations",
            # All the ways a model says "ineligible" ----------------------
            "is ineligible", "are ineligible", "has been found ineligible",
            "found to be ineligible", "found ineligible",
            "not eligible", "does not qualify", "does not meet the",
        ],
        "ineligible": [
            # All the ways a model says "eligible" -------------------------
            "is eligible", "are eligible", "has been found eligible",
            "found to be eligible", "found eligible", "been approved",
            "has been approved", "been found to be eligible",
            "application is eligible", "application qualifies",
            "your application qualifies", "qualifies for the scholarship",
            "qualifies for financial", "you qualify",
            "congratulations",
            # All the ways a model says "needs review" ---------------------
            "needs review", "needs_review", "held for review",
            "held for human", "human review", "further review",
            "further assessment",
        ],
    }

    def _guard_reply(self, reply: str, action: str) -> str:
        """Ensure the reply text agrees with `action`.

        Two-stage check:
        1. The reply must contain at least one required keyword for the action (_REQUIRES).
           If none are present the LLM did not mention the verdict at all — use the fallback.
        2. The reply must not contain any phrase that signals a different verdict (_CONTRADICTS).
           If one is found — use the fallback.

        This is a last-resort safety net. It should never fire when the prompt and the
        fresh-agent fix are working correctly, but it guarantees the demo panel and the
        reply text always show the same verdict regardless of LLM phrasing.
        """
        lower = reply.lower()
        # Stage 1 — required keyword must be present
        required = self._REQUIRES.get(action, [])
        if required and not any(kw in lower for kw in required):
            return self._FALLBACK.get(action, reply)
        # Stage 2 — no contradicting phrase may be present
        for bad in self._CONTRADICTS.get(action, []):
            if bad in lower:
                return self._FALLBACK.get(action, reply)
        return reply

    def _prompt(self, reason: str, verdict) -> str:
        # Pass only verdict.action to the LLM — never the raw decider label or confidence.
        # This is the single source of truth; the LLM explains the action, it does not re-decide.
        action_instructions = {
            "eligible": (
                "The application has been found ELIGIBLE. "
                "Write one or two sentences confirming eligibility and briefly noting "
                "which aspect of the applicant's reason matches the scholarship policy."
            ),
            "needs_review": (
                "The application requires HUMAN REVIEW. "
                "Write one or two sentences explaining that the application has been held "
                "for a human reviewer, and briefly say why (vague reason, low confidence, "
                "or borderline case). Do not approve or reject the application."
            ),
            "ineligible": (
                "The application has been found INELIGIBLE. "
                "Write one or two sentences explaining why the reason does not match the "
                "financial-need scholarship policy (e.g. it describes merit, ambition, or "
                "a non-financial need)."
            ),
        }
        instruction = action_instructions.get(verdict.action, action_instructions["needs_review"])
        return (
            f"Scholarship: {SCHOLARSHIP}\n\n"
            f"Policy: {POLICY}\n\n"
            f"Applicant's reason: {reason}\n\n"
            f"Verdict from Strands Decider 2B: {verdict.action.upper()}\n\n"
            f"Your task: {instruction}"
        )


if __name__ == "__main__":
    import json
    import sys

    _a = HackathonAgent()
    for _msg in sys.argv[1:] or [
        "My father lost his job three months ago and my family can no longer afford my tuition next semester.",
        "I need financial help for my studies.",
        "I have a CGPA of 9.4 and want the scholarship to fund my research project.",
    ]:
        print(json.dumps(_a.chat(_msg), indent=1))
