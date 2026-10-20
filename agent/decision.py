"""REUSABLE DECISION PATTERN: scholarship policy classification  (#policy)

ScholarMatch reads a student's scholarship application reason and decides whether it matches the
Need-Based Financial Aid Scholarship policy, returning eligible / needs_review / ineligible.

What it does
    For each application, Strands Decider 2B reads the student's free-text reason against a written
    scholarship policy and classifies it as eligible, needs_review, or ineligible. Two supporting
    questions check whether the reason is specific/genuine and whether it describes a real financial
    hardship. Low confidence, a vague reason, or absence of financial hardship always escalates to
    needs_review (a human), so the agent never confirms eligibility for a non-financial reason.

Typed questions and options
    eligibility (choice): eligible / needs_review / ineligible, each with a one-line definition.
    genuine (noul): is the reason specific and genuine, rather than vague, generic, or templated?
    financial_need (noul): does the reason describe a real financial hardship affecting tuition?
    See QUESTIONS below.

Wiring
    ScholarshipPolicyCheck.check(reason) builds a state from the reason, calls
    Decider.ask(state, QUESTIONS), and maps the answers to an action in decide(). It runs in
    agent.py before the reply is written (the "before the agent runs" form of the #policy pattern):
    the Strands Agent only explains the verdict the decider already reached; it never re-decides.

How to reuse it
    Copy this file, replace POLICY and SCHOLARSHIP with your own policy prose and name, and reword
    the three questions for your domain. The thresholds that turn probabilities into an action live
    in decide() (sure=0.6, genuine_floor=0.4, genuine_eligible=0.6, financial_floor=0.5): change
    them there, in code you own.

    genuine_eligible > genuine_floor is intentional: failing genuine_floor sends a case to
    needs_review (human review), while genuine_eligible is an additional gate on the eligible path
    so that vague or generic reasons cannot reach eligible even if confidence and financial_need
    scores are high.

    Rule precedence (in decide()):
    Rule 0  — academic merit text, no financial text → ineligible (policy disqualifier, runs first)
    Rule 1  — low confidence or genuine < floor → needs_review
    Rule 2  — eligible label, prob_ineligible ≥ 0.30 → needs_review
    Rule 3  — eligible label, genuine < genuine_eligible → needs_review
    Rule 3a — eligible label, academic merit text, no financial text → ineligible (belt-and-suspenders)
    Rule 3b — eligible label, no signals at all but financial_need high → needs_review
    Rule 4  — eligible label, financial_p < floor → ineligible
    Rule 5  — eligible label, all gates pass → eligible
    Rule 6  — ineligible label → ineligible
    Rule 7  — needs_review or unknown → needs_review
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decider_kit import Decider
from decider_kit.client import Decision

PATTERN_TAG = "#policy"

SCHOLARSHIP = "Need-Based Financial Aid Scholarship"
POLICY = (
    "The Need-Based Financial Aid Scholarship supports students who face a genuine, specific "
    "financial difficulty that affects their ability to pay tuition or to continue studying. A "
    "qualifying application describes a real, concrete situation - such as loss of a parent's "
    "income, an unpaid medical bill, a family emergency, or sudden loss of part-time work - with "
    "enough detail that a reviewer can tell what happened and why aid is needed. Generic requests "
    "for help (for example 'I need money for college'), reasons that describe academic merit or "
    "ambition rather than financial need, and reasons that are vague, templated, or clearly "
    "invented do not qualify. When a reason is borderline or too vague to judge, hold the "
    "application for human review rather than deciding automatically."
)

QUESTIONS = {
    "eligibility": Decider.choice(
        f"Does the applicant's reason match the {SCHOLARSHIP} policy?\nPolicy: {POLICY}",
        {
            "eligible": "a specific, genuine financial difficulty that clearly affects the "
                        "applicant's ability to pay tuition or continue studying, matching the policy",
            "needs_review": "the reason is borderline, too vague, or too short to judge with "
                            "confidence; hold it for a human reviewer",
            "ineligible": "the reason describes academic merit, ambition, a generic wish for "
                          "money, or otherwise does not match the financial-need policy",
        },
    ),
    "genuine": Decider.noul(
        "Is the applicant's reason specific and genuine, rather than vague, generic, or templated?",
        {"true": "it names a concrete situation with real, specific detail",
         "false": "it is generic, templated, or lacks concrete detail"},
    ),
    "financial_need": Decider.noul(
        "Does the applicant's reason describe a real financial hardship that affects their ability "
        "to pay tuition or continue studying?",
        {"true": "it describes loss of income, unpaid bills, a family emergency, or another "
                 "concrete financial difficulty that prevents the applicant from paying tuition",
         "false": "it describes academic achievement, ambition, a preference, a purchase, "
                  "or any reason other than a financial difficulty that affects tuition payment"},
    ),
}

# ---------------------------------------------------------------------------
# Deterministic policy check on the applicant's actual stated reason.
#
# The Strands Decider's financial_need score can be miscalibrated — it may score
# financial_need high for any reason that mentions "scholarship" in the state,
# even when the applicant's reason itself describes academic merit or a generic
# request with no financial hardship.  These functions cross-check the decider's
# financial_need score against the applicant's actual text, so a high score does
# not override a reason that does not establish financial hardship (requirement 3).
# ---------------------------------------------------------------------------

# Keywords in the applicant's reason that signal a real financial hardship.
# If none are present, the financial_need score from the decider is unreliable.
_FINANCIAL_HARDSHIP_SIGNALS = [
    "lost", "job", "income", "salary", "wage", "employ", "unemploy",
    "afford", "cannot pay", "can't pay", "unable to pay",
    "bill", "debt", "loan", "rent",
    "medical", "surgery", "hospital", "ill",
    "emergency",
    "savings", "expense",
    "orphan", "poor", "poverty",
    "flood", "disaster", "accident",
    "hardship", "difficulty",
    "part-time", "sponsor",
    "no longer", "no money",
    "tuition", "fees",
    "financial",
]

# Keywords that signal academic merit or achievement (the opposite of financial need).
_ACADEMIC_MERIT_SIGNALS = [
    "cgpa", "gpa", "grade", "marks", "score", "rank",
    "academic", "research", "internship", "merit", "achievement",
    "excellent", "top", "award", "recognition",
    "study abroad", "resume", "project",
    "brilliant", "outstanding", "distinction",
    "class representative",
]


def has_financial_hardship_signal(reason: str) -> bool:
    """Deterministic text check: does the reason mention a financial hardship?

    Returns True if any financial-hardship keyword appears in the lowercased reason.
    When this returns False, the decider's financial_need score is unreliable — the
    model may have been triggered by 'scholarship' in the state rather than by the
    applicant's actual text.
    """
    lower = (reason or "").lower()
    return any(sig in lower for sig in _FINANCIAL_HARDSHIP_SIGNALS)


def has_academic_merit_signal(reason: str) -> bool:
    """Deterministic text check: does the reason describe academic merit or achievement?"""
    lower = (reason or "").lower()
    return any(sig in lower for sig in _ACADEMIC_MERIT_SIGNALS)


def make_state(reason: str) -> str:
    """The text layout sent to the decider; agent/eval_cases.jsonl uses the same layout."""
    return f"Scholarship: {SCHOLARSHIP}\nApplicant's reason: {reason.strip()}"


@dataclass
class Verdict:
    action: str  # eligible | needs_review | ineligible
    eligibility_label: str
    confidence: float
    genuine: float
    financial_need: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    decider_ms: float = 0.0
    reason: str = ""

    def as_log(self) -> dict[str, Any]:
        return {
            "pattern": PATTERN_TAG,
            "action": self.action,
            "eligibility": self.eligibility_label,
            "confidence": round(self.confidence, 3),
            "genuine": round(self.genuine, 3),
            "financial_need": round(self.financial_need, 3),
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "decider_ms": round(self.decider_ms or 0),
        }


class ScholarshipPolicyCheck:
    """The #policy decision, wired before the agent's reply is written."""

    def __init__(self, decider: Decider, sure: float = 0.6,
                 genuine_floor: float = 0.4, genuine_eligible: float = 0.6,
                 financial_floor: float = 0.5):
        self.decider = decider
        self.sure = sure                          # below this top-label confidence → human review
        self.genuine_floor = genuine_floor        # below this genuine p_yes → human review
        self.genuine_eligible = genuine_eligible  # must exceed this to confirm eligible
                                                  # (stricter than genuine_floor; keeps generic
                                                  # requests — which score genuine near the floor —
                                                  # from reaching eligible even if financial_need
                                                  # and eligibility scores look high)
        self.financial_floor = financial_floor    # below this financial_need p_yes → not eligible
        self.log: list[dict[str, Any]] = []

    def check(self, reason: str, images: Any = None) -> Verdict:
        a = self.decider.ask(make_state(reason), QUESTIONS, images=images)
        v = self.decide(a, reason)
        self.log.append(v.as_log())
        return v

    def decide(self, a: dict[str, Decision], reason: str) -> Verdict:
        elig = a["eligibility"]
        gen = a["genuine"]
        fin = a["financial_need"]
        raw_label = elig.label or "needs_review"
        conf = elig.confidence or 0.0
        genuine_p = gen.p_yes or 0.0
        financial_p = fin.p_yes or 0.0

        # The probability mass on the *ineligible* option from the eligibility question.
        # When this is high (≥ 0.30), the decider itself is significantly uncertain about
        # eligibility, and the financial_need score must not be trusted to override it —
        # the model may score financial_need high for any reason mentioning scholarships.
        prob_ineligible = elig.probabilities.get("ineligible", 0.0)

        # Rule 0 — deterministic policy check on the applicant's actual text.
        # The scholarship policy explicitly excludes "reasons that describe academic
        # merit or ambition rather than financial need."  This check runs BEFORE the
        # confidence and genuine gates, because academic merit is a clear policy
        # disqualification regardless of how the decider scored the reason.
        # A reason that contains academic-merit signals (grades, CGPA, marks, awards…)
        # and contains NO financial-hardship signals is ineligible by policy.
        # This catches the case where the decider scores genuine low (making it vague)
        # but the text itself is unambiguously about academic achievement.
        if has_academic_merit_signal(reason) and not has_financial_hardship_signal(reason):
            action, label = "ineligible", "ineligible"

        # Rule 1 — low confidence or insufficient specificity → needs_review.
        # genuine_floor is a minimum gate: below it the reason is too vague to classify.
        elif conf < self.sure or genuine_p < self.genuine_floor:
            action, label = "needs_review", "needs_review"

        # Rule 2 — eligible claimed but the eligibility distribution shows significant
        # ineligible probability. The financial_need score is unreliable when the
        # eligibility question itself is split; escalate to needs_review for human review.
        elif raw_label == "eligible" and prob_ineligible >= 0.30:
            action, label = "needs_review", "needs_review"

        # Rule 3 — eligible label, but the reason is not specific enough to confirm
        # eligibility. genuine_eligible is stricter than genuine_floor: a vague or generic
        # reason ("I need money for college") may clear genuine_floor (avoiding needs_review)
        # but still fail genuine_eligible and land on needs_review rather than eligible.
        # This is the policy requirement: eligibility needs a real, concrete situation.
        elif raw_label == "eligible" and genuine_p < self.genuine_eligible:
            action, label = "needs_review", "needs_review"

        # Rule 3a — eligible claimed, but the reason text describes academic merit or
        # achievement with NO financial-hardship signal. The decider may score
        # financial_need high (triggered by "scholarship" in the state), but the
        # applicant's actual words do not establish financial hardship. Per the
        # policy: academic merit alone must never result in eligible.
        # This is the deterministic text-based check required by requirement 4.
        elif (raw_label == "eligible"
              and has_academic_merit_signal(reason)
              and not has_financial_hardship_signal(reason)):
            action, label = "ineligible", "ineligible"

        # Rule 3b — eligible claimed, the reason has no academic-merit signal AND no
        # financial-hardship signal in the text, but the decider scored financial_need
        # high. The financial_need score is unreliable (no text-based confirmation), and
        # without independent evidence of financial hardship the application is ambiguous.
        # Per the policy: ambiguous applications must receive needs_review.
        elif (raw_label == "eligible"
              and not has_academic_merit_signal(reason)
              and not has_financial_hardship_signal(reason)
              and financial_p >= self.financial_floor):
            action, label = "needs_review", "needs_review"

        # Rule 4 — eligible label, financial need not confirmed → ineligible by policy.
        # A reason that names a specific situation but describes academic merit or
        # a non-financial goal cannot be eligible for a need-based scholarship.
        elif raw_label == "eligible" and financial_p < self.financial_floor:
            action, label = "ineligible", "ineligible"

        # Rule 5 — eligible label with all criteria confirmed: high confidence, sufficient
        # specificity (genuine_p ≥ genuine_eligible), and financial hardship confirmed.
        elif raw_label == "eligible":
            action, label = "eligible", "eligible"

        # Rule 6 — ineligible label.
        elif raw_label == "ineligible":
            action, label = "ineligible", "ineligible"

        # Rule 7 — needs_review or unknown label.
        else:
            action, label = "needs_review", "needs_review"

        return Verdict(action, label, conf, genuine_p, financial_p,
                       dict(elig.probabilities), self.decider.last_latency_ms or 0,
                       reason.strip())
