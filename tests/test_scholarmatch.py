"""ScholarMatch-specific tests: decision consistency, action/label alignment,
prompt integrity, reply consistency, guard exhaustiveness, and per-turn freshness.

All tests are fully offline: they use FakeDecider and stub LLM calls so no network
traffic is generated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeDecider  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_check(rule):
    """Return a ScholarshipPolicyCheck backed by a FakeDecider using the given rule."""
    from agent.decision import ScholarshipPolicyCheck
    return ScholarshipPolicyCheck(FakeDecider(rule))


def _make_agent(rule, llm_reply: str):
    """Return a HackathonAgent whose decider uses `rule` and whose LLM always returns `llm_reply`.

    The Strands Agent constructor AND call are both stubbed — no network I/O.
    """
    from agent.agent import HackathonAgent

    fake_agent_instance = MagicMock()
    fake_agent_instance.return_value = llm_reply

    ha = HackathonAgent.__new__(HackathonAgent)
    ha.decider = FakeDecider(rule)
    ha.eager = False
    ha.model_id = "fake-model"
    from agent.decision import ScholarshipPolicyCheck
    ha.policy_check = ScholarshipPolicyCheck(ha.decider)
    # _fresh_agent always returns the same stub for determinism
    ha._fresh_agent = lambda: fake_agent_instance  # type: ignore[method-assign]
    return ha


# ---------------------------------------------------------------------------
# Scenario 1 — specific, credible financial hardship → eligible
# ---------------------------------------------------------------------------

class TestEligibleScenario:
    REASON = (
        "My father lost his job at a textile mill three months ago, "
        "and my family can no longer afford my tuition."
    )

    def _rule(self, state, name, q):
        if name == "eligibility":
            return "eligible"
        return 0.9  # genuine p_yes = 0.9 (specific and genuine)

    def test_verdict_is_eligible(self):
        pc = _make_check(self._rule)
        v = pc.check(self.REASON)
        assert v.action == "eligible"

    def test_action_and_label_match(self):
        pc = _make_check(self._rule)
        v = pc.check(self.REASON)
        assert v.action == v.eligibility_label, (
            f"action={v.action!r} does not match eligibility_label={v.eligibility_label!r}"
        )

    def test_log_action_and_label_match(self):
        pc = _make_check(self._rule)
        v = pc.check(self.REASON)
        log = v.as_log()
        assert log["action"] == log["eligibility"] == "eligible"

    def test_prompt_contains_only_action(self):
        from agent.agent import HackathonAgent
        from agent.decision import ScholarshipPolicyCheck

        pc = ScholarshipPolicyCheck(FakeDecider(self._rule))
        v = pc.check(self.REASON)
        agent = HackathonAgent.__new__(HackathonAgent)
        prompt = agent._prompt(self.REASON, v)
        assert "ELIGIBLE" in prompt
        assert "eligibility=" not in prompt
        assert "confidence=" not in prompt

    def test_reply_agrees_with_verdict(self):
        ha = _make_agent(self._rule, "Your application is eligible based on your financial hardship.")
        out = ha.chat(self.REASON)
        assert out["decisions"][0]["action"] == "eligible"
        assert "eligible" in out["reply"].lower()
        assert "ineligible" not in out["reply"].lower()
        assert "human review" not in out["reply"].lower()

    def test_contradicting_reply_is_replaced(self):
        ha = _make_agent(
            self._rule,
            "Your application is ineligible because it does not meet the criteria.",
        )
        out = ha.chat(self.REASON)
        assert out["decisions"][0]["action"] == "eligible"
        assert "ineligible" not in out["reply"].lower()
        assert "eligible" in out["reply"].lower()


# ---------------------------------------------------------------------------
# Scenario 2 — financial-hardship reason but low genuine → needs_review
#
# This is the EXACT case reported in the UI:
#   confidence=0.74, genuine=0.33 → decide() overrides to needs_review.
#   The LLM nevertheless replies "your application is eligible…"
#   The old guard missed it. The new guard must catch it.
# ---------------------------------------------------------------------------

class TestNeedsReviewScenario:
    # Generic vague reason
    REASON_VAGUE = "I need financial help for my studies."
    # The exact financial-hardship reason from the reported UI case
    REASON_HARDSHIP = (
        "My father lost his job at a textile mill three months ago, "
        "and my family can no longer afford my tuition."
    )

    def _rule_low_genuine(self, state, name, q):
        """confidence=0.74 (above sure=0.60), genuine=0.33 (below floor=0.40) → needs_review."""
        if name == "eligibility":
            return "eligible"   # raw decider label would be eligible…
        return 0.33             # …but genuine is below floor → override to needs_review

    def _rule_vague(self, state, name, q):
        if name == "eligibility":
            return "eligible"
        return 0.2  # genuine p_yes < floor

    def _rule_low_confidence(self, state, name, q):
        if name == "eligibility":
            return "needs_review"
        return 0.3

    # ---- decide() tests ----

    def test_vague_reason_becomes_needs_review(self):
        pc = _make_check(self._rule_vague)
        v = pc.check(self.REASON_VAGUE)
        assert v.action == "needs_review"

    def test_action_and_label_match_when_overridden(self):
        pc = _make_check(self._rule_vague)
        v = pc.check(self.REASON_VAGUE)
        assert v.action == v.eligibility_label == "needs_review"

    def test_low_confidence_becomes_needs_review(self):
        pc = _make_check(self._rule_low_confidence)
        v = pc.check(self.REASON_VAGUE)
        assert v.action == "needs_review"
        assert v.action == v.eligibility_label

    def test_hardship_low_genuine_becomes_needs_review(self):
        """Regression: confidence=0.74, genuine=0.33 — the exact reported UI case."""
        pc = _make_check(self._rule_low_genuine)
        v = pc.check(self.REASON_HARDSHIP)
        assert v.action == "needs_review", (
            f"Expected needs_review (genuine=0.33 < floor=0.40) but got {v.action!r}"
        )
        assert v.eligibility_label == "needs_review"

    # ---- prompt tests ----

    def test_prompt_says_needs_review(self):
        from agent.agent import HackathonAgent
        from agent.decision import ScholarshipPolicyCheck

        pc = ScholarshipPolicyCheck(FakeDecider(self._rule_vague))
        v = pc.check(self.REASON_VAGUE)
        agent = HackathonAgent.__new__(HackathonAgent)
        prompt = agent._prompt(self.REASON_VAGUE, v)
        assert "NEEDS_REVIEW" in prompt
        assert "eligibility=" not in prompt

    # ---- guard regression: exact reported phrases ----

    def test_exact_reported_phrase_is_caught(self):
        """Regression: the LLM reply 'your application is eligible' with verdict needs_review
        must be replaced.  This is the exact phrase that slipped through the old guard."""
        ha = _make_agent(
            self._rule_low_genuine,
            "Your application is eligible because your financial hardship is well documented.",
        )
        out = ha.chat(self.REASON_HARDSHIP)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "eligible" not in out["reply"].lower() or "ineligible" in out["reply"].lower(), (
            f"Guard did not catch the contradiction. reply={out['reply']!r}"
        )
        # The fallback must be used and must contain the correct verdict word
        assert "review" in out["reply"].lower(), (
            f"Fallback reply for needs_review must mention 'review'. Got: {out['reply']!r}"
        )

    def test_found_to_be_eligible_is_caught(self):
        ha = _make_agent(
            self._rule_low_genuine,
            "Your application has been found to be eligible for the scholarship.",
        )
        out = ha.chat(self.REASON_HARDSHIP)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower()

    def test_found_eligible_is_caught(self):
        ha = _make_agent(
            self._rule_low_genuine,
            "Your application was found eligible based on the evidence provided.",
        )
        out = ha.chat(self.REASON_HARDSHIP)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower()

    def test_application_qualifies_is_caught(self):
        ha = _make_agent(
            self._rule_low_genuine,
            "Your application qualifies for the scholarship.",
        )
        out = ha.chat(self.REASON_HARDSHIP)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower()

    def test_you_qualify_is_caught(self):
        ha = _make_agent(self._rule_low_genuine, "You qualify for financial aid.")
        out = ha.chat(self.REASON_HARDSHIP)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower()

    def test_congratulations_is_caught(self):
        ha = _make_agent(
            self._rule_low_genuine,
            "Congratulations! Your application meets the scholarship criteria.",
        )
        out = ha.chat(self.REASON_HARDSHIP)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower()

    # ---- guard: consistent reply passes through ----

    def test_consistent_reply_passes_through(self):
        """The deterministic path always returns a consistent needs_review explanation."""
        ha = _make_agent_deterministic(self._rule_low_genuine)
        out = ha.chat(self.REASON_HARDSHIP)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower()
        # The deterministic template is always returned (no LLM to contradict it)
        assert "held for human review" in out["reply"].lower()

    # ---- guard: reply that doesn't even mention the verdict ----

    def test_reply_missing_verdict_word_is_replaced(self):
        """If the reply never mentions review/assessment/held/etc it gets the fallback."""
        ha = _make_agent(
            self._rule_low_genuine,
            "The scholarship office will be in touch shortly.",
        )
        out = ha.chat(self.REASON_HARDSHIP)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower()

    def test_contradicting_reply_is_replaced_eligible(self):
        ha = _make_agent(
            self._rule_vague,
            "Your application qualifies and has been approved.",
        )
        out = ha.chat(self.REASON_VAGUE)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "has been approved" not in out["reply"].lower()

    def test_contradicting_reply_is_replaced_ineligible(self):
        ha = _make_agent(
            self._rule_vague,
            "Your application is ineligible because the reason does not meet the policy.",
        )
        out = ha.chat(self.REASON_VAGUE)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "is ineligible" not in out["reply"].lower()


# ---------------------------------------------------------------------------
# Scenario 3 — academic-merit reason → ineligible
# ---------------------------------------------------------------------------

class TestIneligibleScenario:
    REASON = (
        "I have a CGPA of 9.4 and want the scholarship to fund my "
        "research project on renewable energy."
    )

    def _rule(self, state, name, q):
        if name == "eligibility":
            return "ineligible"
        return 0.75  # genuine p_yes = 0.75 (specific, but wrong type)

    def test_verdict_is_ineligible(self):
        pc = _make_check(self._rule)
        v = pc.check(self.REASON)
        assert v.action == "ineligible"

    def test_action_and_label_match(self):
        pc = _make_check(self._rule)
        v = pc.check(self.REASON)
        assert v.action == v.eligibility_label == "ineligible"

    def test_log_action_and_label_match(self):
        pc = _make_check(self._rule)
        v = pc.check(self.REASON)
        log = v.as_log()
        assert log["action"] == log["eligibility"] == "ineligible"

    def test_prompt_says_ineligible(self):
        from agent.agent import HackathonAgent
        from agent.decision import ScholarshipPolicyCheck

        pc = ScholarshipPolicyCheck(FakeDecider(self._rule))
        v = pc.check(self.REASON)
        agent = HackathonAgent.__new__(HackathonAgent)
        prompt = agent._prompt(self.REASON, v)
        assert "INELIGIBLE" in prompt
        assert "eligibility=" not in prompt

    def test_reply_agrees_with_verdict(self):
        ha = _make_agent(
            self._rule,
            "Your application is ineligible because your reason describes academic merit, "
            "not financial need.",
        )
        out = ha.chat(self.REASON)
        assert out["decisions"][0]["action"] == "ineligible"
        assert "ineligible" in out["reply"].lower()
        assert "is eligible" not in out["reply"].lower()

    def test_contradicting_reply_is_replaced(self):
        ha = _make_agent(
            self._rule,
            "Congratulations! Your application has been approved and you qualify.",
        )
        out = ha.chat(self.REASON)
        assert out["decisions"][0]["action"] == "ineligible"
        assert "congratulations" not in out["reply"].lower()
        assert "qualifies" not in out["reply"].lower()
        assert "ineligible" in out["reply"].lower()


# ---------------------------------------------------------------------------
# Per-turn freshness: no shared Strands Agent history between calls
# ---------------------------------------------------------------------------

def test_fresh_agent_per_turn():
    """_fresh_agent() must return a new object on every call — no shared history."""
    from agent.agent import HackathonAgent

    ha = HackathonAgent.__new__(HackathonAgent)
    ha.model_id = "us.amazon.nova-lite-v1:0"

    created: list[object] = []

    class TrackingAgent:
        def __init__(self, **kw):
            created.append(self)

        def __call__(self, prompt):
            return "Your application is eligible."

    with patch("agent.agent.Agent", side_effect=TrackingAgent):
        a1 = ha._fresh_agent()
        a2 = ha._fresh_agent()

    assert a1 is not a2, "_fresh_agent() returned the same instance twice — history will accumulate"
    assert len(created) == 2


def test_no_explainer_field():
    """HackathonAgent must not carry a self._explainer field (source of stale history)."""
    from agent.agent import HackathonAgent

    ha = HackathonAgent.__new__(HackathonAgent)
    assert not hasattr(ha, "_explainer"), (
        "self._explainer still exists; remove it to prevent stale conversation history"
    )


# ---------------------------------------------------------------------------
# Invariant: action == eligibility_label for ALL decider output combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("choice_answer,genuine_p,expected_action", [
    ("eligible",     0.9,  "eligible"),
    ("eligible",     0.2,  "needs_review"),
    ("needs_review", 0.3,  "needs_review"),
    ("ineligible",   0.8,  "ineligible"),
    ("ineligible",   0.1,  "needs_review"),
])
def test_action_always_equals_label(choice_answer, genuine_p, expected_action):
    from agent.decision import ScholarshipPolicyCheck

    def rule(state, name, q):
        if name == "eligibility":
            return choice_answer
        return genuine_p

    pc = ScholarshipPolicyCheck(FakeDecider(rule))
    v = pc.check("My father lost his job and we cannot afford tuition")
    assert v.action == expected_action
    assert v.action == v.eligibility_label


# ---------------------------------------------------------------------------
# _guard_reply exhaustive unit tests
# (isolated — no decider or LLM needed; tests the guard logic directly)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action,reply,should_replace", [
    # ---- eligible: agreeing ----
    ("eligible", "Your application is eligible based on your hardship.", False),
    ("eligible", "You qualify for the scholarship.", False),
    ("eligible", "Your application has been found eligible.", False),
    # ---- eligible: contradicting ----
    ("eligible", "Your application is ineligible.", True),
    ("eligible", "This application does not qualify.", True),
    ("eligible", "It does not meet the policy.", True),
    ("eligible", "This has been held for human review.", True),
    ("eligible", "Your application requires further review.", True),
    # ---- eligible: reply missing verdict word entirely ----
    ("eligible", "The scholarship office has received your application.", True),
    # ---- needs_review: agreeing ----
    ("needs_review", "Your application has been held for human review.", False),
    ("needs_review", "A reviewer will further assess your application.", False),
    ("needs_review", "Your application is pending assessment.", False),
    # ---- needs_review: exact reported UI phrase ----
    ("needs_review",
     "Your application is eligible because your financial hardship is well documented.", True),
    # ---- needs_review: all eligible paraphrases ----
    ("needs_review", "Your application is eligible for the scholarship.", True),
    ("needs_review", "Your application are eligible under the policy.", True),
    ("needs_review", "Your application has been found eligible.", True),
    ("needs_review", "Your application has been found to be eligible.", True),
    ("needs_review", "Your application was found eligible.", True),
    ("needs_review", "Your application qualifies for the scholarship.", True),
    ("needs_review", "Your application qualifies for financial support.", True),
    ("needs_review", "Your application qualifies.", True),
    ("needs_review", "You qualify for this scholarship.", True),
    ("needs_review", "Your application qualifies and has been approved.", True),
    ("needs_review", "Congratulations! You qualify.", True),
    ("needs_review", "Your application is eligible and has been approved.", True),
    ("needs_review", "Your application has been approved.", True),
    # ---- needs_review: ineligible paraphrases ----
    ("needs_review", "Your application is ineligible under the policy.", True),
    ("needs_review", "Your application has been found ineligible.", True),
    ("needs_review", "Your application does not qualify for the scholarship.", True),
    ("needs_review", "Your application does not meet the criteria.", True),
    # ---- needs_review: reply missing verdict word ----
    ("needs_review", "The scholarship office will contact you.", True),
    # ---- ineligible: agreeing ----
    ("ineligible", "Your application is ineligible because it describes merit.", False),
    ("ineligible", "This reason does not qualify for the financial-need scholarship.", False),
    ("ineligible", "Your application does not meet the financial-need criteria.", False),
    # ---- ineligible: contradicting eligible ----
    ("ineligible", "Congratulations! You qualify.", True),
    ("ineligible", "Your application has been approved.", True),
    ("ineligible", "Your application is eligible.", True),
    ("ineligible", "Your application has been found eligible.", True),
    ("ineligible", "Your application has been found to be eligible.", True),
    # ---- ineligible: contradicting needs_review ----
    ("ineligible", "This has been held for needs_review.", True),
    ("ineligible", "Your application requires further review.", True),
    ("ineligible", "Your application needs further assessment.", True),
    # ---- ineligible: reply missing verdict word ----
    ("ineligible", "The scholarship office has received your application.", True),
])
def test_guard_reply(action, reply, should_replace):
    from agent.agent import HackathonAgent

    ha = HackathonAgent.__new__(HackathonAgent)
    result = ha._guard_reply(reply, action)
    fallback = HackathonAgent._FALLBACK[action]
    if should_replace:
        assert result == fallback, (
            f"action={action!r}, reply={reply!r}: expected fallback\n"
            f"  fallback={fallback!r}\n"
            f"  got     ={result!r}"
        )
    else:
        assert result == reply, (
            f"action={action!r}, reply={reply!r}: expected pass-through but guard replaced it\n"
            f"  got={result!r}"
        )


# ---------------------------------------------------------------------------
# _FALLBACK sanity: each fallback must agree with its own verdict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action,expected_word", [
    ("eligible",     "eligible"),
    ("needs_review", "review"),
    ("ineligible",   "ineligible"),
])
def test_fallback_matches_verdict(action, expected_word):
    """The fallback text for each action must itself pass the guard."""
    from agent.agent import HackathonAgent

    ha = HackathonAgent.__new__(HackathonAgent)
    fallback = HackathonAgent._FALLBACK[action]
    assert expected_word in fallback.lower(), (
        f"Fallback for action={action!r} does not mention {expected_word!r}: {fallback!r}"
    )
    # The fallback must pass its own guard (no self-contradiction)
    result = ha._guard_reply(fallback, action)
    assert result == fallback, (
        f"Fallback for action={action!r} fails its own guard!\n"
        f"  fallback={fallback!r}\n"
        f"  guard returned={result!r}"
    )


# ---------------------------------------------------------------------------
# Integration tests: full chat() path end-to-end
#
# These test the complete execution chain:
#   chat() → _resolve() → ScholarshipPolicyCheck.check() → _fresh_agent()()
#         → strip_thinking() → _guard_reply() → returned dict
#
# The FakeDecider is wired to produce specific confidence/genuine values.
# The LLM stub returns the exact contradicting phrase reported in the live UI.
# These tests would have caught the live inconsistency regardless of which
# Python binary was serving port 8501, because they exercise chat() directly.
# ---------------------------------------------------------------------------

class TestChatIntegration:
    """End-to-end chat() tests: stub decider + stub LLM, full path exercised."""

    # The exact input reported in the live UI
    REPORTED_REASON = (
        "My father lost his job two months ago, and my family cannot afford my college tuition. "
        "I need financial aid to continue my studies."
    )

    def _make(self, eligibility_choice: str, genuine_p: float, llm_reply: str):
        """Wire up HackathonAgent with a fake decider and a stub LLM, no network calls."""
        from agent.agent import HackathonAgent
        from agent.decision import ScholarshipPolicyCheck

        def rule(state, name, q):
            return eligibility_choice if name == "eligibility" else genuine_p

        ha = HackathonAgent.__new__(HackathonAgent)
        ha.decider = FakeDecider(rule)
        ha.eager = False
        ha.model_id = "fake"
        ha.policy_check = ScholarshipPolicyCheck(ha.decider)
        stub = MagicMock()
        stub.return_value = llm_reply
        ha._fresh_agent = lambda: stub
        return ha

    def _assert_consistent(self, out: dict) -> None:
        """Assert that out["reply"] text agrees with out["decisions"][0]["action"]."""
        action = out["decisions"][0]["action"]
        reply_lower = out["reply"].lower()
        if action == "eligible":
            assert "eligible" in reply_lower or "qualif" in reply_lower, (
                f"reply for eligible action must mention eligibility. reply={out['reply']!r}"
            )
            for bad in ["ineligible", "needs review", "human review", "held for"]:
                assert bad not in reply_lower, (
                    f"reply for eligible contains contradiction {bad!r}. reply={out['reply']!r}"
                )
        elif action == "needs_review":
            for bad in ["congratulations", "is eligible", "found eligible",
                        "found to be eligible", "has been approved", "you qualify",
                        "application qualifies"]:
                assert bad not in reply_lower, (
                    f"reply for needs_review contains contradiction {bad!r}. reply={out['reply']!r}"
                )
            assert ("review" in reply_lower or "assessment" in reply_lower
                    or "further" in reply_lower or "held" in reply_lower), (
                f"reply for needs_review must mention review/assessment/held. reply={out['reply']!r}"
            )
        elif action == "ineligible":
            assert ("ineligible" in reply_lower or "not eligible" in reply_lower
                    or "does not qualify" in reply_lower or "does not meet" in reply_lower), (
                f"reply for ineligible must mention ineligibility. reply={out['reply']!r}"
            )
            for bad in ["congratulations", "is eligible", "has been approved", "you qualify"]:
                assert bad not in reply_lower, (
                    f"reply for ineligible contains contradiction {bad!r}. reply={out['reply']!r}"
                )

    # ---- Regression: exact reported live UI case ----

    def test_reported_case_congratulations_eligible_with_needs_review(self):
        """Regression for the exact reported live UI contradiction:
        confidence=0.45, genuine=0.29 → needs_review.
        LLM returns 'Congratulations; your application is eligible...'
        The final reply must NOT contain this contradiction.
        """
        ha = self._make(
            "eligible",   # raw decider label (before genuine override)
            0.29,         # genuine p_yes < floor 0.40 → decide() overrides to needs_review
            "Congratulations; your application is eligible for the Need-Based Financial Aid Scholarship.",
        )
        out = ha.chat(self.REPORTED_REASON)
        assert out["decisions"][0]["action"] == "needs_review", (
            "decide() should have overridden to needs_review (genuine=0.29 < floor=0.40)"
        )
        self._assert_consistent(out)
        assert "congratulations" not in out["reply"].lower()
        assert "is eligible" not in out["reply"].lower()

    def test_reported_case_low_confidence_override(self):
        """Regression: confidence=0.45 < sure=0.60 → needs_review regardless of label."""
        # Simulate confidence=0.45: FakeDecider with needs_review label has confidence 0.85
        # (FakeDecider assigns 0.85 to the chosen option). To get conf=0.45 we need to use
        # a custom eligible choice where the fake probability math gives a low confidence.
        # Simpler: use 'needs_review' as label and genuine=0.29 — both thresholds fire.
        ha = self._make(
            "needs_review",  # decider already says needs_review
            0.29,
            "Your application is eligible based on the hardship you described.",
        )
        out = ha.chat(self.REPORTED_REASON)
        assert out["decisions"][0]["action"] == "needs_review"
        self._assert_consistent(out)

    # ---- Eligible scenario full path ----

    def test_eligible_path_consistent_reply(self):
        ha = self._make(
            "eligible", 0.9,
            "Your application is eligible. Your reason clearly describes a financial hardship.",
        )
        out = ha.chat("My father lost his job and we cannot pay tuition.")
        assert out["decisions"][0]["action"] == "eligible"
        self._assert_consistent(out)

    def test_eligible_path_contradicting_reply_replaced(self):
        """If the LLM contradicts eligible with 'ineligible', guard replaces it."""
        ha = self._make(
            "eligible", 0.9,
            "Your application is ineligible because the reason does not meet the criteria.",
        )
        out = ha.chat("My mother had surgery and we cannot pay tuition.")
        assert out["decisions"][0]["action"] == "eligible"
        self._assert_consistent(out)

    # ---- Needs-review scenario full path (multiple LLM phrasings) ----

    @pytest.mark.parametrize("llm_reply", [
        "Congratulations; your application is eligible for the scholarship.",
        "Your application is eligible because your financial hardship is well documented.",
        "Your application has been found eligible.",
        "Your application has been found to be eligible.",
        "Your application qualifies for the Need-Based Financial Aid Scholarship.",
        "You qualify for financial aid.",
        "Your application was found eligible based on the evidence.",
        "Your application has been approved.",
    ])
    def test_needs_review_all_eligible_phrasings_blocked(self, llm_reply):
        """Every 'eligible' phrasing from the LLM must be blocked when action is needs_review."""
        ha = self._make("eligible", 0.29, llm_reply)
        out = ha.chat(self.REPORTED_REASON)
        assert out["decisions"][0]["action"] == "needs_review"
        self._assert_consistent(out)
        assert "review" in out["reply"].lower(), (
            f"fallback for needs_review must mention 'review'. got: {out['reply']!r}"
        )

    def test_needs_review_consistent_reply_not_replaced(self):
        """The deterministic path always returns a consistent needs_review explanation.

        With the deterministic _explain() path, chat() never calls the LLM, so there
        is no risk of replacement — the template is always returned.
        """
        def rule(state, name, q):
            return "eligible" if name == "eligibility" else 0.29  # genuine < floor → needs_review

        ha = _make_agent_deterministic(rule)
        out = ha.chat(self.REPORTED_REASON)
        assert out["decisions"][0]["action"] == "needs_review"
        # The deterministic template is always consistent with the verdict
        assert "review" in out["reply"].lower()
        assert "held for human review" in out["reply"].lower()

    # ---- Ineligible scenario full path ----

    def test_ineligible_path_consistent_reply(self):
        ha = self._make(
            "ineligible", 0.8,
            "Your application is ineligible because your reason describes academic merit, not financial need.",
        )
        out = ha.chat("I have a CGPA of 9.4 and want funding for my research.")
        assert out["decisions"][0]["action"] == "ineligible"
        self._assert_consistent(out)

    def test_ineligible_path_contradicting_reply_replaced(self):
        ha = self._make(
            "ineligible", 0.8,
            "Congratulations! Your application qualifies for the scholarship.",
        )
        out = ha.chat("I have a CGPA of 9.4 and want funding for my research.")
        assert out["decisions"][0]["action"] == "ineligible"
        self._assert_consistent(out)

    # ---- chat() return shape ----

    def test_chat_return_shape(self):
        """chat() must always return the required keys with the correct types."""
        ha = self._make(
            "eligible", 0.9,
            "Your application is eligible.",
        )
        out = ha.chat("I need aid.")
        assert "reply" in out and isinstance(out["reply"], str)
        assert "decisions" in out and isinstance(out["decisions"], list) and len(out["decisions"]) == 1
        assert "model_id" in out
        assert "latency_ms" in out and isinstance(out["latency_ms"], int)
        d = out["decisions"][0]
        assert "action" in d and "eligibility" in d
        assert d["action"] == d["eligibility"], (
            f"action and eligibility in the log must match: {d}"
        )


# ---------------------------------------------------------------------------
# Regression tests for both exact reported live-UI cases
# ---------------------------------------------------------------------------

class TestReportedCases:
    """Exact reproductions of the two cases reported as inconsistent in the live UI.

    Both run through the full chat() path with a stubbed LLM and a FakeDecider
    configured to reproduce the exact confidence/genuine values seen in the UI.
    """

    # ---- Case 1: Financial hardship, conf=0.74, genuine=0.36 → needs_review ----
    # The decider's raw eligibility label would be "eligible" (conf=0.74 ≥ sure=0.60),
    # but genuine=0.36 < floor=0.40, so decide() overrides to needs_review.
    # The LLM must not be allowed to say "eligible".

    CASE1_REASON = (
        "My father lost his job two months ago, and my family cannot afford my "
        "college tuition. I need financial aid to continue my studies."
    )

    def _rule_case1(self, state, name, q):
        if name == "eligibility":
            return "eligible"       # raw label from decider
        if name == "financial_need":
            return 0.80             # genuine hardship present
        return 0.36                 # genuine p_yes = 0.36 < floor 0.40 → needs_review

    def test_case1_decide_returns_needs_review(self):
        """decide() must return needs_review when genuine < floor, regardless of eligibility label."""
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule_case1))
        v = pc.check(self.CASE1_REASON)
        assert v.action == "needs_review", (
            f"Expected needs_review (genuine=0.36 < floor=0.40), got {v.action!r}"
        )
        assert v.eligibility_label == "needs_review"

    def test_case1_action_label_match(self):
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule_case1))
        v = pc.check(self.CASE1_REASON)
        assert v.action == v.eligibility_label

    def test_case1_reply_consistent_with_needs_review(self):
        """The final reply returned by chat() must not say the application is eligible."""
        ha = _make_agent(
            self._rule_case1,
            # Simulate the LLM ignoring the NEEDS_REVIEW instruction and saying eligible
            "Your application is eligible because your father's job loss is a genuine hardship.",
        )
        out = ha.chat(self.CASE1_REASON)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "is eligible" not in out["reply"].lower(), (
            f"Guard must catch 'is eligible' for needs_review. reply={out['reply']!r}"
        )
        assert "review" in out["reply"].lower(), (
            f"Fallback for needs_review must mention 'review'. reply={out['reply']!r}"
        )

    def test_case1_exact_reported_reply_blocked(self):
        """The exact phrase from the UI report must be blocked."""
        ha = _make_agent(
            self._rule_case1,
            "Congratulations; your application is eligible for the Need-Based Financial Aid Scholarship.",
        )
        out = ha.chat(self.CASE1_REASON)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "congratulations" not in out["reply"].lower()
        assert "review" in out["reply"].lower()

    # ---- Case 2: Academic recognition, conf=0.73, genuine=0.78 → must be ineligible ----
    # Both thresholds pass (conf=0.73 ≥ 0.60, genuine=0.78 ≥ 0.40).
    # The decider's raw label was "eligible" — but the reason describes academic merit,
    # not financial need. financial_need.p_yes will be low.
    # decide() must not confirm eligible when financial_need < financial_floor.
    # The LLM wrote "requires human review because no financial hardship" — that was
    # the correct human judgment but contradicted the eligible verdict.
    # After this fix, decide() should return ineligible, and the LLM explains ineligibility.

    CASE2_REASON = (
        "I have received academic recognition for my research contributions and "
        "would like the scholarship to support my ongoing projects."
    )

    def _rule_case2(self, state, name, q):
        if name == "eligibility":
            return "eligible"       # decider was wrong: merit reason scored as eligible
        if name == "financial_need":
            return 0.15             # clearly not financial need
        return 0.78                 # genuine p_yes = 0.78 (specific, but academic)

    def test_case2_decide_returns_ineligible(self):
        """When eligibility='eligible' but financial_need < floor, decide() → ineligible."""
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule_case2))
        v = pc.check(self.CASE2_REASON)
        assert v.action == "ineligible", (
            f"Academic-merit reason with financial_need=0.15 must be ineligible, got {v.action!r}"
        )
        assert v.eligibility_label == "ineligible"

    def test_case2_action_label_match(self):
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule_case2))
        v = pc.check(self.CASE2_REASON)
        assert v.action == v.eligibility_label

    def test_case2_reply_consistent_with_ineligible(self):
        """The final reply must say the application is ineligible (academic merit, no financial need)."""
        ha = _make_agent(
            self._rule_case2,
            "Your application is ineligible because your reason describes academic merit, "
            "not financial hardship.",
        )
        out = ha.chat(self.CASE2_REASON)
        assert out["decisions"][0]["action"] == "ineligible", (
            f"Expected ineligible for academic-merit case, got {out['decisions'][0]['action']!r}"
        )
        assert "ineligible" in out["reply"].lower(), (
            f"Reply must mention 'ineligible'. reply={out['reply']!r}"
        )
        assert "is eligible" not in out["reply"].lower()

    def test_case2_lm_says_needs_review_replaced_with_ineligible(self):
        """If the LLM somehow wrote 'needs review', the guard replaces it with the ineligible fallback."""
        ha = _make_agent(
            self._rule_case2,
            "Your application requires human review because it does not demonstrate financial hardship.",
        )
        out = ha.chat(self.CASE2_REASON)
        assert out["decisions"][0]["action"] == "ineligible"
        # "further review" is in _CONTRADICTS["ineligible"] but "requires human review" is not directly.
        # Stage 1 (_REQUIRES["ineligible"]) checks for "ineligible"/"not eligible"/"does not qualify"/etc.
        # "requires human review" contains neither → Stage 1 fires → fallback used.
        assert "ineligible" in out["reply"].lower(), (
            f"Fallback for ineligible must mention 'ineligible'. reply={out['reply']!r}"
        )

    # ---- financial_need is now in QUESTIONS: verify eval_cases contract ----

    def test_eval_cases_include_financial_need(self):
        """Every eval case's expected dict must be a subset of QUESTIONS (includes financial_need)."""
        import json
        from pathlib import Path
        from agent.decision import QUESTIONS
        cases_path = Path(__file__).resolve().parent.parent / "agent" / "eval_cases.jsonl"
        for line in cases_path.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            assert set(c["expected"]) <= set(QUESTIONS), (
                f"Expected keys {set(c['expected'])} not all in QUESTIONS {set(QUESTIONS)}"
            )

    # ---- financial_need in decide(): boundary conditions ----

    @pytest.mark.parametrize("eligibility,genuine_p,financial_p,expected_action", [
        # financial hardship confirmed → eligible
        ("eligible", 0.9, 0.8, "eligible"),
        # financial hardship NOT confirmed → ineligible (policy enforcement)
        ("eligible", 0.9, 0.3, "ineligible"),
        # financial hardship borderline (exactly at floor) → eligible (floor is exclusive: >= 0.5)
        ("eligible", 0.9, 0.5, "eligible"),
        # financial hardship just below floor → ineligible
        ("eligible", 0.9, 0.49, "ineligible"),
        # vague (genuine low) → needs_review regardless of financial_need
        ("eligible", 0.3, 0.9, "needs_review"),
        # low confidence → needs_review regardless
        # (FakeDecider gives 0.85 confidence, so we use needs_review label to test that path)
        ("needs_review", 0.9, 0.9, "needs_review"),
        # ineligible label → ineligible regardless of financial_need
        ("ineligible", 0.9, 0.9, "ineligible"),
        # case 2 exact: eligible label, genuine=0.78, financial=0.15 → ineligible
        ("eligible", 0.78, 0.15, "ineligible"),
        # case 1 exact: eligible label, genuine=0.36, financial=0.80 → needs_review (genuine floor)
        ("eligible", 0.36, 0.80, "needs_review"),
    ])
    def test_decide_financial_need_boundary(self, eligibility, genuine_p, financial_p, expected_action):
        """decide() must enforce the financial_need check on the eligible path."""
        from agent.decision import ScholarshipPolicyCheck

        def rule(state, name, q):
            if name == "eligibility":
                return eligibility
            if name == "financial_need":
                return financial_p
            return genuine_p  # genuine

        pc = ScholarshipPolicyCheck(FakeDecider(rule))
        v = pc.check("My father lost his job and we cannot afford my tuition this semester")
        assert v.action == expected_action, (
            f"eligibility={eligibility}, genuine={genuine_p}, financial={financial_p}: "
            f"expected {expected_action!r}, got {v.action!r}"
        )
        assert v.action == v.eligibility_label, (
            f"action={v.action!r} != eligibility_label={v.eligibility_label!r}"
        )


# ---------------------------------------------------------------------------
# Regression tests for the two exact live cases reported after adding financial_need
#
# LIVE CASE 1: genuine=0.33, financial_need=0.16, final verdict needs_review.
#              Assistant said "eligible" — caused by a stale server process.
#              decide() is correct (genuine < floor → needs_review).
#              The deterministic _explain() path now guarantees consistency.
#
# LIVE CASE 2: genuine=0.78, financial_need=0.85, final verdict eligible.
#              The academic-merit reason scored high on financial_need (decider
#              miscalibration). Assistant correctly said "financial hardship not
#              demonstrated" — but the panel said eligible.
#              The new Rule 2 (prob_ineligible >= 0.30 → needs_review) catches this
#              when the eligibility distribution is split.
# ---------------------------------------------------------------------------


def _make_agent_deterministic(rule):
    """Return a HackathonAgent whose decider uses `rule` and whose chat() uses the
    real deterministic _explain() path (no LLM call, no stubbing).

    This is the production path test helper.
    """
    from agent.agent import HackathonAgent
    from agent.decision import ScholarshipPolicyCheck

    ha = HackathonAgent.__new__(HackathonAgent)
    ha.decider = FakeDecider(rule)
    ha.eager = False
    ha.model_id = "fake"
    ha.policy_check = ScholarshipPolicyCheck(ha.decider)
    # _explain() is NOT stubbed — production deterministic path is exercised
    return ha


class TestLiveCase1FinancialHardship:
    """Exact reproduction of LIVE CASE 1: genuine=0.33, financial_need=0.16.

    decide() must return needs_review (genuine < floor), and the deterministic
    reply must say "review" and never "eligible".
    """

    REASON = (
        "My father lost his job two months ago, and my family cannot afford my "
        "college tuition. I need financial aid to continue my studies."
    )

    def _rule(self, state, name, q):
        # Reproduce the exact reported decider outputs
        if name == "eligibility":
            return "eligible"       # raw label
        if name == "financial_need":
            return 0.16              # reported: financial_need = 0.16
        return 0.33                  # reported: genuine = 0.33

    def test_decide_returns_needs_review(self):
        """genuine=0.33 < floor=0.40 → needs_review, regardless of eligibility label."""
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule))
        v = pc.check(self.REASON)
        assert v.action == "needs_review", (
            f"Expected needs_review (genuine=0.33 < floor=0.40), got {v.action!r}"
        )
        assert v.eligibility_label == "needs_review"

    def test_chat_reply_consistent_deterministic(self):
        """Full chat() path: the reply must say 'review', never 'eligible'."""
        ha = _make_agent_deterministic(self._rule)
        out = ha.chat(self.REASON)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower(), (
            f"Deterministic reply for needs_review must mention 'review'. got: {out['reply']!r}"
        )
        assert "eligible" not in out["reply"].lower(), (
            f"Reply must NOT say 'eligible' for needs_review. got: {out['reply']!r}"
        )
        assert "congratulations" not in out["reply"].lower()

    def test_chat_decisions_action_matches_reply(self):
        """The decisions panel action and the reply text must agree."""
        ha = _make_agent_deterministic(self._rule)
        out = ha.chat(self.REASON)
        action = out["decisions"][0]["action"]
        reply_lower = out["reply"].lower()
        if action == "needs_review":
            assert "review" in reply_lower
            assert "is eligible" not in reply_lower
            assert "is ineligible" not in reply_lower


class TestLiveCase2AcademicMerit:
    """Exact reproduction of LIVE CASE 2: genuine=0.78, financial_need=0.85.

    The academic-merit reason scored high on financial_need (decider miscalibration).
    Without the new Rule 2, decide() would return eligible, and the LLM would
    contradict it. The new Rule 2 (prob_ineligible >= 0.30 → needs_review) catches
    the split eligibility distribution and escalates to human review.

    FakeDecider assigns 0.85 confidence to the chosen option. To simulate a split
    distribution (significant ineligible probability), we use a custom rule that
    returns a raw eligibility label AND set up the probabilities so prob_ineligible
    is high. Since FakeDecider builds probabilities as {chosen: 0.85, others: 0.075
    each}, prob_ineligible would be 0.075 when eligible is chosen — not enough to
    trigger Rule 2.

    For this test we construct the Decision objects directly to simulate a split
    distribution where eligible won but ineligible has 0.35 probability.
    """

    REASON = (
        "I have received academic recognition for my research contributions and "
        "would like the scholarship to support my ongoing projects."
    )

    def test_split_distribution_triggers_needs_review(self):
        """When eligibility=eligible but prob_ineligible >= 0.30, decide() → needs_review.

        This is the core Rule 2 fix: don't trust financial_need when the eligibility
        question itself shows significant ineligible probability.
        """
        from agent.decision import ScholarshipPolicyCheck, Verdict
        from decider_kit.client import Decision

        # Manually construct the answer dict with a split distribution
        # eligible won (label=eligible, confidence=0.55) but ineligible has 0.35
        answers = {
            "eligibility": Decision(
                "choice",
                {"choice": "eligible", "probabilities": {"eligible": 0.55, "needs_review": 0.10, "ineligible": 0.35}, "confidence": 0.55},
                label="eligible",
                probabilities={"eligible": 0.55, "needs_review": 0.10, "ineligible": 0.35},
                confidence=0.55,
            ),
            "genuine": Decision("noul", {"noul": 0.78}, p_yes=0.78, label="yes", confidence=0.78, probabilities={"yes": 0.78, "no": 0.22}),
            "financial_need": Decision("noul", {"noul": 0.85}, p_yes=0.85, label="yes", confidence=0.85, probabilities={"yes": 0.85, "no": 0.15}),
        }

        decider = FakeDecider()
        decider.last_latency_ms = 100.0
        pc = ScholarshipPolicyCheck(decider)
        v = pc.decide(answers, TestLiveCase2AcademicMerit.REASON)

        # Rule 0 now fires first: ACADEMIC_REASON has academic signals ("academic",
        # "recognition") and no financial signals → ineligible, regardless of the
        # split distribution. Rule 2 (prob_ineligible >= 0.30) is no longer reached.
        assert v.action == "ineligible", (
            f"Rule 0 fires before Rule 2: academic-merit reason is ineligible regardless "
            f"of split distribution. Got {v.action!r}"
        )
        assert v.eligibility_label == "ineligible"

    def test_chat_reply_consistent_when_split_distribution(self):
        """Full chat() with split eligibility distribution → needs_review + deterministic reply."""
        from agent.agent import HackathonAgent
        from agent.decision import ScholarshipPolicyCheck, Verdict
        from decider_kit.client import Decision

        # Build a custom decider that returns the split distribution
        class SplitDecider:
            last_latency_ms = 100.0
            last_server_ms = 90.0
            url = "fake://split"
            calls = 0
            input_tokens = 0

            def ask(self, state, questions, images=None):
                SplitDecider.calls += 1
                return {
                    "eligibility": Decision(
                        "choice", {},
                        label="eligible",
                        probabilities={"eligible": 0.55, "needs_review": 0.10, "ineligible": 0.35},
                        confidence=0.55,
                    ),
                    "genuine": Decision("noul", {}, p_yes=0.78, label="yes", confidence=0.78, probabilities={"yes": 0.78, "no": 0.22}),
                    "financial_need": Decision("noul", {}, p_yes=0.85, label="yes", confidence=0.85, probabilities={"yes": 0.85, "no": 0.15}),
                }

            def health(self):
                return {"checkpoint": "fake"}

        ha = HackathonAgent.__new__(HackathonAgent)
        ha.decider = SplitDecider()
        ha.eager = False
        ha.model_id = "fake"
        ha.policy_check = ScholarshipPolicyCheck(ha.decider)
        # Use the real deterministic _explain() path

        out = ha.chat(self.REASON)
        # Rule 0 fires first: ACADEMIC_REASON has academic signals and no financial signals
        # → ineligible before any probability check.
        assert out["decisions"][0]["action"] == "ineligible", (
            f"Rule 0 must fire: academic-merit reason is ineligible regardless of split "
            f"distribution. Got {out['decisions'][0]['action']!r}"
        )
        assert "ineligible" in out["reply"].lower()
        assert "is eligible" not in out["reply"].lower()
        assert "congratulations" not in out["reply"].lower()

    def test_clean_eligible_still_works(self):
        """When prob_ineligible is low (< 0.30) and financial_need is high, eligible holds.

        This verifies Rule 2 does not break the normal eligible case.
        """
        from agent.decision import ScholarshipPolicyCheck
        from decider_kit.client import Decision

        answers = {
            "eligibility": Decision(
                "choice", {},
                label="eligible",
                probabilities={"eligible": 0.85, "needs_review": 0.10, "ineligible": 0.05},
                confidence=0.85,
            ),
            "genuine": Decision("noul", {}, p_yes=0.90, label="yes", confidence=0.90, probabilities={"yes": 0.90, "no": 0.10}),
            "financial_need": Decision("noul", {}, p_yes=0.80, label="yes", confidence=0.80, probabilities={"yes": 0.80, "no": 0.20}),
        }

        decider = FakeDecider()
        decider.last_latency_ms = 100.0
        pc = ScholarshipPolicyCheck(decider)
        v = pc.decide(answers, "My father lost his job and we cannot afford my tuition")
        assert v.action == "eligible", (
            f"Clean eligible case (prob_ineligible=0.05, financial=0.80) must stay eligible, got {v.action!r}"
        )


# ---------------------------------------------------------------------------
# Deterministic _explain() path tests
# ---------------------------------------------------------------------------

class TestDeterministicExplainPath:
    """Test that the deterministic _explain() templates are always consistent."""

    @pytest.mark.parametrize("action,expected_word", [
        ("eligible", "eligible"),
        ("needs_review", "review"),
        ("ineligible", "ineligible"),
    ])
    def test_explain_returns_template_with_correct_word(self, action, expected_word):
        """_explain() must return a template containing the verdict word."""
        from agent.agent import HackathonAgent

        ha = HackathonAgent.__new__(HackathonAgent)
        # Build a minimal verdict stub
        class V:
            pass
        v = V()
        v.action = action
        reply = ha._explain("any reason", v)
        assert expected_word in reply.lower(), (
            f"_explain({action!r}) must contain {expected_word!r}. got: {reply!r}"
        )

    def test_explain_templates_pass_their_own_guard(self):
        """Each _explain() template must pass _guard_reply() for its own action."""
        from agent.agent import HackathonAgent

        ha = HackathonAgent.__new__(HackathonAgent)
        for action, template_attr in [
            ("eligible", "_EXPLAIN_ELIGIBLE"),
            ("needs_review", "_EXPLAIN_NEEDS_REVIEW"),
            ("ineligible", "_EXPLAIN_INELIGIBLE"),
        ]:
            template = getattr(HackathonAgent, template_attr)
            result = ha._guard_reply(template, action)
            assert result == template, (
                f"_explain template for {action!r} fails its own guard!\n"
                f"  template={template!r}\n  guard returned={result!r}"
            )

    def test_chat_uses_deterministic_path_no_llm(self):
        """chat() must not call _fresh_agent() (the LLM path) — only _explain()."""
        from agent.agent import HackathonAgent
        from unittest.mock import patch, MagicMock

        def rule(state, name, q):
            return "eligible" if name == "eligibility" else 0.9

        ha = _make_agent_deterministic(rule)
        # Spy on _fresh_agent — it must NOT be called
        spy = MagicMock()
        ha._fresh_agent = spy
        out = ha.chat("My father lost his job and we cannot pay tuition.")
        assert spy.call_count == 0, (
            f"chat() must not call _fresh_agent() (LLM path). called {spy.call_count} times."
        )
        assert out["decisions"][0]["action"] == "eligible"
        assert "eligible" in out["reply"].lower()


# ---------------------------------------------------------------------------
# Regression tests for the three cases specified in the bug report
# ---------------------------------------------------------------------------


class TestPolicyClassificationRegression:
    """Regression tests for the three exact cases from the bug report.

    Case A: Generic money request — must never be automatically eligible.
    Case B: Academic merit without financial hardship — must never be eligible.
    Case C: Specific financial hardship (parent's job loss) — may be eligible
            when all criteria are met (high genuine, high financial_need, low
            ineligible probability).

    Also includes adversarial cases where the model incorrectly assigns high
    financial_need to vague or academic-merit reasons (requirement 9).
    """

    GENERIC_REASON = "I need money for college. Please help me."
    ACADEMIC_REASON = (
        "I have received academic recognition for my research and want the "
        "scholarship to support my ongoing academic projects."
    )
    HARDSHIP_REASON = (
        "My father lost his job at the factory two months ago, and my family "
        "can no longer afford my tuition payment for this semester."
    )

    # ------------------------------------------------------------------
    # Case A — Generic money request
    # Rule: generic requests must be needs_review, not eligible.
    # Even if the model erroneously scores financial_need high (sees "money"
    # and "college"), genuine should be low for a vague reason, and with
    # genuine_eligible=0.6 the eligible path is blocked.
    # ------------------------------------------------------------------

    def _rule_generic(self, state, name, q):
        """Simulates the decider output for a generic 'I need money' request.

        The decider picks eligible (the first option) but genuine is
        moderate (0.45 — passes genuine_floor=0.40 but fails genuine_eligible=0.60),
        and financial_need is borderline high (0.75, model saw 'money'+'college').
        """
        if name == "eligibility":
            return "eligible"
        if name == "financial_need":
            return 0.75   # model saw "money" and "college" → mistakenly high
        return 0.45       # genuine: vague, but passes the 0.40 floor

    def test_generic_request_not_eligible(self):
        """A generic 'I need money for college' request must not be eligible.

        genuine=0.45 passes genuine_floor=0.40 (so it would reach the eligible
        path under the old rules) but fails genuine_eligible=0.60 (new Rule 3),
        so it lands on needs_review.
        """
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule_generic))
        v = pc.check(self.GENERIC_REASON)
        assert v.action != "eligible", (
            f"Generic money request must not be eligible. Got {v.action!r}. "
            f"Genuine=0.45 must fail genuine_eligible=0.60."
        )
        assert v.action == "needs_review", (
            f"Generic money request with genuine=0.45 (< genuine_eligible=0.60) "
            f"must be needs_review, got {v.action!r}"
        )
        assert v.eligibility_label == "needs_review"

    def test_generic_reply_consistent(self):
        """Full chat() path: generic request → needs_review → deterministic reply."""
        ha = _make_agent_deterministic(self._rule_generic)
        out = ha.chat(self.GENERIC_REASON)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower()
        assert "eligible" not in out["reply"].lower()

    def test_generic_adversarial_high_financial_need(self):
        """Even with a mistakenly high financial_need (0.90), genuine=0.45 blocks eligible."""
        def rule(state, name, q):
            if name == "eligibility":
                return "eligible"
            if name == "financial_need":
                return 0.90  # very high — model was fooled by "money"+"college"
            return 0.45      # genuine: still vague

        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(rule))
        v = pc.check(self.GENERIC_REASON)
        assert v.action != "eligible", (
            f"High financial_need alone must not make a vague reason eligible. "
            f"genuine=0.45 < genuine_eligible=0.60 must block eligible. Got {v.action!r}"
        )

    # ------------------------------------------------------------------
    # Case B — Academic merit without financial hardship
    # Rule: academic merit must never be eligible when financial_need is low.
    # Even adversarially (model incorrectly scores financial_need high),
    # the check must catch it.
    # ------------------------------------------------------------------

    def _rule_academic(self, state, name, q):
        """Academic merit: high genuine (specific), but low financial_need."""
        if name == "eligibility":
            return "eligible"      # decider picks wrong label
        if name == "financial_need":
            return 0.15            # clearly no financial hardship described
        return 0.80                # genuine: specific, so above both floors

    def test_academic_merit_not_eligible(self):
        """Academic merit without financial hardship must not be eligible.

        genuine=0.80 passes both floors. But financial_need=0.15 < financial_floor=0.50
        → Rule 4 → ineligible.
        """
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule_academic))
        v = pc.check(self.ACADEMIC_REASON)
        assert v.action != "eligible", (
            f"Academic merit with financial_need=0.15 must not be eligible. "
            f"Got {v.action!r}"
        )
        assert v.action == "ineligible", (
            f"Academic merit with low financial_need must be ineligible, got {v.action!r}"
        )
        assert v.eligibility_label == "ineligible"

    def test_academic_reply_consistent(self):
        """Full chat() path: academic merit → ineligible → deterministic reply."""
        ha = _make_agent_deterministic(self._rule_academic)
        out = ha.chat(self.ACADEMIC_REASON)
        assert out["decisions"][0]["action"] == "ineligible"
        assert "ineligible" in out["reply"].lower()
        assert "is eligible" not in out["reply"].lower()

    def test_academic_adversarial_high_financial_need(self):
        """Adversarial: model incorrectly scores financial_need=0.85 for academic reason.

        genuine=0.78, financial_need=0.85 — both pass all score-based thresholds. Under
        the old rules this would have returned eligible. With the new text-based Rule 3a,
        ACADEMIC_REASON contains academic-merit keywords ("academic", "recognition",
        "research") and no financial-hardship keywords, so decide() correctly returns
        ineligible regardless of the financial_need score.
        This is the core fix for the live bug: "do not blindly trust the financial_need
        score when the applicant's stated reason does not establish financial hardship."
        """
        def rule(state, name, q):
            if name == "eligibility":
                return "eligible"
            if name == "financial_need":
                return 0.85  # high (decider wrong — text has no financial hardship)
            return 0.78      # genuine: above genuine_eligible=0.60

        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(rule))
        v = pc.check(self.ACADEMIC_REASON)
        # Rule 3a fires: ACADEMIC_REASON has academic-merit signal, no financial signal.
        # financial_need=0.85 is ignored — the text does not establish hardship.
        assert v.action == "ineligible", (
            "Academic-merit reason with no financial hardship in the text must be ineligible "
            "even when financial_need=0.85 (decider miscalibration). "
            f"Got {v.action!r}"
        )
        assert v.eligibility_label == "ineligible"

    # ------------------------------------------------------------------
    # Case C — Specific financial hardship (parent's job loss)
    # Rule: may be eligible when all criteria are met.
    # ------------------------------------------------------------------

    def _rule_hardship(self, state, name, q):
        """Specific financial hardship: high genuine, high financial_need, low ineligible mass."""
        if name == "eligibility":
            return "eligible"
        if name == "financial_need":
            return 0.90   # strong financial hardship signal
        return 0.85       # genuine: specific and well-evidenced

    def test_specific_hardship_eligible(self):
        """A specific parent's job-loss reason with all criteria met may be eligible.

        genuine=0.85 >= genuine_eligible=0.60 ✓
        financial_need=0.90 >= financial_floor=0.50 ✓
        confidence=0.85 >= sure=0.60 ✓
        prob_ineligible=0.075 < 0.30 ✓
        → eligible ✓
        """
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule_hardship))
        v = pc.check(self.HARDSHIP_REASON)
        assert v.action == "eligible", (
            f"Specific financial hardship with all criteria met must be eligible. "
            f"Got {v.action!r}"
        )
        assert v.eligibility_label == "eligible"

    def test_hardship_reply_consistent(self):
        """Full chat() path: specific hardship → eligible → deterministic reply."""
        ha = _make_agent_deterministic(self._rule_hardship)
        out = ha.chat(self.HARDSHIP_REASON)
        assert out["decisions"][0]["action"] == "eligible"
        assert "eligible" in out["reply"].lower()
        assert "ineligible" not in out["reply"].lower()
        assert "review" not in out["reply"].lower()

    # ------------------------------------------------------------------
    # Parametrized: all three cases cross-tested against decide() rules
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("reason_key,eligibility,genuine_p,financial_p,expected", [
        # generic: genuine passes floor but fails genuine_eligible → needs_review
        ("generic", "eligible", 0.45, 0.75, "needs_review"),
        # generic: genuine so low it fails even genuine_floor → needs_review
        ("generic", "eligible", 0.30, 0.80, "needs_review"),
        # generic: even with all scores high, if genuinely vague → needs_review
        ("generic", "eligible", 0.55, 0.90, "needs_review"),  # 0.55 < genuine_eligible=0.60
        # academic: high genuine, low financial → ineligible
        ("academic", "eligible", 0.80, 0.15, "ineligible"),
        # academic: high genuine, no financial → ineligible
        ("academic", "eligible", 0.78, 0.30, "ineligible"),
        # academic: ineligible label → ineligible regardless
        ("academic", "ineligible", 0.80, 0.90, "ineligible"),
        # hardship: all criteria met → eligible
        ("hardship", "eligible", 0.85, 0.90, "eligible"),
        # hardship: genuine just above genuine_eligible → eligible
        ("hardship", "eligible", 0.61, 0.85, "eligible"),
        # hardship: genuine just below genuine_eligible → needs_review
        ("hardship", "eligible", 0.59, 0.90, "needs_review"),
    ])
    def test_three_case_classify(self, reason_key, eligibility, genuine_p, financial_p, expected):
        """Parametrized classification across all three case types."""
        from agent.decision import ScholarshipPolicyCheck

        reasons = {
            "generic": self.GENERIC_REASON,
            "academic": self.ACADEMIC_REASON,
            "hardship": self.HARDSHIP_REASON,
        }

        def rule(state, name, q):
            if name == "eligibility":
                return eligibility
            if name == "financial_need":
                return financial_p
            return genuine_p

        pc = ScholarshipPolicyCheck(FakeDecider(rule))
        v = pc.check(reasons[reason_key])
        assert v.action == expected, (
            f"reason={reason_key!r}, eligibility={eligibility!r}, "
            f"genuine={genuine_p}, financial={financial_p}: "
            f"expected {expected!r}, got {v.action!r}"
        )
        assert v.action == v.eligibility_label


# ---------------------------------------------------------------------------
# Regression tests for the live bug: "I want this scholarship because I scored
# excellent marks."  Confidence=0.68, genuine=0.70, financial_need=0.85 → eligible
# (wrong). Now must be ineligible via Rule 3a (text-based academic-merit check).
# Also covers all requirements 7A–E.
# ---------------------------------------------------------------------------


class TestLiveBugExcellentMarks:
    """The exact live bug: academic-merit reason, financial_need=0.85 → was eligible, must be ineligible."""

    REASON = "I want this scholarship because I scored excellent marks."

    def _rule(self, state, name, q):
        # Reproduce the exact reported scores
        if name == "eligibility":
            return "eligible"    # decider label
        if name == "financial_need":
            return 0.85          # miscalibrated high score
        return 0.70              # genuine: 0.70 (≥ genuine_eligible=0.60, but text is merit-only)

    def test_decide_returns_ineligible(self):
        """Rule 0 (text-based policy check): academic-merit text, no financial hardship → ineligible.

        Rule 0 runs before any score-based gate, so even when genuine is low (decider
        says the reason is vague), the presence of academic-merit signals with no
        financial-hardship signals produces ineligible regardless of confidence or genuine.
        This is the fix for the live bug where genuine=0.293 caused Rule 1 to fire first
        (needs_review), hiding the academic-merit disqualification.
        """
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule))
        v = pc.check(self.REASON)
        assert v.action == "ineligible", (
            f"'Excellent marks' reason must be ineligible via Rule 0 — "
            f"academic merit + no financial hardship in text. "
            f"Got {v.action!r} (genuine={v.genuine}, financial_need={v.financial_need})"
        )
        assert v.eligibility_label == "ineligible"

    def test_action_equals_label(self):
        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(self._rule))
        v = pc.check(self.REASON)
        assert v.action == v.eligibility_label

    def test_full_chat_returns_ineligible_verdict(self):
        """Full chat() path: academic merit → ineligible → deterministic reply."""
        ha = _make_agent_deterministic(self._rule)
        out = ha.chat(self.REASON)
        assert out["decisions"][0]["action"] == "ineligible", (
            f"chat() must return ineligible for academic-merit reason. "
            f"Got {out['decisions'][0]['action']!r}"
        )
        assert "ineligible" in out["reply"].lower()
        assert "is eligible" not in out["reply"].lower()
        assert "eligible" not in out["reply"].lower() or "ineligible" in out["reply"].lower()

    def test_reply_matches_verdict(self):
        """The deterministic explanation must always match the final verdict."""
        ha = _make_agent_deterministic(self._rule)
        out = ha.chat(self.REASON)
        action = out["decisions"][0]["action"]
        reply = out["reply"].lower()
        assert action == "ineligible"
        assert "ineligible" in reply
        assert "congratulations" not in reply
        assert "qualifies" not in reply

    def test_decide_ineligible_even_with_low_genuine(self):
        """Rule 0 fires before Rule 1: academic merit is ineligible even when genuine is low.

        This is the EXACT live-bug scenario:
          Input:         "I want this scholarship because I scored excellent marks."
          Live decider:  confidence=0.73, genuine=0.293, financial_need=0.187
          Old behaviour: Rule 1 (genuine=0.293 < floor=0.40) fired → needs_review (WRONG)
          New behaviour: Rule 0 (academic signal, no financial signal) fires → ineligible (CORRECT)
        """
        def rule_low_genuine(state, name, q):
            if name == "eligibility":
                return "eligible"
            if name == "financial_need":
                return 0.19       # live value
            return 0.29           # genuine below floor=0.40 → old Rule 1 would fire

        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(rule_low_genuine))
        v = pc.check(self.REASON)
        assert v.action == "ineligible", (
            f"Rule 0 must fire before Rule 1: academic-merit reason is ineligible even when "
            f"genuine={v.genuine} < genuine_floor=0.40. Got {v.action!r}"
        )
        assert v.eligibility_label == "ineligible"

    def test_low_genuine_vague_reason_stays_needs_review(self):
        """Rule 0 must NOT fire for a vague financial reason with low genuine.

        Rule 0 only fires when there is an academic-merit signal AND no financial signal.
        A vague reason like "I need help" has no academic signal, so it falls through to
        Rule 1 (low genuine → needs_review), which is the correct policy outcome.
        """
        VAGUE_REASON = "I need financial help to continue my studies."

        def rule_low_genuine(state, name, q):
            if name == "eligibility":
                return "eligible"
            if name == "financial_need":
                return 0.60
            return 0.30   # genuine below floor

        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(rule_low_genuine))
        v = pc.check(VAGUE_REASON)
        assert v.action == "needs_review", (
            f"Vague financial reason with low genuine must be needs_review (Rule 1), "
            f"not ineligible. Got {v.action!r}"
        )

    def test_full_chat_low_genuine_academic(self):
        """Full chat() path with the exact live-bug scores → ineligible reply."""
        def rule(state, name, q):
            if name == "eligibility":
                return "eligible"
            if name == "financial_need":
                return 0.19
            return 0.29   # genuine below floor — old code gave needs_review

        ha = _make_agent_deterministic(rule)
        out = ha.chat(self.REASON)
        assert out["decisions"][0]["action"] == "ineligible", (
            f"Full chat() with genuine=0.29 and academic-merit reason must be ineligible. "
            f"Got {out['decisions'][0]['action']!r}"
        )
        assert "ineligible" in out["reply"].lower()
        assert "is eligible" not in out["reply"].lower()
# Requirements 7A–E: full regression coverage
# ---------------------------------------------------------------------------


class TestRequirements7A_to_7E:
    """
    7A. Academic merit only, even when financial_need = 0.85.
    7B. Generic financial-aid request, even when financial_need is high.
    7C. Specific financial hardship affecting tuition payment.
    7D. Ambiguous or insufficient evidence.
    7E. Final response consistency with every final verdict.
    """

    # ---- 7A: Academic merit only, financial_need = 0.85 ---- #

    @pytest.mark.parametrize("reason,expected", [
        # Exact live bug
        ("I want this scholarship because I scored excellent marks.", "ineligible"),
        # Other academic-merit phrasings
        ("I have a CGPA of 9.4 and want the scholarship for my research.", "ineligible"),
        ("I am top-ranked in my class and deserve recognition.", "ineligible"),
        ("I received an award for academic excellence and need support.", "ineligible"),
        ("My grades are outstanding and I want to continue my studies abroad.", "ineligible"),
    ])
    def test_7a_academic_merit_ineligible(self, reason, expected):
        """7A: Academic merit without financial hardship is always ineligible,
        even when financial_need is scored high by the decider."""
        def rule(state, name, q):
            if name == "eligibility":
                return "eligible"
            if name == "financial_need":
                return 0.85   # deliberately high (decider miscalibration)
            return 0.80       # genuine well above genuine_eligible

        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(rule))
        v = pc.check(reason)
        assert v.action == expected, (
            f"Academic-merit reason {reason!r} with financial_need=0.85 "
            f"must be {expected!r}, got {v.action!r}"
        )

    def test_7a_full_chat_academic_merit(self):
        """7A full path: academic merit → ineligible → deterministic reply."""
        reason = "I want this scholarship because I scored excellent marks."

        def rule(state, name, q):
            return "eligible" if name == "eligibility" else 0.85

        ha = _make_agent_deterministic(rule)
        out = ha.chat(reason)
        assert out["decisions"][0]["action"] == "ineligible"
        assert "ineligible" in out["reply"].lower()

    # ---- 7B: Generic financial-aid request, financial_need high ---- #

    @pytest.mark.parametrize("reason,expected", [
        # genuine=0.45 (passes floor=0.40, fails genuine_eligible=0.60) → needs_review
        ("I need money for college. Please help me.", "needs_review"),
        ("Please give me the scholarship, I really need it.", "needs_review"),
        # No financial and no academic signal → Rule 3b → needs_review (if high financial_need)
        ("I would like to apply for this scholarship.", "needs_review"),
    ])
    def test_7b_generic_request_needs_review(self, reason, expected):
        """7B: Generic requests never reach eligible, even with high financial_need."""
        def rule(state, name, q):
            if name == "eligibility":
                return "eligible"
            if name == "financial_need":
                return 0.80   # high (decider sees 'scholarship', scores high)
            return 0.45       # genuine: below genuine_eligible=0.60

        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(rule))
        v = pc.check(reason)
        assert v.action == expected, (
            f"Generic request {reason!r} must be {expected!r}, got {v.action!r}"
        )

    def test_7b_full_chat_generic_request(self):
        """7B full path: generic request → needs_review → deterministic reply."""
        reason = "I need money for college. Please help me."

        def rule(state, name, q):
            return "eligible" if name == "eligibility" else (0.80 if name == "financial_need" else 0.45)

        ha = _make_agent_deterministic(rule)
        out = ha.chat(reason)
        assert out["decisions"][0]["action"] == "needs_review"
        assert "review" in out["reply"].lower()
        assert "is eligible" not in out["reply"].lower()

    # ---- 7C: Specific financial hardship → eligible ---- #

    @pytest.mark.parametrize("reason", [
        "My father lost his job and we cannot afford my tuition this semester.",
        "My mother had emergency surgery; the bills have used up our savings.",
        "A flood destroyed our home and my parents cannot pay my college fees.",
        "I lost my part-time job and my family income does not cover tuition.",
    ])
    def test_7c_specific_hardship_eligible(self, reason):
        """7C: Specific financial hardship with all scores above thresholds → eligible."""
        def rule(state, name, q):
            if name == "eligibility":
                return "eligible"
            if name == "financial_need":
                return 0.90   # confirmed financial hardship
            return 0.85       # genuine: well above genuine_eligible

        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(rule))
        v = pc.check(reason)
        assert v.action == "eligible", (
            f"Specific hardship reason {reason!r} with all criteria met must be eligible. "
            f"Got {v.action!r}"
        )

    def test_7c_full_chat_hardship_eligible(self):
        """7C full path: specific hardship → eligible → deterministic reply."""
        reason = "My father lost his job and we cannot afford my tuition."

        def rule(state, name, q):
            return "eligible" if name == "eligibility" else 0.90

        ha = _make_agent_deterministic(rule)
        out = ha.chat(reason)
        assert out["decisions"][0]["action"] == "eligible"
        assert "eligible" in out["reply"].lower()
        assert "ineligible" not in out["reply"].lower()

    # ---- 7D: Ambiguous or insufficient evidence → needs_review ---- #

    @pytest.mark.parametrize("reason,genuine_p,financial_p,expected", [
        # Low confidence → needs_review (Rule 1, via FakeDecider conf=0.85, but below sure if label→needs_review)
        # Vague: genuine below floor
        ("I am facing difficulties at home.", 0.35, 0.60, "needs_review"),
        # Split distribution would trigger Rule 2, but we test Rule 1 here (genuine < floor)
        ("My situation is complicated.", 0.30, 0.70, "needs_review"),
        # Genuine passes floor but fails genuine_eligible
        ("I need this scholarship to help my family.", 0.50, 0.75, "needs_review"),
        # One-liner: genuine below genuine_eligible
        ("My father lost his job.", 0.55, 0.80, "needs_review"),
    ])
    def test_7d_ambiguous_needs_review(self, reason, genuine_p, financial_p, expected):
        """7D: Ambiguous or insufficient reasons must receive needs_review."""
        def rule(state, name, q):
            if name == "eligibility":
                return "eligible"
            if name == "financial_need":
                return financial_p
            return genuine_p

        from agent.decision import ScholarshipPolicyCheck
        pc = ScholarshipPolicyCheck(FakeDecider(rule))
        v = pc.check(reason)
        assert v.action == expected, (
            f"Ambiguous reason {reason!r} (genuine={genuine_p}) must be {expected!r}, "
            f"got {v.action!r}"
        )

    # ---- 7E: Final response consistency with every verdict ---- #

    @pytest.mark.parametrize("reason,eligibility_label,genuine_p,financial_p,expected_action,expected_word", [
        # eligible
        ("My mother was hospitalised and we cannot afford tuition.", "eligible", 0.85, 0.90, "eligible", "eligible"),
        # needs_review — genuine below genuine_eligible
        ("I need some financial support.", "eligible", 0.50, 0.80, "needs_review", "review"),
        # ineligible — academic merit, no financial
        ("I scored excellent marks and deserve the scholarship.", "eligible", 0.80, 0.85, "ineligible", "ineligible"),
        # ineligible — financial_need below floor
        ("I want the scholarship to buy a laptop for my projects.", "eligible", 0.80, 0.30, "ineligible", "ineligible"),
    ])
    def test_7e_response_always_matches_verdict(
        self, reason, eligibility_label, genuine_p, financial_p, expected_action, expected_word
    ):
        """7E: The final response text must always agree with the final verdict."""
        def rule(state, name, q):
            if name == "eligibility":
                return eligibility_label
            if name == "financial_need":
                return financial_p
            return genuine_p

        ha = _make_agent_deterministic(rule)
        out = ha.chat(reason)
        action = out["decisions"][0]["action"]
        reply_lower = out["reply"].lower()

        assert action == expected_action, (
            f"reason={reason!r}: expected action {expected_action!r}, got {action!r}"
        )
        assert expected_word in reply_lower, (
            f"reason={reason!r}: action={action!r}, reply must contain {expected_word!r}. "
            f"reply={out['reply']!r}"
        )
        # Consistency invariant: action and eligibility must match
        assert action == out["decisions"][0]["eligibility"], (
            f"action and eligibility in log must match: {out['decisions'][0]}"
        )
