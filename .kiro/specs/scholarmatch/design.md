# ScholarMatch — design

## Architecture
`HackathonAgent.chat` resolves an application (by student id via the sample data, or by treating
the message as the reason), runs the `#policy` decision (`ScholarshipPolicyCheck` in
`agent/decision.py`) through Strands Decider 2B, then a Strands Agent
(`decider_kit.llm.agent_model`) writes a one-to-two-sentence explanation of that verdict. The
verdict is logged to `decisions` for the demo UI.

## Files
- `agent/decision.py` — REUSABLE DECISION PATTERN (`#policy`): `PATTERN_TAG`, `QUESTIONS`
  (`eligibility` choice + `genuine` noul), `ScholarshipPolicyCheck` wiring class.
- `agent/tools.py` — `get_application`, `record_decision` `@tool` functions over in-memory sample
  data (mirrored from `agent/data/applications.json`); plain helpers the agent calls directly.
- `agent/agent.py` — `NAME`, `HackathonAgent.chat` -> `{reply, decisions, model_id, latency_ms}`.
- `agent/eval_cases.jsonl` — 26 labelled cases (`eligibility` + `genuine`), split tune/test.
- `agent/demo_script.yaml` — three turns: eligible, needs_review, ineligible.
- `agent/features.yaml` — AgentCore Memory/Gateway both off (no deploy).
- `agent/data/applications.json` — synthetic student applications.

## Decision wiring
`ScholarshipPolicyCheck.check(reason)` builds `state = "Scholarship: <name>\nApplicant's reason:
<reason>"`, calls `Decider.ask(state, QUESTIONS)` (the policy is in the `eligibility` question
text), and maps answers to an action in `decide()`:
- confidence < 0.6 OR genuine p_yes < 0.4 -> `needs_review`
- elif eligibility label == `eligible` -> `eligible`
- elif `ineligible` -> `ineligible`
- else -> `needs_review`

This is the "before the agent runs" form of the `#policy` pattern (decider_kit/patterns/policy.py):
the decider runs before the reply is written; the Strands Agent only explains the verdict, it
never re-decides.

## LLM
The explanation is generated through a Strands Agent built with `decider_kit.llm.agent_model`
(`LLM_PROVIDER` in `.env` — currently the UGM cloud; swapping to local Ollama only changes `.env`).
One LLM call per turn.

## Honesty
No measured numbers are invented. `kit bench` reports the real decider accuracy and latency vs an
LLM baseline on the held-out test split. With no deploy there are no AWS cost/latency numbers
(those require a deployed agent); that is reported plainly, not invented.
