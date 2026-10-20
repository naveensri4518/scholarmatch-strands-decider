# ScholarMatch — requirements

## Problem
A university scholarship office reviews many student applications by hand: a reviewer reads each
applicant's reason and decides whether it matches the scholarship's written policy. It is
repetitive and slow.

## Users
Scholarship office staff (reviewers). They submit a student's application (by student id, or by
pasting the reason) and receive an eligibility verdict with a short explanation.

## The decision (Strands Decider 2B — `#policy`)
For each application, Strands Decider 2B reads the student's free-text reason against the
Need-Based Financial Aid Scholarship policy and classifies it:
- `eligible` — a specific, genuine financial difficulty that matches the policy
- `needs_review` — borderline, vague, or low-confidence; held for a human
- `ineligible` — describes merit, ambition, a generic wish for money, or does not match the
  financial-need policy

A second question checks whether the reason is specific and genuine. Low confidence or a vague
reason always escalates to `needs_review` (a human); the agent never auto-approves an unsure case.

## Out of scope for the decider
No numeric thresholds are handed to the decider (the rules say compute those in code). This
simplified version decides purely on the reason text versus the policy prose.

## Scope of this build (per the participant)
- Local only: runs via `.\kit.cmd run` / `.\kit.cmd demo`.
- No external services (no AgentCore Gateway/Memory, no external APIs); the LLM and decider use
  the already-configured UGM cloud backends (a `.env` setting; swapping to local Ollama changes no
  agent code).
- No deploy, no AWS. Honest consequence: this is **not** a valid hackathon submission on its own —
  the rules require an agent on AWS with Amazon Bedrock. Deploy is kept out by instruction; it can
  be added later as a small extra step if the participant decides to submit.
- Synthetic sample student data in `agent/data/`.

## Data
Hand-authored synthetic student applications (`student_id, name, cgpa, credits, income_bracket,
scholarship, reason`) in `agent/data/applications.json`, deterministic for tests and the demo.
