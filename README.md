# ScholarMatch

**a scholarship reviewer that reads a student's application reason and decides whether**

An agent on AWS that hands one decision to **Strands Decider 2B** and the conversation to **Amazon Bedrock**. Decision pattern: `#policy`. Tags: `#strands-decider-2b` `#policy`.

- Video: (add the YouTube/Vimeo link)
- Article: (add the AWS Builder Center link)
- Deployed: run `./kit deploy`

## The project

- **Pitch:** a scholarship reviewer that reads a student's application reason and decides whether
  it matches the scholarship policy, returning eligible / needs_review / ineligible with a short
  explanation.
- **Who it is for:** a university scholarship office, so staff are not reading every application by
  hand; unsure cases still go to a human.
- **Features:**
  - reads a student's free-text application reason against the Need-Based Financial Aid Scholarship
    policy;
  - classifies `eligible` / `needs_review` / `ineligible` with calibrated confidence;
  - low-confidence or vague reasons are held for a human, never auto-approved;
  - a short, plain explanation that references the policy clause;
  - look up an application by student id from sample data.
- **The decision handed to Strands Decider 2B:** does the applicant's reason match the scholarship
  policy? (`#policy`)
- **Why a decision model:** the check is a language judgement (does this written reason satisfy a
  policy written in prose), it must return a confidence the code can branch on, and it runs on
  every application.
- **Local only:** this build runs locally (`.\kit.cmd run` / `.\kit.cmd demo`); it is not deployed.
- **Built with the AWS User Group Madurai Strands Decider hackathon kit.**



## How it is built

Strands Agents with Amazon Bedrock for the generative part, the tools in `agent/tools.py`, deployed to Amazon Bedrock AgentCore Runtime, and one decision handed to Strands Decider 2B at a clear seam in the agent's path.

Every Amazon Bedrock model id is set in `.env` (`BEDROCK_MODEL_ID`, and `BEDROCK_SMALL/MEDIUM/LARGE_MODEL_ID` for routing), Amazon Nova by default.



## Reusable decision pattern: scholarship policy classification  (#policy)

The pattern lives in [`agent/decision.py`](agent/decision.py), labelled `REUSABLE DECISION PATTERN`. It contains the typed questions and their options, the wiring that puts the decision in the agent's path, and a note on how to reuse it:

```text
REUSABLE DECISION PATTERN: scholarship policy classification  (#policy)

ScholarMatch reads a student's scholarship application reason and decides whether it matches the
Need-Based Financial Aid Scholarship policy, returning eligible / needs_review / ineligible.

What it does
    For each application, Strands Decider 2B reads the student's free-text reason against a written
    scholarship policy and classifies it as eligible, needs_review, or ineligible. A second question
    checks whether the reason is specific and genuine. Low confidence, or a vague/templated reason,
    always escalates to needs_review (a human), so the agent never auto-approves an unsure case.

Typed questions and options
    eligibility (choice): eligible / needs_review / ineligible, each with a one-line definition.
    genuine (noul): is the reason specific and genuine, rather than vague, generic, or templated?
    See QUESTIONS below.

Wiring
    ScholarshipPolicyCheck.check(reason) builds a state from the reason, calls
    Decider.ask(state, QUESTIONS), and maps the answers to an action in decide(). It runs in
    agent.py before the reply is written (the "before the agent runs" form of the #policy pattern):
    the Strands Agent only explains the verdict the decider already reached; it never re-decides.

How to reuse it
    Copy this file, replace POLICY and SCHOLARSHIP with your own policy prose and name, and reword
    the two questions for your domain. The thresholds that turn probabilities into an action live
    in decide() (sure=0.6, genuine_floor=0.4): change them there, in code you own.
```

Typed questions:

```python
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
}
```

## Measured

Decider: strands-decider-2b-hobson-v19 (ugmdu). 13 labelled cases, 26 decisions.

| | Strands Decider 2B | qwen3.5-4b |
|---|---|---|
| Accuracy | 73.1% | 65.4% |
| Latency p50 per decision | 4019 ms | 17781 ms |
| Latency p95 | 4956 ms | 19312 ms |
| Cost per 1,000 decisions | $0.0350 | $0.0000 |
| Answered alone at confidence >= 0.7 | 27% | 100% |
| Accuracy when confident | 100.0% | 65.4% |

At the median the decider was 4.4x faster than the LLM.

Reproduce with `./kit bench` (cases in `agent/eval_cases.jsonl`; charts in `submission/bench/`).

## Run it

```bash
./kit setup        # checks your machine and AWS, sets up Strands Decider 2B (local model or hosted API)
./kit demo         # web UI at http://localhost:8501
./kit bench        # the decision vs an Amazon Bedrock LLM
./kit deploy       # Amazon Bedrock AgentCore Runtime
```

Requirements: Python 3.10+, AWS credentials with Amazon Bedrock access, and for deploying, Node 20+ with the AgentCore CLI (`./kit agentcore-cli`).

## Credits

Built with the AWS User Group Madurai Strands Decider hackathon kit (https://github.com/awsugmdu). The kit's reusable library (`decider_kit/`, the demo UI, the video and article tooling, and the Kiro steering, skills and hooks) is the user group's; the agent in `agent/` is mine. Strands Decider: https://github.com/strands-labs/strands-decider (Apache-2.0).

Licence: Apache-2.0 (see LICENSE).
