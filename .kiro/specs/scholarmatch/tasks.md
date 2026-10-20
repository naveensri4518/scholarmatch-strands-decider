# ScholarMatch — tasks

- [x] Spec written (requirements.md, design.md)
- [ ] `agent/decision.py` — `#policy` pattern, `QUESTIONS`, `ScholarshipPolicyCheck`
- [ ] `agent/tools.py` — `get_application`, `record_decision`, sample data helpers
- [ ] `agent/agent.py` — `HackathonAgent.chat`
- [ ] `agent/data/applications.json` — synthetic applications
- [ ] `agent/eval_cases.jsonl` — 26 labelled cases (>= 12 in test split)
- [ ] `agent/demo_script.yaml` — 3 turns (eligible / needs_review / ineligible)
- [ ] `agent/features.yaml` — Memory/Gateway off
- [ ] `agent/AGENT.md` — use case
- [ ] `.\kit.cmd test` passes
- [ ] `.\kit.cmd run` passes (one scripted conversation)
- [ ] `.\kit.cmd bench --split tune`, then `--split test`
- [ ] `.\kit.cmd readme`
- [ ] (skipped per participant: deploy, article, video, publish)
