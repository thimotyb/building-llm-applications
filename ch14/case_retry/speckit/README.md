# Spec Kit artifacts of the payment-retry case

The specification-driven (SDD) version of the case used by the harness in `../`. It was produced with
[GitHub Spec Kit](https://github.com/github/spec-kit) in the augmented-coding course
([`demo-augmented-coding`](https://github.com/thimotyb/demo-augmented-coding), module 3, step 2) and is copied here
unchanged, so the harness example and the documents that define its goal live together. The text is in Italian.

| File | Spec Kit step | Content |
| --- | --- | --- |
| `.specify/memory/constitution.md` | `constitution` | Principles that bind all the work |
| `specs/001-payment-retry/spec.md` | `specify` | Scope, actors, FR-1..FR-3, acceptance scenarios AC-1..AC-3, exclusions (identical to `../spec.md`) |
| `specs/001-payment-retry/decisions.md` | `clarify` | Open decisions that block dependent work (idempotency, repeated checkout, non-transient rejections) |
| `specs/001-payment-retry/plan.md` | `plan` | Technical approach for a local demo |
| `specs/001-payment-retry/checklists/requirements.md` | `checklist` | Quality review of the requirements |
| `specs/001-payment-retry/tasks.md` | `tasks` | Ordered tasks, each linked to requirements and a verification |
| `prompts/` | | The prompts given to the agent for `constitution`, `specify`, `clarify`, `plan` |

How it connects to the harness: AC-1, AC-2 and AC-3 become the three tests in `../test_payment_retry.py`, which
`../harness.py` runs on every candidate; the exclusions become the static rules in `../harness.py`; the decisions
in `decisions.md` are what the agent must not code. The prompts refer to an `inputs/` folder with the original
case material, which is not part of this repository.
