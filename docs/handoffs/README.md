# Handoff guide

This repository uses handoff documents to let a new ChatGPT session, coding agent, developer, or future maintainer resume work without reconstructing the full project history.

## Purpose

A handoff is a **control document**, not another experiment log. It should answer:

1. What is frozen and must not be changed?
2. What branch/commit is the starting point?
3. What is already experimentally proven?
4. What question is being investigated next?
5. What files/contracts are authoritative?
6. What is the first executable step?
7. What must not be mixed into the current experiment?
8. What result is required before the next stage is allowed?

## Handoff structure

Each substantial new research/development line should have:

```text
docs/handoffs/<topic>/README.md
```

Prefer one authoritative README over many overlapping notes. Create a separate closeout README only when a completed infrastructure/benchmark/migration stage needs a durable final record.

A good handoff should contain:

- frozen baseline and tag;
- current `main` and intended branch name;
- data/map/model contracts and critical hashes where relevant;
- architecture boundary;
- research hypotheses;
- ordered stages and gates;
- metrics;
- known failure cases;
- reproducibility/deployment constraints;
- exact first commands only when they are stable;
- explicit non-goals;
- conditions for closeout.

## Agent/session resume protocol

A new session or coding agent should:

1. read the topic handoff first;
2. inspect current Git branch, HEAD and working-tree status;
3. inspect only the source/config files named by the handoff before broad repository exploration;
4. verify frozen assets/contracts before modifying code;
5. execute one gated stage at a time;
6. preserve existing experiment outputs and use fresh run IDs;
7. distinguish measured evidence from hypotheses;
8. update the handoff/closeout when a stage materially changes the project state.

Do not infer current state from old branch names alone. Git, the frozen tag, current branch, and the handoff document are authoritative together.

## Git discipline

- `main` should stay understandable and runnable.
- Frozen benchmark tags are immutable.
- Research changes belong on named research branches.
- Infrastructure/refactor commits should be separate from algorithm/metric-changing commits.
- Avoid combining dependency modernization, architecture refactors, model changes and policy changes into one commit.
- Prefer small commits with a single experimental or architectural purpose.
- Do not silently overwrite prior run directories.
- Heavy generated artifacts and local model/data assets stay outside Git unless explicitly promoted.

## AI/coding-agent readiness

When the runtime architecture stabilizes, add a root `AGENTS.md` that points to the authoritative architecture, test commands, branch policy and safety constraints. Keep it short and repository-specific.

Agent-friendly development depends more on stable interfaces than on long prompts:

- deterministic CLI commands;
- explicit config schemas;
- machine-readable JSON reports;
- small parity fixtures;
- clear input/output contracts;
- no hidden dependence on developer-local paths;
- no secrets in configs/logs;
- tests that can distinguish infrastructure regressions from research-result changes.

GitHub automation and coding agents should be introduced only after these contracts exist. Agents should work through protected branches/PRs rather than directly mutating the frozen baseline.
