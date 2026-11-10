# W10D4 — Human-in-the-Loop with LangGraph

## Objective
Pause before sending a response, collect approval or edits, and resume with Command(resume=...).

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

## Project structure
- `src/human_review/` — implementation
- `tests/` — automated tests
- `docs/SELF_REVIEW.md` — submission checklist
- `docs/VIVA.md` — viva preparation

## Run
Use the examples in the tests as executable usage examples. LangGraph interrupts require a checkpointer and a stable `configurable.thread_id`. To resume an interrupted run, invoke the compiled graph with `Command(resume=human_value)` and the same thread ID.

## Git submission
```bash
git init
git switch -c feat/aiml-W10-akshata
git add .
git commit -m "feat: implement LangGraph workflow"
git add README.md docs tests
git commit -m "test: add routing and workflow coverage"
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin feat/aiml-W10-akshata
```
Create a pull request on GitHub. Replace the branch suffix if your mentor requires a different spelling.

## Important
This ZIP contains a local project scaffold and tests. It does not create GitHub commits, a remote push, a PR, CIA interaction logs, or lesson completion records. Complete those actions in the relevant platforms. Review package/API versions against your course environment before submission.
