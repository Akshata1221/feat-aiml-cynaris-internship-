# Git workflow (Windows CMD)

Create an empty GitHub repository called `W5D1-Ollama-Local-LLM`, then in this extracted folder run:

```bat
git init
git checkout -b feat/aiml-W5-Akshata
```

First complete a CIA Full Stack Mentor review, implement any feedback, and run tests. Then:

```bat
git add src tests requirements.txt .gitignore
git commit -m "feat: ollama setup — local llm inference"
```

Run the experiment, fill in documentation and complete the second CIA interaction. Do not commit raw private prompts or secrets. Then:

```bat
git add README.md docs
git commit -m "docs: add model comparison review and submission checklist"
git remote add origin https://github.com/YOUR_USERNAME/W5D1-Ollama-Local-LLM.git
git push -u origin feat/aiml-W5-Akshata
```

Replace YOUR_USERNAME with your GitHub username. Open GitHub, select Compare & pull request, set base `main` if present, and submit the PR. If the remote repository is empty and has no main branch, first create main on GitHub or establish a base branch before opening a PR.
