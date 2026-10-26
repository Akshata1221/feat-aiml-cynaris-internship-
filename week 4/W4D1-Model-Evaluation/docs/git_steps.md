# Git workflow — Windows CMD

Create a **new GitHub repository** called `W4D1-Model-Evaluation` and copy its URL. From the extracted project directory:

```bat
git init
git checkout -b feat/aiml-W4-Akshata
git add src tests requirements.txt .gitignore
git commit -m "Implement model evaluation and stratified cross-validation"
git add README.md docs outputs
git commit -m "Document metrics and add evaluation evidence"
git remote add origin https://github.com/Akshata1221/W4D1-Model-Evaluation.git
git push -u origin feat/aiml-W4-Akshata
```

**Before commits:** run tests, run the script, and obtain a genuine CIA Full Stack Mentor code review with two logged interactions. The second commit should include generated output evidence. If Git reports an existing remote, inspect `git remote -v` before changing it. On GitHub choose **Compare & pull request** and select the correct base branch. If the repository has no `main`, create/push `main` before opening a PR; do not create a PR against a nonexistent base.
