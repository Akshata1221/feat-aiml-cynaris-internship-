# Git workflow (Windows CMD)

Run from the project folder. Create your own empty GitHub repository first.

```bat
git init
git checkout -b feat/aiml-W4-Akshata
git add src tests requirements.txt .gitignore
git commit -m "Implement W4 model workflow and tests"
git add README.md docs outputs
git commit -m "Document results and self-review"
git remote add origin https://github.com/Akshata1221/REPLACE_WITH_REPO_NAME.git
git push -u origin feat/aiml-W4-Akshata
```

Replace the remote URL with your actual repository. Complete the CIA reviews before committing. Create a PR from the pushed feature branch into `main` on GitHub (create main first if this is a new repository). Do not claim the PR is raised until GitHub confirms it.
