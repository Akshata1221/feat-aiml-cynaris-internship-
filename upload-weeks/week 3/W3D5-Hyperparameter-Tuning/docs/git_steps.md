# Git steps — Windows CMD

Create an empty GitHub repository named `W3D5-Hyperparameter-Tuning` first. Run these commands from this extracted project directory:

```bat
git init
git checkout -b feat/aiml-W3-Akshata
git add src tests requirements.txt .gitignore
git commit -m "Implement SVM and KNN grid and random search tuning"
git add README.md docs
git commit -m "Document experiments self-review and viva preparation"
git remote add origin https://github.com/Akshata1221/W3D5-Hyperparameter-Tuning.git
git push -u origin feat/aiml-W3-Akshata
```

After running the project, commit evidence separately if needed:
```bat
git add outputs
git commit -m "Add hyperparameter tuning experiment evidence"
git push
```

On GitHub, open **Compare & pull request**, set base `main` (create it if your new repo has no base branch), and submit the PR. If your repository already exists with a main branch, clone it first rather than initializing a separate history. Do not claim the PR is raised until it is actually created.
