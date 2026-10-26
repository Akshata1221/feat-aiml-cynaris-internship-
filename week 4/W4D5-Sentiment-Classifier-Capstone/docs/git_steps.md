# Git workflow (Windows CMD)
Create an EMPTY GitHub repository for this project first.
```bat
git init
git switch -c feat/aiml-W4-Akshata
git add src tests requirements.txt .gitignore
git commit -m "feat: classification — logistic vs random forest comparison"
git add README.md docs outputs
git commit -m "docs: add tests evidence and self review"
git remote add origin https://github.com/Akshata1221/REPLACE_WITH_REPO.git
git push -u origin feat/aiml-W4-Akshata
```
Create PR on GitHub targeting main. Complete CIA reviews BEFORE committing; update docs with genuine feedback.
