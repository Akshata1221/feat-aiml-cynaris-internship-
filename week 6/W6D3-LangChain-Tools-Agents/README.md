# W6D3-LangChain-Tools-Agents

# Setup (Windows CMD)
```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python -m pytest -q
```
Ollama must be running locally. Actual output evidence must be generated on your computer.

## Git workflow
```bat
git init
git checkout -b feat/aiml-W6-Akshata
git add .
git commit -m "feat: implement core workflow"
git add .
git commit -m "docs: add test evidence and self review"
git remote add origin YOUR_REPOSITORY_URL
git push -u origin feat/aiml-W6-Akshata
```
Make a second commit only after making and staging real changes. Raise a PR on GitHub. Obtain two genuine CIA Full Stack Mentor reviews before committing.

## Run
```bat
python -m src.demo
```
