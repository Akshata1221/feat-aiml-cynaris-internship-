# W1D3 — Data Loading, Cleaning & Inspection

Load CSV data, inspect duplicates, missing values and types, then apply reproducible cleaning. If `data/students.csv` is absent, a small demonstration dataset is generated.

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -v
python -m src.clean_data
```

Evidence: `outputs/inspection.json` and `outputs/cleaned_students.csv` (created after running). This is demonstration data, not a real-world evaluation dataset.
