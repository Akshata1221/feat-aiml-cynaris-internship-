"""Load, inspect and clean a reproducible tabular dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

def sample_data():
    return pd.DataFrame({'student_id':[1,2,3,3,4,5], 'age':[20,21,np.nan,np.nan,22,19], 'score':[75,88,92,92,np.nan,68], 'city':['Bengaluru','Mysuru','Bengaluru','Bengaluru',None,'Mysuru']})

def clean(df):
    df = df.drop_duplicates().copy()
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(df['score'].median())
    df['city'] = df['city'].fillna('Unknown').str.strip()
    return df

def main():
    out = BASE/'outputs'; out.mkdir(exist_ok=True)
    source = BASE/'data'/'students.csv'
    source.parent.mkdir(exist_ok=True)
    if not source.exists(): sample_data().to_csv(source,index=False)
    raw = pd.read_csv(source)
    report = {'raw_rows':len(raw), 'duplicates':int(raw.duplicated().sum()), 'missing_by_column':raw.isna().sum().to_dict(), 'dtypes':raw.dtypes.astype(str).to_dict()}
    import json
    (out/'inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    result = clean(raw); result.to_csv(out/'cleaned_students.csv',index=False)
    print('Original shape:',raw.shape,'Clean shape:',result.shape)
    print('Missing values after cleaning:',result.isna().sum().to_dict())
    print(result.head().to_string(index=False))

if __name__=='__main__': main()
