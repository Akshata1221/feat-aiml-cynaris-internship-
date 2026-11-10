import requests

def predict(payload, endpoint="http://127.0.0.1:5001/invocations"):
    response = requests.post(endpoint, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    sample = {"dataframe_split": {
        "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        "data": [[5.1, 3.5, 1.4, 0.2]]
    }}
    print(predict(sample))
