1. I trained and saved a complete Logistic Regression pipeline and served it using FastAPI. The pipeline avoids inconsistent scaling.
2. Input validation and loading the model safely were the main challenges. I used Pydantic and loaded only the locally generated joblib file.
3. I would add Docker, request monitoring, authentication, and deployment CI.
