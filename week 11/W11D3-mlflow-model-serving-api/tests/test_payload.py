def test_dataframe_split_payload_shape():
    payload = {"dataframe_split": {"columns": ["a", "b"], "data": [[1.0, 2.0]]}}
    assert len(payload["dataframe_split"]["columns"]) == 2
    assert len(payload["dataframe_split"]["data"][0]) == 2
