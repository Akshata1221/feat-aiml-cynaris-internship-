from src.clean_data import sample_data, clean

def test_duplicates_removed():
    assert len(clean(sample_data())) == 5

def test_no_missing():
    assert not clean(sample_data()).isna().any().any()

def test_city_filled():
    assert 'Unknown' in clean(sample_data())['city'].values
