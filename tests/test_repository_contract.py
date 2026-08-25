from pathlib import Path

def test_required_demo_artifacts_exist():
    root = Path(__file__).parents[1]
    assert (root / 'main.py').exists()
    assert (root / 'requirements.txt').exists()
    assert (root / 'preprocessor.pkl').exists()
    assert (root / 'stress_predictor.pkl').exists()
