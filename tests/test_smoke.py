from pathlib import Path

from utils.helper import load_dataset, infer_target_column, detect_task_type


def test_sample_dataset_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "data" / "sample_dataset.csv"
    df = load_dataset(dataset)
    target = infer_target_column(df)
    task_type = detect_task_type(df[target])
    assert not df.empty
    assert target == "target"
    assert task_type == "classification"
