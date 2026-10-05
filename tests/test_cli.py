import json

import pandas as pd
import pytest

from dat_parkinsons.cli import main


def test_fitting_patients_cannot_be_used_as_holdout(tmp_path):
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "pred_probability": [0.2, 0.8, 0.3, 0.7],
        }
    )
    frame.to_csv(tmp_path / "preds.csv", index=False)
    tables = tmp_path / "tables.json"
    tables.write_text(json.dumps({"one": "preds.csv"}))
    blend = tmp_path / "blend.json"
    main(["fit-blend", "--tables", str(tables), "--output", str(blend)])
    args = [
        "evaluate-blend",
        "--tables",
        str(tables),
        "--blend",
        str(blend),
        "--output",
        str(tmp_path / "metrics.json"),
    ]
    with pytest.raises(ValueError, match="overlap"):
        main(args)
    frame["patient_id"] = ["e", "f", "g", "h"]
    frame.to_csv(tmp_path / "preds.csv", index=False)
    main(args)
    assert json.loads((tmp_path / "metrics.json").read_text())["n"] == 4
