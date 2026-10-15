import pandas as pd
from src.pipeline_utils import SENSOR_FEATURES, add_missing_flags, make_group_ids

def test_missing_flags_added():
    x = pd.DataFrame([{c: 1.0 for c in SENSOR_FEATURES}])
    x.loc[0, "temperature_c"] = float("nan")
    out = add_missing_flags(x)
    assert out.loc[0, "missing_temperature_c"] == 1
    assert out.loc[0, "missing_gyro_x_dps"] == 0

def test_group_ids_keep_fault_episode_together():
    d = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-06-01 11:55", "2025-06-01 12:05", "2025-06-01 12:10"
        ]),
        "fault_episode_id": [7, 7, 0],
    })
    groups = make_group_ids(d)
    assert groups.iloc[0] == groups.iloc[1]
