# Landslide Sensor ML Pipeline (synthetic-data baseline)

**Important:** the bundled dataset is synthetic. Its risk labels are illustrative rules, not verified landslide outcomes. Metrics here are not real-world accuracy.

## Files
- `data/synthetic_landslide_sensor_dataset.csv`: original 14,400-row synthetic dataset.
- `train_landslide_model.py`: trains the risk-scenario classifier.
- `train_fault_model.py`: trains the sensor-fault classifier.
- `src/pipeline_utils.py`: shared features, group-aware splits, training and evaluation.
- `predict.py`: predicts from a CSV that has the eight sensor feature columns.
- `models/`: saved models and feature schemas (created during training).
- `results/`: metrics, classification reports, split summaries and confusion matrices.

## Windows PowerShell setup
From the project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked in the current PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

## Train

```powershell
python train_landslide_model.py
python train_fault_model.py
```

Evaluation outputs are saved under `results/`; models are saved under `models/`.

## Predict from a feature CSV
The input CSV must contain these columns, with the same units as the training dataset:
`gyro_x_dps`, `gyro_y_dps`, `gyro_z_dps`, `rainfall_intensity_mm_h`, `rainfall_accum_24h_mm`, `humidity_rh_pct`, `temperature_c`, `vibration_g`.

```powershell
python predict.py --task risk --input path\to\sensor_features.csv
python predict.py --task fault --input path\to\sensor_features.csv
```

The risk baseline rejects missing sensor readings. The fault model adds missing-value indicators and imputes remaining missing values inside the saved pipeline. Live ESP32 parsing/serial integration is not implemented yet.

## Feature and leakage decisions
- Risk model target: `landslide_risk_label`. It trains only on rows where `sensor_fault_label == Normal`, because risk labels are assigned from the underlying synthetic environment while injected sensor faults alter observed measurements. Inputs are the eight measured sensor columns.
- Fault model target: `sensor_fault_label`. Inputs are the eight measured sensor columns plus one missingness indicator per sensor. It excludes `faulty_sensor`, `fault_episode_id`, `environmental_scenario`, `dataset_source`, IDs, timestamps, and target labels.
- Splits group normal rows into 12-hour time blocks across all nodes and group all rows of each nonzero fault episode together, even if an episode crosses a block boundary. This keeps fault episodes together and reduces temporal leakage. Fault episodes can overlap in time with normal rows from the same period, so evaluation remains an approximate synthetic-data assessment.
- Rolling features are intentionally not included in this first baseline. We can add them after a verified live stream exists and the feature-generation code can be shared exactly between training and inference.

## Limitations
- Synthetic measurements do not reproduce real terrain, sensor calibration, weather, or landslide physics.
- Some sensor/fault combinations have no examples in this dataset. The model cannot be expected to learn those combinations.
- `alarm` is not a training feature; it is a rule-based output from the firmware, not an independent sensor measurement.
- Before live inference, debug invalid readings (for example, implausible temperature or `NaN` sensor fields), calibrate the rain sensor, verify units, and compare real feature distributions with training data.
- Do not claim real-world landslide prediction accuracy from this dataset.
