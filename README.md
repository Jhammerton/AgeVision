# AgeVision

An end-to-end computer-vision system for estimating **apparent age** from a face image.
AgeVision supports exact-age regression and age-group classification, reports uncertainty,
and evaluates errors across age and demographic slices. It is a research/education project,
not an identity, eligibility, or medical decision system.

## System

```text
UTKFace images -> validate/split -> crop + normalize -> ResNet18 -> age estimate
                                                               -> evaluation report
Uploaded image -> FastAPI -> same transforms -> estimate + plausible range -> web UI
```

## Features

- UTKFace filename parsing and reproducible train/validation/test manifests
- ResNet18 and EfficientNet-B0 transfer learning or a small CNN baseline
- Optional moderate age-group balancing during training
- YuNet face detection and cropping for uploaded portraits
- Regression and age-group classification experiments
- MAE, RMSE, error percentiles, and metrics by age, gender, and ethnicity labels
- FastAPI image-upload endpoint and browser interface
- Unit tests, Docker image, and GitHub Actions CI

## Quick start

1. Use Python 3.10+ and install `pip install -e ".[dev]"`.
2. Download UTKFace and place its `.jpg` files under `data/raw/UTKFace/`.
3. Build manifests: `python -m src.ingestion --input data/raw/UTKFace`.
4. Train: `python -m src.train --task regression --epochs 10`.
   Add `--balanced-sampling` to moderately oversample rarer age groups.
5. Evaluate: `python -m src.evaluate --checkpoint models/agevision_regression.pt`.
   Use `--output reports/experiment-name.json` to preserve a named experiment report.
6. Serve: `uvicorn src.api:app --reload`, then open <http://localhost:8000>.

Run `pytest` for tests. Training defaults live in `configs/model.yaml`; CLI flags can
override the most common settings. The API returns HTTP 503 until a checkpoint exists.
The upload flow expects a single, front-facing portrait; UTKFace itself contains aligned faces.
Set `AGEVISION_CHECKPOINT` to serve a checkpoint other than
`models/agevision_regression.pt`.
The repository includes that optimized serving checkpoint; other generated training
checkpoints remain excluded from version control.

Uploaded portraits are processed with the MIT-licensed YuNet detector from the
[OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
Uploads with no detectable face or more than one detected face are rejected.

## Experiment results

Five-epoch regression experiments on the current UTKFace split produced:

| Model | Test MAE | Test RMSE | 60+ MAE |
| --- | ---: | ---: | ---: |
| ResNet18 | 5.251 | 7.452 | 8.245 |
| EfficientNet-B0 | **5.204** | **7.133** | **7.176** |
| ResNet18 with balanced sampling | 5.296 | 7.470 | 9.289 |

Moderate balanced sampling did not improve the overall or 60+ result in this run, so
EfficientNet-B0 remains the strongest evaluated checkpoint. Full overall and sliced metrics
are stored in `reports/`.

## Deploy on Render

The repository includes a `render.yaml` Blueprint for a Docker-based Render web service.
In Render, create a new Blueprint, connect this GitHub repository, and apply the detected
service configuration. Render deploys after the GitHub checks pass and exposes the app on
an `onrender.com` URL.

The free service is suitable for a portfolio demonstration but sleeps after inactivity and
can have a noticeable cold start. If the 512 MB free instance cannot accommodate PyTorch,
select a Render plan with at least 2 GB of memory.

## Dataset and responsible use

UTKFace encodes `age_gender_race_...jpg` in filenames. Its labels and demographic
categories have limitations and may not reflect self-identified attributes. Dataset splits
are stratified by age group where possible. Always report per-slice results alongside the
overall score. Predictions describe appearance only and must not be treated as chronological
age or used for consequential decisions.
