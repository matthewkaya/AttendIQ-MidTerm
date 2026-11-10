---
title: AttendIQ Maple Physio no-show check
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12
app_file: app.py
---

# Maple Physio no-show check

## What it does

AttendIQ estimates the chance that a booked Maple Physio appointment will be a
no-show and shows whether a reminder call is recommended. The call costs $8;
an uncalled no-show costs $35. The public Hugging Face Space URL will be added
after a public Space is created and verified.

## How to run it locally

Use Python 3.12 and Conda. From the repository root:

```bash
conda env create -f environment.yml
conda activate midterm-no-shows
python -m noshows.train
gradio app.py
```

`python -m noshows.train` must create `models/no_show.joblib`,
`models/model_info.json`, and the reports from the source data. Keep the
scikit-learn version in `requirements.txt` identical to the version written to
`model_info.json`. To install the pinned dependencies with pip instead, run
`python -m pip install -r requirements-dev.txt` in Python 3.12.

The app can start without trained artifacts and will show a model-unavailable
message while keeping the input page available. Predictions require both
trained model files.

## Team

| Member | Owned parts |
| --- | --- |
| Muharrem Kaya | Project lead, integration, Gradio app, deployment |
| Ibrahim Gebrel | Machine learning and training |
| Stebin George | Appointment class and testing |
| Joshua Obiha | Evaluation, costs, and threshold optimization |

## Decisions

This section will document each source column, its feature/exclusion decision,
and the reason after model integration. The app's input limits follow the
assignment: lead days 0–120, age 12–110, prior no-shows 0–30, hour 8–17,
first visit 0 or 1, distance 0.1–150 km or empty, weekdays Monday–Friday, and
the listed booking, insurance, and treatment categories. Empty distance is
accepted and warned about because the pipeline imputes it. Invalid values and
unseen categories must be rejected by `Appointment` with a field-specific
error. `appointment_id` is an identifier and `fee_charged` is recorded after
the visit; neither may be used as a prediction feature.

## Model comparison

Pending Ibrahim's training results. Record the five-fold stratified training
comparison (including mean and standard deviation), chosen model, and tuning
comparison here. Do not report placeholder or fabricated scores.

## Result

Pending Joshua's held-out test evaluation. Record test metrics, the costs for
calling nobody, calling everybody, and using the model, plus the leakage check.
The selected threshold must come from training-only out-of-fold probabilities.
Explain uncertainty in the test-set cost difference after the real evaluation
is available.

## Tests

Tests cover deployment metadata and app startup as well as the data, feature,
cost, validation, and prediction contracts. From a fresh Python 3.12
environment, run:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Deployment

Create a **public** Hugging Face Space using the Gradio SDK and ZeroGPU (Free).
The Space settings above select Gradio 6.31.0, Python 3.12, and `app.py`.
Upload or push the repository contents, including both trained files under
`models/`; do not add an access token to the repository. Open the public Space
in a signed-out browser and verify a prediction before describing the Space as
live. The guarded `spaces.GPU` placeholder in `app.py` allows ZeroGPU's startup
check to find the required decorated function while preserving local imports
when the `spaces` package is unavailable.

## Demo

The four-minute demo should use the verified public Space. In order, explain
the problem and cost rule, enter a valid booking, show a rejected invalid
booking, explain the three test costs, and state one limitation. Ensure spoken
numbers match the final README results. Add the recording link here if the
group records the demo.

## Limitations

The prediction estimates risk from the appointment fields provided to the
model. It cannot account for unrecorded circumstances, guarantee attendance,
or establish that a call caused a patient to attend. Include the model's
measured limitations here after evaluation is complete.
