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
an uncalled no-show costs $35. The public application is [AttendIQ-MidTerm on
Hugging Face Spaces](https://huggingface.co/spaces/mmatthewkaya/AttendIQ-MidTerm).

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
| Muharrem Kaya | Project lead and integration; Gradio app, predictor, cost evaluation, threshold optimization, and deployment |
| Ibrahim Gebrel | Instructor-provided data module integration, model training, cross-validation, and model comparison |
| Stebin George | Appointment validation class, appointment tests, and saved-model integration checks |

Contributions above reflect the implementation and authorship visible in the
repository history. The three named members each have at least three
non-merge commits on `main`.

## Decisions

The source table contains the following columns. Features are available when
the appointment is booked; the target and post-visit fields are kept out of
the production input.

| Source column | Model decision and accepted values | Reason |
| --- | --- | --- |
| `appointment_id` | Excluded | Identifies a row; it is not a booking characteristic and should not drive a prediction. |
| `lead_days` | Included, numeric, 0–120 whole days | Days from booking to visit; known at booking. The input bound follows the assignment. |
| `age` | Included, numeric, 12–110 whole years | Patient age is available at booking; values outside the assignment's supported range are refused. |
| `prior_no_shows` | Included, numeric, 0–30 whole appointments | Prior attendance history is available at booking; values outside the supported range are refused. |
| `hour` | Included, numeric, 8–17 whole hours | Appointment start time; accepted range follows the assignment. |
| `weekday` | Included, categorical: `mon`, `tue`, `wed`, `thu`, `fri` | Weekday is known at booking; unsupported or misspelled categories are refused. |
| `booking` | Included, categorical: `online`, `phone`, `front_desk` | Booking channel is known at booking; only the listed categories are accepted. |
| `insurance` | Included, categorical: `private`, `provincial`, `self_pay` | Coverage category is known at booking; only the listed categories are accepted. |
| `treatment` | Included, categorical: `physio`, `massage`, `acupuncture` | Planned treatment is known at booking; only the listed categories are accepted. |
| `first_visit` | Included, binary integer: `0` or `1` | First-visit status is known at booking; other values are refused. |
| `distance_km` | Included, numeric 0.1–150 km, or blank | Distance is available at booking but may be missing. Blank is accepted with a warning and imputed by the fitted median imputer. |
| `fee_charged` | Excluded; used only in the separate leakage experiment | The fee is recorded after the visit and is strongly tied to the outcome; it is unavailable at prediction time and would leak information. |
| `no_show` | Target only: `1` for no-show, `0` otherwise | This is the outcome being predicted and cannot be an input feature. |

Numeric inputs must be finite whole numbers where the field requires whole
numbers; `distance_km` accepts a finite number or a blank. `Appointment`
rejects invalid values and unseen categories with a field-specific error. The
one-hot encoder ignores unseen categories internally, so validation must reject
them before prediction rather than silently treating a typo as a new category.

## Model comparison

Four pipelines were compared with five-fold stratified cross-validation on the
3,000-row training split. Values below are mean ± standard deviation across
folds; accuracy is included because it can hide poor detection of the minority
no-show class.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy Classifier | 0.5000 ± 0.0000 | 0.7383 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Logistic Regression (`class_weight="balanced"`) | 0.7273 ± 0.0120 | 0.6750 ± 0.0086 | 0.4225 ± 0.0105 | 0.6599 ± 0.0306 | 0.5150 ± 0.0151 |
| Random Forest | 0.6973 ± 0.0072 | 0.7497 ± 0.0057 | 0.5582 ± 0.0271 | 0.2064 ± 0.0250 | 0.3007 ± 0.0290 |
| HistGradientBoosting | 0.6725 ± 0.0125 | 0.7413 ± 0.0098 | 0.5113 ± 0.0342 | 0.2752 ± 0.0177 | 0.3576 ± 0.0222 |

Logistic Regression was selected for its strongest candidate ROC-AUC and
substantially higher recall than the tree candidates. The Dummy Classifier's
0.7383 accuracy comes from predicting the majority class and yields zero
no-show recall, so accuracy alone would be misleading.

GridSearchCV used ROC-AUC and eight combinations: four `C` values (0.01, 0.1,
1.0, 10.0) crossed with unweighted and balanced class weights. The selected
settings were `C=0.1` and `class_weight="balanced"`, with best CV ROC-AUC
0.72745. Compared with the untuned balanced Logistic Regression mean of
0.72729, the gain is about 0.00016, far below its fold standard deviation of
0.01196; tuning did not improve ROC-AUC beyond the observed fold-to-fold spread.

## Result

The threshold was selected by minimizing operational cost over the inclusive
0.05–0.95 grid using only the 3,000 training rows' out-of-fold probabilities.
When thresholds have equal cost, the optimizer chooses the lowest threshold;
this deterministic rule favors calling more patients among tied policies. The
selected threshold is **0.48**.

On the untouched 1,000-row test split, the model achieved ROC-AUC **0.7283**,
accuracy **0.6660**, precision **0.4174**, recall **0.6947**, and F1 **0.5215**.
The call-none policy cost **$9,170**, call-all cost **$8,000**, and the model
policy cost **$6,288**. On this test split, the model policy costs $1,712 less
than call-all and $2,882 less than call-none.

The separate five-fold training-only leakage experiment had mean ROC-AUC
**0.7275** (SD **0.0118**) for the production features and **0.9232** (SD
**0.0064**) after adding `fee_charged`. That column is recorded after the visit
and reveals information about the outcome, so the larger score cannot be used
to judge a deployable model; `fee_charged` remains excluded from production.

These costs are estimates from one 1,000-row synthetic held-out sample. The
difference should be treated as uncertain and specific to the assignment's
data and assumed costs; it does not show that calls cause attendance. Full
results, including the split seed and selected threshold, are saved in
`reports/metrics.json` and `models/model_info.json`.

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

AttendIQ is deployed as a public Gradio application on Hugging Face Spaces:

**Live demo:** https://huggingface.co/spaces/mmatthewkaya/AttendIQ-MidTerm

The Space uses Gradio 6.31.0, Python 3.12, and the trained model
artifacts in `models/`.

Deployment was verified on November 15, 2026. The Space reported
**Running**, and a live prediction returned:

- No-show probability: **58.1%**
- Recommendation: **Call this patient**

These results matched the local application for the same appointment
inputs.


## Demo

Four-minute English demo plan using the public Space:

| Time | Speaker | Demo segment |
| --- | --- | --- |
| 0:00–0:45 | Muharrem Kaya | Introduce Maple Physio's no-show problem and the cost rule: $8 per call plus $35 for each no-show not called. |
| 0:45–1:35 | Ibrahim Gebrel | Enter a valid booking on the public Space; show the no-show probability and call recommendation, and explain that model selection used training-only cross-validation. |
| 1:35–2:15 | Stebin George | Enter an invalid booking (for example, age 111) and show the field-specific validation message; mention that blank distance is accepted with an imputation warning. |
| 2:15–3:15 | Muharrem Kaya | Compare the test-set costs: $9,170 call-none, $8,000 call-all, and $6,288 optimized policy. State the OOF-selected threshold of 0.48. |
| 3:15–4:00 | Stebin George, with Ibrahim Gebrel | Explain the main limitation: results use one synthetic held-out sample and assumed costs; a prediction does not establish that a call causes attendance or account for unrecorded circumstances. Ibrahim notes that `fee_charged` is post-visit and excluded from production. |

Use the public Space rather than a local copy and ensure the inputs shown and
spoken numbers match the report. Add a recording link here if the group records
the demo.

## Limitations

The model learns from instructor-generated synthetic appointments, not a
representative sample of real clinic operations. The reported cost difference
comes from one 1,000-row test split and should be treated as uncertain; it may
not carry over to another clinic, patient mix, or cost structure. The model
cannot account for circumstances absent from the booking fields, guarantee
attendance, or establish that a reminder call causes attendance. The separate
`fee_charged` experiment demonstrates why a post-visit field cannot be used at
booking time, even when it improves measured ROC-AUC.
