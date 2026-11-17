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

# AttendIQ – Maple Physio No-Show Prediction

## What it does

AttendIQ estimates the probability that a booked appointment at the fictional Maple Physio clinic will be missed, then recommends whether to make a reminder call. Under the assignment assumptions, a call costs **$8**, an uncalled no-show costs **$35**, and a called patient attends. These are simplified assumptions, not measured real-world effects.

**Live application:** https://huggingface.co/spaces/mmatthewkaya/AttendIQ-MidTerm

**Recorded group presentation:** https://drive.google.com/file/d/1KzGgVZEtkIJD1bAbGTs13556CzPMdMZG/view?usp=sharing

## How to run it locally

Use Python 3.12 and Conda from the repository root:

```bash
conda env create -f environment.yml
conda activate midterm-no-shows
python -m noshows.train
gradio app.py
```

Training creates `models/no_show.joblib`, `models/model_info.json`, and the evaluation reports. Keep the scikit-learn version in `requirements.txt` consistent with the saved model metadata. Alternatively, install pinned dependencies with `python -m pip install -r requirements-dev.txt` using Python 3.12.

The interface can start without trained artifacts, but predictions require both saved model files.

## Team and responsibilities

| Member | Contributions |
| --- | --- |
| Muharrem Kaya | Project lead and integration; Gradio interface, predictor, cost evaluation, threshold optimization, deployment, and live application demonstration |
| Ibrahim Gebrel | Instructor-provided data module integration, model training, cross-validation, model comparison, and presentation of machine learning approach and business results |
| Stebin George | Appointment validation class, appointment tests, saved-model integration checks, and presentation of the business problem and limitations |

The repository history records contributions from all three members, each with at least three non-merge commits on `main`.

## Data and feature decisions

The model uses information available at the time of booking. The outcome and post-visit fields are not used as prediction inputs.

| Source column | Decision / accepted values | Reason |
| --- | --- | --- |
| `appointment_id` | Excluded | Identifier, not a predictive booking feature. |
| `lead_days` | Included: integer 0–120 | Time between booking and appointment. |
| `age` | Included: integer 12–110 | Patient age at booking. |
| `prior_no_shows` | Included: integer 0–30 | Prior missed appointments. |
| `hour` | Included: integer 8–17 | Scheduled start hour. |
| `weekday` | Included: `mon`, `tue`, `wed`, `thu`, `fri` | Appointment day. |
| `booking` | Included: `online`, `phone`, `front_desk` | Booking channel. |
| `insurance` | Included: `private`, `provincial`, `self_pay` | Coverage type. |
| `treatment` | Included: `physio`, `massage`, `acupuncture` | Planned service. |
| `first_visit` | Included: `0` or `1` | First-visit indicator. |
| `distance_km` | Included: 0.1–150 or blank | Missing values accepted with a warning and median imputation. |
| `fee_charged` | Excluded from production | Post-visit information; would introduce data leakage. |
| `no_show` | Target only: `0` or `1` | Outcome to predict. |

Whole-number fields require finite integers. The `Appointment` validation layer rejects out-of-range values and unseen categories with field-specific errors. Although the encoder can ignore unknown categories internally, the application explicitly rejects them before prediction.

## Model comparison

Four pipelines were evaluated using five-fold stratified cross-validation on the 3,000-row training split. The values below are mean ± standard deviation.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy Classifier | 0.5000 ± 0.0000 | 0.7383 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Logistic Regression (`class_weight="balanced"`) | 0.7273 ± 0.0120 | 0.6750 ± 0.0086 | 0.4225 ± 0.0105 | 0.6599 ± 0.0306 | 0.5150 ± 0.0151 |
| Random Forest | 0.6973 ± 0.0072 | 0.7497 ± 0.0057 | 0.5582 ± 0.0271 | 0.2064 ± 0.0250 | 0.3007 ± 0.0290 |
| HistGradientBoosting | 0.6725 ± 0.0125 | 0.7413 ± 0.0098 | 0.5113 ± 0.0342 | 0.2752 ± 0.0177 | 0.3576 ± 0.0222 |

Logistic Regression was selected because it had the strongest candidate ROC-AUC and substantially better recall than the tree-based candidates. The Dummy Classifier's higher accuracy than some models is misleading because it identifies no actual no-shows.

GridSearchCV compared eight combinations of `C` (0.01, 0.1, 1.0, 10.0) and balanced/unweighted class weights, using ROC-AUC. The selected settings were **`C=0.1`** and **`class_weight="balanced"`**, with best CV ROC-AUC **0.72745**. The gain over the untuned balanced Logistic Regression (0.72729) was much smaller than cross-validation variability.

## Threshold and test results

The decision threshold was chosen by minimizing the assumed operating cost on out-of-fold predictions from the **training set only**, searching thresholds from 0.05 through 0.95 inclusive. Ties select the lowest threshold. The chosen threshold was **0.48**.

On the untouched **1,000-appointment synthetic test set**:

| Metric | Result |
| --- | ---: |
| ROC-AUC | 0.7283 |
| Accuracy | 66.60% |
| Precision | 41.74% |
| Recall | 69.47% |
| F1 | 0.5215 |

### Estimated operating cost

| Policy | Cost |
| --- | ---: |
| Call nobody | $9,170 |
| Call everybody | $8,000 |
| AttendIQ recommendation policy | **$6,288** |

The model-based policy cost **$2,882 less than calling nobody** and **$1,712 less than calling everybody** on this sample under the assignment assumptions. These figures are estimates, not validated real-clinic savings.

### Data leakage check

A separate training-only five-fold experiment produced ROC-AUC **0.7275 ± 0.0118** with production features versus **0.9232 ± 0.0064** when `fee_charged` was added. Because the fee is recorded after the visit, it is excluded from production even though it artificially increases apparent performance.

Detailed results and the split/threshold metadata are saved in `reports/metrics.json` and `models/model_info.json`.

## Tests

The tests cover data preparation, feature processing, costs, input validation, prediction contracts, saved-model integration, application startup, and deployment metadata.

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

The final development review reported **141 passing tests**.

## Deployment

AttendIQ is publicly deployed as a Gradio application on Hugging Face Spaces, using **Gradio 6.31.0**, **Python 3.12**, and saved model artifacts in `models/`.

**Live app:** https://huggingface.co/spaces/mmatthewkaya/AttendIQ-MidTerm

Deployment was verified on **November 15, 2026**: the Space was reported as Running, and a sample prediction returned **58.1% no-show probability** with the recommendation **Call this patient**, matching the local application. This is the recorded verification date, not a claim of continuous availability.

## Group presentation and demonstration

The final presentation was prepared and recorded by **Muharrem Kaya, Stebin George, and Ibrahim Gebrel**. It covers six slides:

| Slide | Topic | Presenter |
| --- | --- | --- |
| 1 | Introduction and project overview | Muharrem Kaya |
| 2 | Business problem and cost assumptions | Stebin George |
| 3 | Live Gradio application: valid prediction and invalid input validation | Muharrem Kaya |
| 4 | Machine learning workflow and model evaluation | Ibrahim Gebrel |
| 5 | Business cost comparison and estimated savings | Ibrahim Gebrel |
| 6 | Limitations and conclusion | Stebin George |

The live demonstration shows a valid booking that returns a no-show probability and call recommendation, followed by an invalid age input to demonstrate field-specific validation. The presentation also explains the threshold, held-out test results, cost assumptions, and limitations.

**[Watch the recorded presentation](https://drive.google.com/file/d/1KzGgVZEtkIJD1bAbGTs13556CzPMdMZG/view?usp=sharing)**

## Limitations and future work

The model was trained and tested on instructor-generated synthetic data, not a representative sample of real clinical appointments. Estimated costs depend on one test split and simplified assumptions about call effectiveness; a prediction does not establish that reminder calls cause attendance. Predictions can be wrong, and the model cannot consider circumstances absent from the booking data. Results may not transfer to another clinic or patient population.

Future work would require permissioned real-world data, ongoing validation, monitoring for performance changes, and direct measurement of reminder-call effectiveness before operational use.
