# AttendIQ

AttendIQ is a course project for predicting whether a patient will miss a scheduled appointment and estimating whether a reminder call is worthwhile.

## Project status

This repository contains the initialization scaffold. Feature modules, model artifacts, evaluation results, and deployment configuration are intentionally pending their owners' work. The instructor-provided `noshows/data.py` is not included because its source must be copied and verified exactly.

## Architecture contracts

- `noshows/data.py` will generate the deterministic source dataset and remain unchanged.
- `noshows/pipeline.py` will be the single source for feature metadata, preprocessing construction, training split/CV, and model comparison contracts. Honest training excludes `appointment_id` and `fee_charged`.
- `noshows/appointment.py` will own immutable input validation and convert an `Appointment` to one row with the ordered training features.
- `noshows/train.py` will orchestrate deterministic training, tuning, out-of-fold predictions, serialization, and metadata output.
- `noshows/evaluation.py` will calculate costs, choose the threshold from training-only out-of-fold probabilities, and evaluate the held-out test set once.
- `noshows/predictor.py` will load the fitted pipeline and metadata once, then return probability and action for validated appointments.
- `app.py` will derive UI controls from shared feature metadata and use the appointment/predictor interfaces.

These interfaces are planning contracts; implementation is assigned to the team guides under `/workspaces/team/`.

## Development

Install `requirements-dev.txt` in a Python 3.10+ environment. Run tests with `pytest`. Train with `python -m noshows.train` after the assigned modules and verified instructor data source are available.

## Assignment and AI disclosure

The assignment PDF remains the authoritative source for requirements and academic AI-use disclosure. Verify the complete PDF and required disclosure language before implementation or submission.
