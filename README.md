# URL
https://aci606-cloud-fleet-challenge-bdve.onrender.com

# ACI606 - Cloud Fleet Challenge

Starter repository for **Exercise 4** of ACI606 Cloud Computing.

## Repository name

Use:

`aci606-cloud-fleet-challenge`

## The idea in one sentence

A virtual fleet sends telemetry to a remote platform. Your group must transform the starter into a **persistent Cloud service** that stores the data, trains and serves an ML model, exposes metrics, and presents the fleet in a dashboard.

## What is already provided

- Virtual multi-robot fleet simulator.
- Labelled dataset generator.
- FastAPI starter service.
- Working endpoints `/health`, `/telemetry`, `/robots` and a RAM-only `/history`.
- Minimal web dashboard at `/dashboard`.
- Training script skeleton.
- Smoke test.

## What your group must build

### 1. Persistent ingestion
Replace the starter RAM storage with persistent storage. The service must not lose the complete history when a client disconnects.

Required endpoints:

- `GET /health`
- `POST /telemetry`
- `GET /robots`
- `GET /robots/{robot_id}/history`

### 2. ML training
Generate or collect a training dataset, perform EDA, compare at least two classifiers, justify the final choice, and save a reproducible model pipeline.

Target classes:

- `NORMAL`
- `WARNING`
- `CRITICAL`

Minimum evidence:

- class distribution;
- confusion matrix;
- macro F1;
- train/test separation;
- technical justification of the selected model.

### 3. Cloud inference service
Implement:

`POST /predict`

Input contract:

```json
{
  "robot_id": "R03",
  "timestamp": 1780000000.0,
  "x": 4.21,
  "y": 7.82,
  "speed": 0.63,
  "battery": 71.0,
  "motor_temp": 48.2,
  "vibration": 0.43,
  "motor_current": 3.7,
  "obstacle_distance": 0.42
}
```

Required response contract:

```json
{
  "robot_id": "R03",
  "predicted_status": "WARNING",
  "confidence": 0.87,
  "model_version": "v1",
  "inference_ms": 12.4
}
```

### 4. Dashboard
Extend `/dashboard` so that an operator can see, at minimum:

- connected robots;
- most recent telemetry;
- predicted condition;
- battery;
- at least one historical or aggregate indicator;
- API/service status.

### 5. Observability
Extend `GET /metrics` with meaningful operational data. At minimum:

- number of telemetry requests;
- number of prediction requests;
- errors;
- average prediction latency;
- service uptime or equivalent operational indicator.

### 6. Git traceability
Your repository must show real development history:

- meaningful commits;
- at least one feature branch;
- merge into `main`;
- README describing how to run the system.

### 7. Remote execution
The final system must be remotely consumable. A purely local notebook is not a Cloud solution.

## Quick start

### A. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In Codespaces you can normally use the existing Python environment instead of creating another one.

### B. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `/docs`
- `/dashboard`

### C. Start the virtual fleet

In another terminal:

```bash
python simulator/fleet_simulator.py --api http://127.0.0.1:8000 --robots 5
```

The dashboard should begin to show robots.

### D. Generate training data

```bash
python simulator/generate_dataset.py --samples 3000
python ml/train_model.py
```

`train_model.py` is intentionally incomplete.

## First checkpoint - end of kickoff class

Your group should have:

- repository cloned/open in Codespaces;
- API running;
- simulator connected;
- `/dashboard` showing multiple robots;
- first meaningful Git commit;
- feature branch created;
- responsibilities divided within the group.

## Final challenge

The instructor may evaluate your service using an unseen fleet scenario. The assessment can consider functionality, prediction quality, latency, robustness and the technical defence of the architecture.

## Important

The robot simulator is **not** the deliverable. It is a client/load generator.

The deliverable is the **Cloud platform** your group builds around it.
