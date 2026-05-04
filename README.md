# Coup Detection

Observer + advisor tool for the game Coup. It replays public events, tracks role probabilities, and recommends challenges.

## Reading
Please have a look at `documents/paper.pdf`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run CLI

```bash
python -m coup.cli --events data/sample_game.json
```

Run CLI from vision pipeline output:

```bash
python -m coup.cli --vision-pipeline data/pipeline_output.json
```

Run CLI from vision pipeline output with frame-to-frame diff:

```bash
python -m coup.cli \
  --vision-pipeline data/pipeline_output.json \
  --previous-state data/prev_state_from_vision.json
```

### Hidden Role Probabilities

`Hidden Role Probabilities` is the model's current probability distribution for each player's still-hidden influence.
Revealed cards are listed separately in the `Revealed Roles` table.

- Standard Coup (default): duplicate roles in a player's hand are allowed, so a revealed role can still have non-zero hidden probability.
- Strict no-duplicate-hand: when a player has one hidden influence left, the revealed role is forced to zero probability and the other roles are renormalized.

Strict mode example:

```bash
python -m coup.cli --events data/sample_game.json --strict-no-duplicate-hand
```

## Simulation + Visualization

Generate synthetic Coup games (JSON events compatible with the engine):

```bash
python -m coup.sim.cli --players 4 --seed 123 --games 50 --out data/sim --trace
```

Outputs:

- `data/sim/game_<n>.json`: event log for one simulated game
- `data/sim/summary.csv`: per-game metrics (winner, turns, challenges, challenge accuracy, advisor agreement, reveal-probability bins)
- `data/sim/trace_<n>.json` (when `--trace` is set): replay snapshots (public state + belief + advisor)

Aggregate advisor decision metrics from simulation summary:

```bash
python -m coup.research.advisor_eval --summary data/sim/summary.csv --out data/research/advisor_eval
```

This writes:
- `advisor_game_rows.csv`
- `advisor_metrics_summary.csv`
- `advisor_report.json`

Open Streamlit and use the `Simulate` tab to run game batches, pick a game, and replay step-by-step with:

- current public state
- hidden-role belief table
- advisor recommendations on claim events
- full event timeline

Sample simulated game:

```bash
cat data/sim_example.json
```

Simulation simplification note: when a challenged player proves a role, this simulator does not swap that card with the deck.

## Research Dataset + Baseline

Build a training dataset from simulated games:

```bash
python -m coup.research.dataset --sim-dir data/sim --out data/research/research_dataset.csv --horizon 5
```

Build Markov transition dataset (state, action, reward, next_state, done):

```bash
python -m coup.research.markov --sim-dir data/sim --out data/research/markov_transitions.csv
```

Use the minimal RL environment wrapper on transitions:

```python
from pathlib import Path
from coup.research.rl_env import OfflineCoupRLEnv

env = OfflineCoupRLEnv.from_csv(Path("data/research/markov_transitions.csv"))
state = env.reset()
step = env.step(0)  # action index into [Income, Foreign Aid, Tax, Steal, Assassinate, Coup, Exchange]
print(len(state), step.reward, step.done, step.info)
```

Train a behavior-cloning policy and benchmark against random:

```bash
python -m coup.research.bc --transitions data/research/markov_transitions.csv --out data/research/bc --seed 42
```

Train a minimal DQN policy and benchmark against random (optionally compare with BC benchmark):

```bash
python -m coup.research.rl_train \
  --transitions data/research/markov_transitions.csv \
  --out data/research/dqn \
  --episodes 200 \
  --seed 42 \
  --compare-bc data/research/bc
```

Dataset labels:

- `y_next_action`: next action class (`Income`, `Foreign Aid`, `Tax`, `Steal`, `Assassinate`, `Coup`, `Exchange`, `NONE`)
- `y_next_is_challenge`: whether next event is a challenge
- `y_coup_within_horizon`: whether a Coup occurs in the next `horizon` events
- `y_current_claim_challenged`: for claim events only (`1` challenged, `0` not challenged, `-1` not applicable)

Train a baseline model pack (requires pandas + scikit-learn in training environment):

```bash
python -m coup.research.train --dataset data/research/research_dataset.csv --out data/research/models
```

Generate an evaluation artifact pack (metrics table + confusion matrices + calibration CSVs):

```bash
python -m coup.research.report \
  --dataset data/research/research_dataset.csv \
  --models data/research/models \
  --out data/research/report \
  --bins 10
```

Generate visual graphs from report artifacts:

```bash
python -m coup.research.plot --report-dir data/research/report --out data/research/plots
```

## Run the new FastAPI + React UI

```bash
# Terminal 1 — API
source .venv/bin/activate
uvicorn server.main:app --reload

# Terminal 2 — frontend
cd web && npm install && npm run dev
# open http://localhost:5173
```

Three tabs:

- **Match replay** — load `data/sample_game.json` or any event-schema JSON and step through it.
- **Manual play** — pick 2–6 player names, then drive the game turn-by-turn via the `TurnBuilder`. Apply, undo, or reset events against a live session; the advisor card updates after every applied claim.
- **Simulation lab** — batch-run sims via the `/api/sim/run` endpoint.

Key API routes: `/api/health`, `/api/games`, `/api/games/{sid}/events|replay|reset`,
`/api/advisor/recommend`, `/api/sim/run`, `/api/sim/bots`, `/api/research/metrics`.
Full OpenAPI schema at `/docs`.

## Run legacy Streamlit (retired)

```bash
streamlit run app/streamlit_app.py
```

### Vision sample frame

Place a sample frame at `data/test.png`, then in the app open “Vision Capture + OCR” and click “Load sample frame”.

## CV Dataset (Phase 1)

We start with a simple object detection dataset for `card_back`, `card_face`, and `coin`.

```bash
python -m vision.dataset init --root data/cv_dataset
python -m vision.dataset validate --root data/cv_dataset
python -m vision.train_placeholder --root data/cv_dataset
```

Label images in YOLO format under `data/cv_dataset/images` and `data/cv_dataset/labels`.

## Role Classifier (Phase 2)

Build a simple 5-role classification dataset from role images:

```bash
python -m vision.role_dataset --source data/cv_dataset/images/train --output data/role_dataset
```

Train classifier:

```bash
python -m vision.train_role_classifier --data data/role_dataset --model yolov8n-cls.pt
```

## Vision Pipeline (Phase 3)

Run detector + role classifier + player-zone summary on one image:

```bash
python -m vision.pipeline \
  --image data/cv_dataset/images/val/table_1.png \
  --detector model/best.pt \
  --classifier runs/classify/training_result/role_classifier/train/weights/best.pt \
  --zones data/player_zones.example.json \
  --output-json data/pipeline_output.json
```

Edit `data/player_zones.example.json` to match your table layout.

Convert pipeline output into Coup `PublicState` snapshot + generated events:

```bash
python -m vision.state_adapter --pipeline data/pipeline_output.json --output data/state_from_vision.json
```

Emit only frame-to-frame diff events (coin deltas + newly revealed roles):

```bash
python -m vision.state_adapter \
  --pipeline data/pipeline_output.json \
  --previous data/prev_state_from_vision.json \
  --output data/state_from_vision.json
```

## Camera Test App

Use webcam capture in Streamlit and run detector + role classifier on the captured frame:

```bash
streamlit run app/camera_test_app.py
```

## OCR Prerequisite (optional)

The vision module uses `pytesseract`. Install the system Tesseract binary:

```bash
brew install tesseract
```

## Run Tests

```bash
pytest
ruff check .
```


# NOTE, debug for CV, Vision Capture and OCR, make sure the image can present the right one
# NOTE:  STRATEGY OPTIMIZATION FOR EACH ROUND, AND EACH CHALLENGING in order to advise players with few of strategy


# WHY TO CHOOSE COUP, WHAT MAKES IT SPECIAL -> USE GAME THOERY, NASH, DILLEMA. put it in introduction
# BASE LINE, imrpove random point
# NOTHING TO COMPARE, SO IT DOESNT TELL THAT THE MODEL IS GOOD, IMPROVE THIS
# IF yes, we can try and explore some of the state - Thompson Sampling for the bandit


# Change April 18th
# Fix the manual edit, like who challenge who button every single time, for example P2 select steal, and then model came up like is anybody challenge or something Manually revealed/challenge button the simulation, by edit it.
# Manually set the player as the main character, for example me as P2 or P3, maybe before the game I will be Pth and then start the game. And me, ith Player who will try to observe the game
# And remember, if a player exchanges card, the probability might be distrubed, I dont really know
# If any plan or future improvement, please let me know or ask me
