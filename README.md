# SAYL | سَيْل
### Intelligent Multi-Agent Flood Rescue System

**Course:** CAI461 — Agent-Based Systems  
**University:** Princess Nourah bint Abdulrahman University  
**Section:** 72U  
**Technology:** Python, Streamlit, scikit-learn, LangSmith (observability)

## Project Overview

**SAYL** is a Python-based multi-agent flood rescue simulation set in a synthetic urban environment inspired by Jeddah. The system processes Arabic distress reports, estimates incident severity, prioritizes cases, and coordinates drones, rescue boats, and shelters through cooperating agents. Changes in water depth, road access, resource availability, and shelter capacity affect simulated decisions.

SAYL is an **academic simulation**, not a deployed emergency-response or dispatch service.

## Key Capabilities

- Arabic distress-message analysis with a hybrid rule-based and machine-learning severity classifier (character TF-IDF and Logistic Regression).
- Incident prioritization using severity, waiting time, and flood conditions.
- Inter-agent communication with ACL-like messages and Contract Net task allocation.
- Flood progression, road closures, shelter availability, and agent failure recovery.
- Rescue accounting that distinguishes **rescued**, **sheltered**, and **temporary staging**.
- Interactive Streamlit dashboard with simulation controls, agent status, incident tracking, and a searchable ACL log.
- **Optional LangSmith tracing and evaluation** of actual severity-analysis and Contract Net decisions.

## Agents

| Agent | Type | Responsibility |
|---|---|---|
| Distress Agent | Reactive | Interprets Arabic reports and creates prioritized incidents. |
| Command Center | Deliberative | Prioritizes tasks, gathers offers through Contract Net, and coordinates rescue and shelter allocation. |
| Flood Monitor | Reactive | Tracks flood risks, road closures, and shelter safety. |
| Drone | Hybrid | Verifies incidents and reports field observations. |
| Rescue Boat | Hybrid | Picks up and transports residents to shelters or temporary staging. |
| Shelter | Reactive | Reserves places, confirms admission, or refuses unsafe/full requests. |

Agents use ACL-like performatives, including `REQUEST`, `INFORM`, `CFP`, `PROPOSE`, `ACCEPT`, `REJECT`, `AGREE`, `REFUSE`, and `FAILURE`.

## Installation

**Recommended:** Python 3.11 or 3.12 (the project uses Python 3.10+ compatible syntax). Install the dependencies in a virtual environment if possible.

```bash
git clone https://github.com/lenahaobani12-dev/SAYL-Multi-Agent-Flood-Rescue.git
cd SAYL-Multi-Agent-Flood-Rescue
python -m pip install -r requirements.txt
```

For the supplied ZIP, extract the project folder and run the `pip install` command from the folder containing `app.py` and `requirements.txt`.

## Run the Dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`). From the **right-side Mission Control** panel, choose a scenario, adjust any needed settings, and select **Apply scenario / Reset**. Use **Run**, **Pause**, and **Step** to control the simulation. Scenario presets can be overridden by disabling **Use scenario preset settings**. You can also enter a simulated Arabic distress report and inspect the resulting incidents and ACL messages.

### Scenarios

| Scenario | Purpose |
|---|---|
| Normal Operation | Verifies reports, assigns rescuers, and confirms shelter admission. |
| Resource Conflict | Tests multiple simultaneous incidents competing for limited rescue resources and shelter places. |
| Agent Failure | Tests failure reports and reassignment of drone and boat tasks. |

## Testing and Local Evaluation

```bash
python -m pytest -q
python run_headless.py
python run_headless.py 3 100
python evaluate_nlp.py
```

`run_headless.py` runs the scripted scenarios without the dashboard. `evaluate_nlp.py` reports agreement with the 100 labelled Arabic messages distributed with the project; these in-sample results are not an independent generalization estimate.

## LangSmith Integration (Lab 9 Concepts)

LangSmith provides **observability**, not an additional rescue agent or a replacement for the existing decision logic. The implementation traces two real operations:

- `SAYL_Severity_Analysis` — inputs and outputs of the Arabic severity classifier in `sayl/severity.py`.
- `SAYL_Contract_Net_Decision` — candidate proposals, selected drone/boat, and estimated arrival time in `sayl/agents/command_center.py`.

These traces support inspection of **inputs, outputs, errors, and latency**. The LangSmith dashboard also provides **Monitoring/Analytics** for run counts and timing.

### Enable tracing (optional)

1. Sign in at https://smith.langchain.com and generate a **LangSmith API key**.
2. Configure the following environment variables **locally** before running Python. Do not put keys in the source code, ZIP, screenshots, or GitHub.

**Windows PowerShell:**

```powershell
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_PROJECT = "SAYL-Flood-Rescue"
$secret = Read-Host "LangSmith API Key" -AsSecureString
$env:LANGSMITH_API_KEY = [System.Net.NetworkCredential]::new("", $secret).Password
Remove-Variable secret
```

**macOS/Linux (bash/zsh):**

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=SAYL-Flood-Rescue
read -rs -p "LangSmith API Key: " LANGSMITH_API_KEY; echo
export LANGSMITH_API_KEY
```

3. In that **same terminal**, run `python run_headless.py 1 100` (or launch the dashboard), then inspect **Tracing** under the `SAYL-Flood-Rescue` project in LangSmith.

Tracing requires an API key and network access. The core local simulation does not rely on LangSmith's remote service for rescue decisions. Turn it off with `LANGSMITH_TRACING=false` or unset the variable when tracing is unnecessary.

### LangSmith evaluation

In **Datasets & Experiments**, create a dataset called **`SAYL-Arabic-Severity`** if it does not already exist. Then, with a valid API key available in the environment, run:

```bash
python evaluate_langsmith.py
```

The script uploads the project's 100 labelled messages only if the dataset is empty, evaluates the existing `analyze()` function, and records an experiment with the **`severity_exact_match`** evaluator. In the documented LangSmith run, the experiment completed **100/100** examples and measured **80% exact agreement**. This is **in-sample agreement** because the classifier trains on the same bundled examples; it is **not** held-out test accuracy. The report separately presents five-fold cross-validation (69% exact match for the hybrid method), which measures a different evaluation protocol. These percentages must not be compared as if they were obtained under identical test conditions.

Use only **synthetic distress messages** for cloud traces, because input text and outputs may be transmitted to LangSmith.

## Project Structure

```text
SAYL-Multi-Agent-Flood-Rescue/
├── .streamlit/config.toml
├── app.py                       # Streamlit dashboard
├── run_headless.py              # Scripted simulations
├── evaluate_nlp.py              # Local severity agreement
├── evaluate_langsmith.py        # LangSmith dataset + experiment
├── check_langsmith.py           # Local trace-emission example
├── requirements.txt
├── README.md
├── data/distress_messages.json
├── sayl/
│   ├── acl.py
│   ├── environment.py
│   ├── model.py
│   ├── scenarios.py
│   ├── severity.py
│   ├── ui_theme.py
│   └── agents/
│       ├── base.py
│       ├── boat.py
│       ├── command_center.py
│       ├── distress_agent.py
│       ├── drone.py
│       ├── flood_agent.py
│       └── shelter.py
└── tests/
```

## Academic Scope and Limitations

The terrain, reports, flood dynamics, rescue activities, and shelter operations are simulated. Scenario results illustrate multi-agent coordination under controlled assumptions and are not validated for live emergency deployment. LangSmith metrics describe the traced processes and should not be interpreted as evidence of field operational readiness.
