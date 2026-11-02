# SAYL (سَيْل) – Multi-Agent Flood Rescue System

SAYL is a Python multi-agent system that simulates a flash flood in a Saudi neighbourhood (Jeddah context).
Cooperating agents read Arabic distress messages, rank them by severity, and coordinate drones,
rescue boats and shelters so that the most critical cases are reached first.

CAI 461 – Agent-Based Systems, Project 2 · Section 72U · Princess Nourah bint Abdulrahman University

## Agents

| Agent | Type | Role |
|---|---|---|
| Distress Agent (NLP) | Reactive | Reads Arabic distress messages, extracts severity, people and tags, creates incidents |
| Command Center | Deliberative | Ranks incidents and assigns drones, boats and shelters (Contract Net Protocol) |
| Flood Monitor | Reactive | Watches water depth, escalates incidents in deep water, closes flooded shelters |
| Drone (×n) | Hybrid | Flies to a reported location and verifies it; limited battery |
| Rescue Boat (×n) | Hybrid | Bids for rescue tasks, picks people up, takes them to a shelter |
| Shelter (×3) | Reactive | Accepts or refuses people based on its free capacity |

Agents communicate only through ACL-like messages (`REQUEST`, `INFORM`, `CFP`, `PROPOSE`,
`ACCEPT`, `REJECT`, `REFUSE`, `AGREE`, `FAILURE`), which are shown in the message log.

## Installation

Requires Python 3.9 or newer.

```bash
git clone <repository-url>
cd sayl
python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Running the system

```bash
streamlit run app.py
```

The browser opens automatically. In the sidebar choose a scenario, adjust the sliders,
then press **Run**, **Step** or **Reset**. Type an Arabic distress message in the box at the top
(and choose its x/y position on the map) to see the agents react to it.

### Auto-stop of the three scripted scenarios (Person 4 integration update)

In the Streamlit UI, **Run** automatically pauses when **all scheduled distress reports**
have been processed, all incident cases have been resolved, and all rescue boats
have completed delivery to a shelter or emergency staging area. A scenario will
not stop prematurely while a later report or shelter transfer is still pending.
The dashboard then shows **Scenario complete** and the Tick counter stays fixed.
Choose another scenario and press **Reset** to load it; then press **Run**.

Under the standard seeded scenario settings, the completion ticks are approximately:
Normal operation **44**, Many reports **94**, Agent failure **41**.
These numbers change if you edit parameters or agent behavior.

**Free play** intentionally does not auto-stop; pause it manually. If you enable
ongoing random distress messages, the scenario is also open-ended.

The headless runner (`python run_headless.py 1 120`) intentionally runs the
specified number of ticks, which is useful for reproducible offline evidence.

Terminal mode (no interface), useful for testing:

```bash
python run_headless.py          # runs the three scenarios and prints the results
python run_headless.py 3 60     # scenario 3 for 60 ticks with the full message log
python evaluate_nlp.py          # accuracy of the Arabic severity analysis
python -m pytest tests          # automatic checks
```

## Testing scenarios

1. **Normal operation** – three messages; agents verify, rescue and shelter everyone.
2. **Many reports (conflict)** – eight messages in three ticks, few drones and boats, small shelters.
   Shows prioritisation by severity and shelters refusing when full.
3. **Agent failure** – a drone fails while flying and a boat gets stuck;
   the Command Center receives `FAILURE` and reassigns the tasks.

## Project structure

```
app.py                     Streamlit user interface
run_headless.py            terminal runner
sayl/acl.py                ACL message format and message bus
sayl/environment.py        flood grid, water level, incidents
sayl/severity.py           Arabic NLP: rules + TF-IDF/Logistic Regression severity classifier
sayl/model.py              builds and steps the simulation (SimParams = all parameters)
sayl/scenarios.py          the three testing scenarios
sayl/agents/*.py           one file per agent
data/distress_messages.json  100 labelled Arabic distress messages (severity 1-5)
evaluate_nlp.py            cross-validated evaluation of the severity analysis
tests/test_basic.py        automatic checks
```

## Team

| Member | Responsibility |
|---|---|
| Person 1 | Environment, Flood Monitor, report 4.1 / 4.2 / 4.4, poster |
| Person 2 | Distress Agent (NLP), Command Center |
| Person 3 | Drone, Rescue Boat, Shelter agents, conclusion |
| Person 4 | ACL messaging, scenarios, integration and testing |
| Person 5 | User interface, GitHub repository, abstract and final formatting |

## Phase 2 dashboard (Person 5)

The redesigned Streamlit dashboard includes a responsive operations header, live KPI cards,
a flood grid with road closures, drone/boat/shelter markers, agent status, mission progress,
filterable ACL message logs, CSV export, an incident/assignment table, Arabic distress input,
and adjustable controls for scenario selection, fleet size, rain, shelter capacity and failure rate.
One tick represents one simulated minute; this is **not** a live emergency response service.

To run from a Windows PowerShell terminal in the extracted `sayl` folder:

```powershell
cd "C:\Users\acer\Downloads\SAYL_COMPLETE\sayl"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

After changing scenarios, press **Apply scenario / Reset**. Leave **Use scenario preset settings** checked
for published repeatable scenario conditions; uncheck it to apply your fleet/capacity sliders.
Python 3.10+ is recommended. Captures from the browser (not the headless plots) should be included
in the final submission report.

### Repository handoff
Create a GitHub repository, upload the contents of this project folder at repository root,
replace `<repository-url>` in README with the actual clone URL, and invite all group members
and the instructor/teaching assistant through **Settings → Collaborators** using their real
GitHub usernames. Do not upload `.env`, passwords, personal access tokens or `.venv`.

### Team note
The included report is an integration draft from available materials. The team should review
and sign off their sections, add genuine screenshots from the final running UI, and verify
all factual claims before Blackboard submission.
