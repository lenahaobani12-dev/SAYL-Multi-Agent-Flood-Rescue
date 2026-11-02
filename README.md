# SAYL (سَيْل) – Intelligent Multi-Agent Flood Rescue System

SAYL is a Python-based multi-agent system designed to simulate and coordinate emergency rescue operations during flash floods in a Saudi neighborhood, using Jeddah as the simulation context.

The system uses cooperating intelligent agents to analyze Arabic distress messages, prioritize emergency incidents based on severity, and coordinate drones, rescue boats, and shelters. Its main objective is to demonstrate how multi-agent collaboration can improve emergency prioritization, resource allocation, and decision-making under changing flood conditions.

**Course:** CAI461 – Agent-Based Systems  
**Section:** 72U  
**University:** Princess Nourah bint Abdulrahman University


## 1. System Overview

SAYL simulates a dynamic flood environment where water levels increase, roads become inaccessible, and rescue resources are limited.

The system integrates Arabic distress-message analysis, incident prioritization, autonomous rescue agents, and ACL-based communication within an interactive monitoring dashboard.

SAYL is an academic simulation and decision-support prototype, not a real emergency dispatch service.

## 2. Intelligent Agents

| Agent | Type | Responsibility |
|---|---|---|
| Distress Agent (NLP) | Reactive | Analyzes Arabic distress messages, identifies severity levels, and creates incidents. |
| Command Center | Deliberative | Prioritizes incidents, coordinates resources, and assigns rescue tasks. |
| Flood Monitor | Reactive | Monitors flood conditions and updates affected areas and infrastructure. |
| Drone Agent | Hybrid | Verifies incident locations and monitors affected areas while managing battery limitations. |
| Rescue Boat Agent | Hybrid | Performs rescue missions and transports affected people to available shelters. |
| Shelter Agent | Reactive | Monitors shelter capacity and accepts or refuses evacuation requests. |

## 3. Agent Communication

Agents coordinate through an ACL-like communication protocol.

The system supports the following message performatives:

- **REQUEST:** Requests an action from another agent.
- **INFORM:** Communicates information or status updates.
- **CFP:** Requests proposals for a task.
- **PROPOSE:** Offers to execute a requested task.
- **ACCEPT:** Accepts an agent's proposal.
- **REFUSE / REJECT:** Declines a request or rejects a proposal.
- **AGREE:** Confirms an agent's agreement to perform an action.
- **FAILURE:** Reports an unsuccessful operation or agent failure.

These messages enable coordination, resource negotiation, task allocation, and failure recovery.

## 4. Interactive Dashboard

SAYL includes a Streamlit-based dashboard that provides:

- Real-time simulation status and system indicators.
- Interactive flood map with water-level visualization.
- Road closures and rescue resource locations.
- Live status monitoring for drones, boats, shelters, and coordination agents.
- Arabic distress-message submission.
- Incident tracking and rescue assignments.
- ACL communication logs with filtering and CSV export.
- Adjustable simulation parameters, including rainfall, rescue resources, shelter capacity, and agent failure rate.
- Run, Pause, Step, and Reset controls.

## 5. Installation

### Requirements

- Python 3.9 or newer (Python 3.10+ recommended).
- Git.
- Python dependencies listed in `requirements.txt`.

### Step 1: Clone the Repository

```bash
git clone https://github.com/lenahaobani12-dev/SAYL-Multi-Agent-Flood-Rescue.git
cd SAYL-Multi-Agent-Flood-Rescue
```


### Step 2: Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```powershell
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 6. Running the System

Start the interactive dashboard using:

```bash
python -m streamlit run app.py
```

After starting the application, open the local URL shown in the terminal, typically:

`http://localhost:8501`

### How to Use SAYL

1. Select one of the available simulation scenarios.
2. Adjust the simulation parameters using the sidebar controls.
3. Apply or reset the selected scenario.
4. Press **Run** to begin automatic simulation or **Step** to advance one simulation tick.
5. Observe changes in the flood map, rescue resources, and agent status.
6. Submit a custom Arabic distress message and select its simulated location.
7. Monitor incident prioritization, rescue assignments, and ACL communication.
8. Use **Pause** to stop execution temporarily or **Reset** to restart the scenario.

## 7. Testing Scenarios

Three scenarios demonstrate the system's capabilities.

### Scenario 1: Normal Rescue Operations

Simulates a flood emergency with a small number of distress reports.

The agents analyze incoming messages, prioritize incidents, coordinate available rescue resources, and perform evacuation operations.

### Scenario 2: Multiple Distress Reports and Resource Conflicts

Simulates several emergency reports arriving within a short period while drones, boats, and shelter spaces are limited.

This scenario demonstrates incident prioritization, resource negotiation, and shelter-capacity constraints.

### Scenario 3: Agent Failure and Recovery

Simulates unexpected failures affecting rescue agents during emergency operations.

The system demonstrates how failure notifications are communicated and how the Command Center responds by attempting to reassign tasks.

### Scenario Completion

For scripted scenarios, the simulation automatically pauses when scheduled reports have been processed, incidents have been resolved, and required rescue operations have been completed.

Simulation results and completion times may vary depending on the configured parameters.

## 8. Testing and Evaluation

The project includes automated tests and command-line simulation tools.

Run the automated tests:

```bash
python -m pytest tests
```

Run the three scenarios without the graphical interface:

```bash
python run_headless.py
```

Run a specific scenario:

```bash
python run_headless.py 3 60
```

Evaluate the Arabic distress-message severity classifier:

```bash
python evaluate_nlp.py
```

These commands support functional verification, scenario execution, and evaluation of the distress-message analysis component.

## 9. Project Structure

```text
SAYL-Multi-Agent-Flood-Rescue/
│
├── app.py
├── README.md
├── requirements.txt
├── run_headless.py
├── evaluate_nlp.py
├── .gitignore
│
├── sayl/
│   ├── acl.py
│   ├── environment.py
│   ├── severity.py
│   ├── model.py
│   ├── scenarios.py
│   └── agents/
│       └── Agent implementation files
│
├── data/
│   └── distress_messages.json
│
└── tests/
    └── Automated test files
```

## 10. Technologies Used

- **Python:** Core system implementation.
- **Streamlit:** Interactive user interface.
- **Multi-Agent Systems:** Distributed decision-making and collaboration.
- **Arabic NLP:** Distress-message processing and severity classification.
- **TF-IDF and Logistic Regression:** Components of the severity-classification approach.
- **ACL-like Messaging:** Structured communication between intelligent agents.
- **GitHub:** Source-code management and team collaboration.


## 11. Project Scope

SAYL is developed for academic demonstration, experimentation, and training purposes.

The simulated neighborhood, environmental conditions, incident reports, and rescue resources are synthetic. The system does not connect to actual emergency response services or provide real-world rescue dispatch capabilities.


