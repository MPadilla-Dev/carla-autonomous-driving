# CARLA AI Autopilot

An autonomous driving agent built on top of [CARLA 0.9.13](https://carla.org/), implementing a blackboard-style architecture across four modules: Monitor, Analyser, Planner, and Executor. Developed and tested on Town03.

---

## Requirements

- CARLA 0.9.13 server running locally on port 2000
- Python 3.7+
- CARLA's `PythonAPI/carla` folder on your `PYTHONPATH` (needed for `GlobalRoutePlanner`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the main demo

`ai_hudtest.py` is the primary entry point. It opens a pygame window with a live chase camera and telemetry overlay, and runs the vehicle through a milestone scenario.

```bash
python ai_hudtest.py -m 1   # Milestone 1: speed and steering control
python ai_hudtest.py -m 2   # Milestone 2: path planning (roundabout route)
python ai_hudtest.py -m 3   # Milestone 3: obstacle avoidance
```

Press `ESC` to quit. Recordings are saved automatically to `recordings/` next to the script.

---

## Tunable parameters

All tunables are class-level constants at the top of each class, marked with `# TUNABLE` comments. No changes to logic are needed — just flip the value and re-run.

### Steering algorithm — `Executor` in `ai_control.py`

```python
STEER_MODE = 'pure_pursuit'   # 'aim' or 'pure_pursuit'
```

- `aim` — original threshold-based steering: points the car directly at the next waypoint.
- `pure_pursuit` — smooth lookahead-based steering: aims at a point further ahead along the path, reducing oscillation on curves.

Pure pursuit lookahead distance is itself tunable:

```python
LOOKAHEAD_BASE = 4.0   # meters of lookahead at standstill
LOOKAHEAD_K    = 0.15  # extra meters per km/h of speed
```

### Speed control — `Executor` in `ai_control.py`

```python
THROTTLE_MODE = 'pid'   # 'pid' or 'threshold'
```

- `threshold` — bang-bang: full throttle below target speed, full brake above it.
- `pid` — smooth PID controller that holds target speed precisely.

PID gains are also tunable (`PID_KP`, `PID_KI`, `PID_KD`).

### Path planner — `Planner` in `ai_control.py`

```python
PLANNER_MODE = 'astar'   # 'greedy', 'astar', or 'none'
```

- `greedy` — picks the next waypoint closest to the goal at each step. Fast but fails at roundabouts and split intersections.
- `astar` — uses CARLA's `GlobalRoutePlanner` (A* on the full road topology graph). Handles roundabouts and complex junctions correctly. Recommended.
- `none` — no planning; executor steers directly at the raw destination coordinate.

### Traffic light handling — `Analyser` in `ai_parser.py`

```python
TRAFFIC_LIGHT_ENABLED = True
```

Set to `False` to run the car without any traffic light awareness (useful for comparing behaviour before and after M3).

### Obstacle avoidance — `Analyser` in `ai_parser.py`

```python
OBSTACLE_AVOIDANCE_ENABLED = True
AVOIDANCE_MODE = 'stopgo'   # 'brake', 'escape', or 'stopgo'
```

- `brake` — stop and wait for the obstacle to clear, then resume.
- `escape` — immediately attempt a lane change via an escape path.
- `stopgo` — brake first; if the obstacle is still there after `MAX_BRAKE_WAIT_FRAMES`, escalate to a lane-change escape.

Lidar detection zone is also tunable:

```python
THREAT_FORWARD_MIN = 2.0       # meters - ignore points right at the bumper
THREAT_FORWARD_MAX = 9.0       # meters - how far ahead to scan
THREAT_LATERAL_HALF_WIDTH = 1.25  # meters - width of the detection cone
THREAT_MIN_POINTS = 10         # minimum lidar returns to count as a real object
```

---

## Multi-vehicle stress test

To run several AI cars simultaneously around the Town03 roundabout:

```bash
python spawn_custom_npc_record.py -n 6
```

Each car is given a random destination on arrival and re-routed automatically. Collision and healing events are printed to the console. The session is recorded to `roundabout_test.log`.

To replay the session with a HUD:

```bash
python spawn_hudview.py
```

---

## File reference

### Core AI modules

| File | Purpose |
|---|---|
| `ai_knowledge.py` | Shared blackboard (the `Knowledge` class). All modules read from and write to this single object. Also defines the `Status` enum: `DRIVING`, `ARRIVED`, `HEALING`, `CRASHED`. |
| `ai_parser.py` | `Monitor` — spawns and reads sensors (lane invasion, lidar, traffic lights) into Knowledge every tick. `Analyser` — interprets Knowledge to set target speed, detect lidar threats, and trigger the HEALING state. |
| `ai_control.py` | `Planner` — builds waypoint paths (greedy or A*) and advances through them. `Executor` — computes and applies throttle, brake, and steering every tick using the chosen algorithms. Also contains `PIDController`. |
| `custom_ai.py` | `Autopilot` — wires all four modules together. The main script only needs to call `autopilot.update()` each tick and `autopilot.set_destination()` to drive. |

### Test and demo scripts

| File | Purpose |
|---|---|
| `ai_hudtest.py` | **Primary demo script.** Runs a milestone scenario with a pygame chase-camera window and live telemetry overlay. Supports `--no-record` and `--label` flags. |
| `ai_modtest.py` | Headless version of the milestone test. No pygame window; status is printed to the console. Useful for quick runs without a display. |
| `ai_test.py` | Earlier headless test script, slightly cleaner than `ai_modtest.py`. No recorder integration. |
| `spawn_custom_npc_record.py` | Spawns multiple AI-controlled vehicles near the Town03 roundabout, each driving to random destinations. Tracks and prints per-vehicle crash and healing counts. Records the session to `roundabout_test.log`. |
| `spawn_hudview.py` | Replays `roundabout_test.log` and attaches a pygame camera to the first vehicle it finds, with a basic speed and controls overlay. |

### CARLA utility scripts

| File | Purpose |
|---|---|
| `show_recorder_file_info.py` | Prints metadata about a `.log` recording file (actor list, frame count, etc.). Standard CARLA utility. |
| `start_recording.py` | Spawns a configurable number of CARLA Traffic Manager vehicles and starts a recording. Standard CARLA utility. |
| `start_replaying.py` | Replays a `.log` file with configurable start time, duration, camera actor, and time factor. Standard CARLA utility. |

---

## Architecture overview

```
Main loop (ai_hudtest / ai_modtest / spawn_custom_npc_record)
    │
    └── Autopilot.update()  [custom_ai.py]
            │
            ├── Monitor.update()    → writes sensor data to Knowledge
            ├── Analyser.update()   → reads Knowledge, sets target_speed / HEALING
            ├── Planner.update()    → advances waypoint queue, updates destination
            └── Executor.update()   → reads destination + target_speed, applies controls
```

All inter-module communication flows through the `Knowledge` blackboard. No module calls another directly, which makes it straightforward to swap algorithms (e.g. change `STEER_MODE`) without touching any other module.

---

## Team

Most of the work was done collaboratively. The split below reflects primary ownership rather than exclusive authorship.

**Matin Moradi — Control & Sensing**
- `ai_knowledge.py` — blackboard design, Status enum, arrival logic
- `ai_parser.py` — Monitor (lidar, lane invasion, traffic light sensors) and Analyser (threat detection, speed control, healing state machine)
- `ai_control.py` — Executor: both steering modes (aim-at-next, pure pursuit), both throttle modes (threshold, PID), and the PIDController class

**Manuel Padilla — Planning & Integration**
- `ai_control.py` — Planner: greedy and A* path building, escape path logic, path visualization
- `custom_ai.py` — Autopilot wiring, telemetry interface
- `ai_hudtest.py` — pygame HUD, chase camera, telemetry overlay, synchronous mode setup
- `spawn_custom_npc_record.py` — multi-vehicle stress test, per-vehicle collision and healing tracking
- Testing scripts (`ai_test.py`, `ai_modtest.py`, `spawn_hudview.py`)


## Demo videos
All milestone recordings are available here:
[Google Drive — Demo Videos](https://drive.google.com/drive/folders/1PNJ3RwJwwOncB9w1EClW7YTBrN3qZSEj?usp=sharing)