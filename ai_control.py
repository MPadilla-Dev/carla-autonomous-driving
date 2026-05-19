#!/usr/bin/env python
"""
ai_control.py
==============
Planner (decides WHERE to go) + Executor (decides HOW and applies controls).

WHAT'S IN HERE
--------------
1. PIDController       - reusable PID block, used for speed control.
2. Executor            - low-level controller. Two steering modes:
                         (a) AIM_AT_NEXT (original threshold-based)
                         (b) PURE_PURSUIT (smooth lookahead-based, recommended)
                         Speed control uses PID with throttle/brake mapping.
3. Planner             - high-level planner. Two path-building modes:
                         (a) GREEDY (fast, fails at roundabouts)
                         (b) ASTAR  (correct, slower)
                         Plus a small build_escape_path() for HEALING (M2).

PRESENTATION TIPS
-----------------
- Toggle STEER_MODE between 'aim' and 'pure_pursuit' in Executor to compare.
- Toggle PLANNER_MODE between 'greedy' and 'astar' in Planner to compare.
- All knobs are at the top of each class with TUNABLE comments.
"""

import glob
import os
import sys
import math
import heapq
from collections import deque

import numpy as np

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import ai_knowledge as data
from ai_knowledge import Status


# =============================================================================
# PID Controller
# =============================================================================
class PIDController(object):
    """
    Standard PID. error = target - measured. Output is unbounded; caller
    clamps as needed.

    Anti-windup: integral is clamped to ±integral_max.
    """

    def __init__(self, kp, ki, kd, integral_max=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error, dt):
        if dt <= 0:
            dt = 1e-3
        self.integral += error * dt
        # Anti-windup clamp
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max

        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# =============================================================================
# Executor - low-level vehicle control
# =============================================================================
class Executor(object):
    """
    Reads target waypoint + current state from Knowledge, computes
    throttle/steer/brake, applies to vehicle.

    Two steering algorithms available - flip STEER_MODE to compare.
    Speed is always PID-controlled.
    """

    # ---- TUNABLE: STEERING ALGORITHM ----------------------------------------
    # 'aim'          - simple: aim at next waypoint (original behavior).
    # 'pure_pursuit' - smoother: aim at a point LOOKAHEAD_DISTANCE meters
    #                  ahead along the path. Recommended.
    STEER_MODE = 'pure_pursuit'

    # ---- TUNABLE: PURE PURSUIT PARAMETERS -----------------------------------
    # Lookahead distance grows with speed: Ld = LOOKAHEAD_BASE + LOOKAHEAD_K * speed
    # Bigger Ld = smoother but cuts corners. Smaller Ld = aggressive, snake-y.
    LOOKAHEAD_BASE = 4.0   # meters at 0 speed
    LOOKAHEAD_K = 0.15     # meters of extra lookahead per (km/h) of speed
    WHEELBASE = 2.5        # nissan.micra wheelbase (m), used in pure pursuit
    MAX_STEER_RAD = 0.61   # ~35 deg, typical max steering angle

    # ---- TUNABLE: SPEED CONTROL ---------------------------------------------
    # PID gains. Defaults are conservative; tune per vehicle.
    PID_KP = 0.4
    PID_KI = 0.05
    PID_KD = 0.05

    # When PID output is positive, throttle. Negative -> brake.
    # The output of PID(error in km/h) is also roughly km/h-ish.
    # We normalize via THROTTLE_GAIN / BRAKE_GAIN.
    THROTTLE_GAIN = 0.10   # PID output of +5 km/h -> throttle 0.5
    BRAKE_GAIN = 0.05      # PID output of -10 km/h -> brake 0.5
    MAX_THROTTLE = 0.75
    MAX_BRAKE = 0.6

    # Default speed if Knowledge has none yet.
    DEFAULT_TARGET_SPEED = 30.0  # km/h

    def __init__(self, knowledge, vehicle):
        self.vehicle = vehicle
        self.knowledge = knowledge
        self.speed_pid = PIDController(self.PID_KP, self.PID_KI, self.PID_KD)

    # -------------------------------------------------------------------------
    # Main update loop
    # -------------------------------------------------------------------------
    def update(self, time_elapsed):
        """
        Drive when status is DRIVING or HEALING (escape needs movement).
        Otherwise apply hold-brake.
        """
        status = self.knowledge.get_status()
        if status == Status.DRIVING or status == Status.HEALING:
            dest = self.knowledge.get_current_destination()
            self.update_control(dest, [1], time_elapsed)
        else:
            self._apply_stop()

    def _apply_stop(self):
        """Bring the vehicle to a halt and reset PID so it doesn't wind up."""
        self.speed_pid.reset()
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        control.hand_brake = False
        self.vehicle.apply_control(control)

    # -------------------------------------------------------------------------
    # Steering
    # -------------------------------------------------------------------------
    def _steer_aim_at(self, vehicle_transform, target_loc):
        """
        Original-style steering: angle between forward vector and target.
        Returns steer in [-1, 1].
        """
        fwd = vehicle_transform.get_forward_vector()
        loc = vehicle_transform.location
        fwd_arr = np.array([fwd.x, fwd.y])
        tgt_arr = np.array([target_loc.x - loc.x, target_loc.y - loc.y])

        fwd_norm = np.linalg.norm(fwd_arr)
        tgt_norm = np.linalg.norm(tgt_arr)
        if fwd_norm < 1e-3 or tgt_norm < 1e-3:
            return 0.0

        fwd_unit = fwd_arr / fwd_norm
        tgt_unit = tgt_arr / tgt_norm
        dot = float(np.clip(np.dot(fwd_unit, tgt_unit), -1.0, 1.0))
        angle = math.acos(dot)
        cross = fwd_unit[0] * tgt_unit[1] - fwd_unit[1] * tgt_unit[0]
        if cross < 0:
            angle = -angle
        return max(-1.0, min(1.0, angle / (math.pi / 2)))

    def _steer_pure_pursuit(self, vehicle_transform, current_speed_kmh):
        """
        Pure pursuit: pick a point Ld meters ahead along the planned path,
        compute the steer angle that would arc the car to it.

        delta = atan(2 * L * sin(alpha) / Ld)
        where L = wheelbase, alpha = angle from car forward to target,
        Ld = lookahead distance.
        """
        # Pull the path from knowledge - Planner stores it under 'path_locations'
        path_locs = self.knowledge.retrieve_data('path_locations', None)
        if not path_locs:
            # Fallback to aim-at-next if no path is published
            target = self.knowledge.get_current_destination()
            target_loc = self._coerce_location(target)
            return self._steer_aim_at(vehicle_transform, target_loc)

        veh_loc = vehicle_transform.location
        Ld = self.LOOKAHEAD_BASE + self.LOOKAHEAD_K * current_speed_kmh

        # Walk the path until we accumulate Ld meters from the car.
        # Use 'sum of distances along the path' starting from the closest path
        # point ahead of the car. Simpler approximation: take the first point
        # at least Ld meters away from the car.
        target_loc = None
        for p in path_locs:
            if veh_loc.distance(p) >= Ld:
                target_loc = p
                break
        if target_loc is None:
            target_loc = path_locs[-1]  # End of path

        fwd = vehicle_transform.get_forward_vector()
        dx = target_loc.x - veh_loc.x
        dy = target_loc.y - veh_loc.y

        # alpha = angle from car forward direction to target
        target_angle = math.atan2(dy, dx)
        car_yaw = math.atan2(fwd.y, fwd.x)
        alpha = target_angle - car_yaw
        # Normalize to [-pi, pi]
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi

        # Pure pursuit formula
        actual_Ld = max(0.5, math.hypot(dx, dy))  # avoid div-by-zero
        delta = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha), actual_Ld)
        # Map physical steering angle to [-1, 1]
        steer = max(-1.0, min(1.0, delta / self.MAX_STEER_RAD))
        return steer

    # -------------------------------------------------------------------------
    # Throttle / brake
    # -------------------------------------------------------------------------
    def _throttle_brake(self, current_speed_kmh, target_speed_kmh, dt_s):
        """PID on speed error. Positive output -> throttle, negative -> brake."""
        # Special case: target_speed = 0 means "stop now" (red light).
        if target_speed_kmh <= 0.1:
            self.speed_pid.reset()
            return 0.0, self.MAX_BRAKE

        error = target_speed_kmh - current_speed_kmh
        u = self.speed_pid.step(error, dt_s)

        if u >= 0:
            throttle = max(0.0, min(self.MAX_THROTTLE, u * self.THROTTLE_GAIN))
            brake = 0.0
        else:
            throttle = 0.0
            brake = max(0.0, min(self.MAX_BRAKE, -u * self.BRAKE_GAIN))
        return throttle, brake

    # -------------------------------------------------------------------------
    # Combined control update
    # -------------------------------------------------------------------------
    def update_control(self, destination, additional_vars, delta_time_ms):
        """
        Build VehicleControl and apply. delta_time_ms is in milliseconds
        as supplied by Autopilot.
        """
        vehicle_transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        dt_s = max(0.001, delta_time_ms / 1000.0)

        # Target speed from Knowledge (Analyser may zero it for red lights)
        target_speed = self.knowledge.retrieve_data(
            'target_speed', self.DEFAULT_TARGET_SPEED)
        if target_speed is None:
            target_speed = self.DEFAULT_TARGET_SPEED

        # ---- Steering -------------------------------------------------------
        if self.STEER_MODE == 'pure_pursuit':
            steer = self._steer_pure_pursuit(vehicle_transform, current_speed)
        else:
            target_loc = self._coerce_location(destination)
            steer = self._steer_aim_at(vehicle_transform, target_loc)

        # ---- Throttle / brake -----------------------------------------------
        throttle, brake = self._throttle_brake(current_speed, target_speed, dt_s)

        # ---- Apply ----------------------------------------------------------
        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.steer = float(steer)
        control.brake = float(brake)
        control.hand_brake = False
        self.vehicle.apply_control(control)

    @staticmethod
    def _coerce_location(maybe_vec_or_loc):
        """Accept Vector3D or Location, return a Location."""
        if isinstance(maybe_vec_or_loc, carla.Location):
            return maybe_vec_or_loc
        return carla.Location(
            x=maybe_vec_or_loc.x,
            y=maybe_vec_or_loc.y,
            z=maybe_vec_or_loc.z)


# =============================================================================
# Planner - high-level path planning
# =============================================================================
class Planner(object):
    """
    Generates a list of waypoints from the vehicle to the destination.

    Two modes via PLANNER_MODE:
      - 'greedy' : pick the next-waypoint closest to the goal at each step.
                   Fast but fails at roundabouts and split intersections.
      - 'astar'  : A* shortest-path on the CARLA waypoint graph.
                   Correct on complex topologies. Recommended.

    Also handles the HEALING state (M2) by building an escape path.
    """

    # ---- TUNABLE: PLANNER MODE ----------------------------------------------
    PLANNER_MODE = 'greedy'   # 'greedy' or 'astar'

    # ---- TUNABLE: PATH GENERATION -------------------------------------------
    STEP_DISTANCE = 2.0        # meters between waypoints
    GOAL_TOLERANCE = 3.0       # meters - "close enough" to goal in search
    GREEDY_MAX_STEPS = 100    # safety cap on greedy iterations
    ASTAR_MAX_ITERATIONS = 8000  # safety cap on A* node expansions

    # ---- TUNABLE: VISUALIZATION ---------------------------------------------
    DRAW_PATH = True
    DRAW_LIFETIME = 60.0       # seconds the markers stay visible

    # ---- TUNABLE: ESCAPE PATH (M2) ------------------------------------------
    ESCAPE_LATERAL_OFFSET = 3.5   # meters to swerve sideways
    ESCAPE_FORWARD_DISTANCE = 12.0  # meters of forward travel during escape

    def __init__(self, knowledge, vehicle=None):
        self.knowledge = knowledge
        self.vehicle = vehicle
        self.path = deque([])

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def make_plan(self, source, destination):
        """Build the full plan from source to destination and start driving."""
        self.path = self.build_path(source, destination)
        self._publish_path_locations()
        self.update_plan()
        self.knowledge.update_destination(self.get_current_destination())
        if self.DRAW_PATH:
            self._draw_path(color=carla.Color(255, 0, 0))

    def update(self, time_elapsed):
        """Tick - advance the plan if we've arrived at the next waypoint."""
        # If we're in HEALING, the path was replaced by an escape path.
        # Once that drains, transition back to DRIVING (caller should re-plan
        # if it wants to reach the original goal).
        self.update_plan()
        self.knowledge.update_destination(self.get_current_destination())

    def update_plan(self):
        """Advance through the path, popping waypoints as we reach them."""
        # Don't overwrite HEALING - Analyser owns that transition. If the
        # escape path drains, just leave status alone; Analyser's HEALING
        # timer will re-plan back to the original goal.
        if self.knowledge.get_status() == Status.HEALING:
            if len(self.path) > 0 and self.knowledge.arrived_at(self.path[0]):
                self.path.popleft()
                self._publish_path_locations()
            return

        if len(self.path) == 0:
            return
        if self.knowledge.arrived_at(self.path[0]):
            self.path.popleft()
            self._publish_path_locations()
        if len(self.path) == 0:
            self.knowledge.update_status(Status.ARRIVED)
        else:
            self.knowledge.update_status(Status.DRIVING)

    def get_current_destination(self):
        """Return the current target point given the AI's status."""
        status = self.knowledge.get_status()
        if status == Status.DRIVING and len(self.path) > 0:
            return self.path[0]
        if status == Status.HEALING and len(self.path) > 0:
            return self.path[0]   # escape path waypoints
        if status == Status.ARRIVED:
            return self.knowledge.get_location()
        if status == Status.CRASHED:
            return self.knowledge.get_location()
        return self.knowledge.get_location()

    def build_escape_path(self, threat_direction='right'):
        """
        Used when Analyser flags an incoming threat (M2). Generates a short
        escape path that swerves AWAY from the threat.

        Replaces self.path with the escape sequence. After the escape ends
        you should call make_plan() again to resume the original route.
        """
        if self.vehicle is None:
            return

        carla_map = self.vehicle.get_world().get_map()
        veh_tf = self.vehicle.get_transform()

        # Direction to swerve: opposite of threat
        swerve = -1 if threat_direction == 'right' else 1   # +1 = left

        # Forward unit
        fwd = veh_tf.get_forward_vector()
        # Right unit. Prefer CARLA's built-in (works on most 0.9.x). Falls back
        # to derivation from yaw. CARLA convention: at yaw=0 (facing +x), the
        # car's right is +y, so right_unit = (-sin(yaw), cos(yaw)).
        try:
            right_vec = veh_tf.rotation.get_right_vector()
            right_x, right_y = right_vec.x, right_vec.y
        except Exception:
            yaw_rad = math.radians(veh_tf.rotation.yaw)
            right_x = -math.sin(yaw_rad)
            right_y = math.cos(yaw_rad)

        veh_loc = veh_tf.location
        offset = self.ESCAPE_LATERAL_OFFSET * swerve
        forward_d = self.ESCAPE_FORWARD_DISTANCE

        # Build a few escape waypoints: a swerve-out, glide, and re-center.
        escape_points = []
        # Phase 1: short forward + swerve
        p1 = carla.Location(
            x=veh_loc.x + fwd.x * forward_d * 0.4 + right_x * offset,
            y=veh_loc.y + fwd.y * forward_d * 0.4 + right_y * offset,
            z=veh_loc.z)
        escape_points.append(p1)
        # Phase 2: forward, hold lateral
        p2 = carla.Location(
            x=veh_loc.x + fwd.x * forward_d * 0.7 + right_x * offset,
            y=veh_loc.y + fwd.y * forward_d * 0.7 + right_y * offset,
            z=veh_loc.z)
        escape_points.append(p2)
        # Phase 3: re-center to road
        p3_world = carla.Location(
            x=veh_loc.x + fwd.x * forward_d,
            y=veh_loc.y + fwd.y * forward_d,
            z=veh_loc.z)
        p3_wp = carla_map.get_waypoint(p3_world)
        escape_points.append(p3_wp.transform.location)

        self.path = deque(escape_points)
        self._publish_path_locations()
        if self.DRAW_PATH:
            self._draw_path(color=carla.Color(255, 200, 0), label_prefix='E')

    # -------------------------------------------------------------------------
    # Path building
    # -------------------------------------------------------------------------
    def build_path(self, source, destination):
        """Dispatch to greedy or A*."""
        if self.vehicle is None:
            # No map access -> fallback straight-line
            self.path = deque([])
            self.path.append(self._coerce_location(destination))
            return self.path

        if self.PLANNER_MODE == 'greedy':
            return self._build_path_greedy(source, destination)
        return self._build_path_astar(source, destination)

    # ---- Greedy -------------------------------------------------------------
    def _build_path_greedy(self, source, destination):
        path = deque([])
        world = self.vehicle.get_world()
        carla_map = world.get_map()
        source_loc = self._extract_location(source)
        dest_loc = self._coerce_location(destination)

        FORK_PEEK = 15.0   # used only when a fork is detected at the small step

        fork_colors = [
            carla.Color(0, 255, 255), carla.Color(255, 0, 255),
            carla.Color(255, 255, 0), carla.Color(255, 128, 0),
        ]

        current_wp = carla_map.get_waypoint(source_loc)
        for i in range(self.GREEDY_MAX_STEPS):
            wp_loc = current_wp.transform.location
            if wp_loc.distance(dest_loc) < self.GOAL_TOLERANCE:
                break

            # Small-step peek: detects fork presence cheaply
            small_peek = current_wp.next(self.STEP_DISTANCE)
            if not small_peek:
                break

            if len(small_peek) > 1 and wp_loc.distance(dest_loc) > FORK_PEEK:
                big_peek = current_wp.next(FORK_PEEK)
                if not big_peek or len(big_peek) < 2:
                    big_peek = small_peek

                target = min(big_peek, key=lambda w: w.transform.location.distance(dest_loc))

                # Visualize and debug
                print("FORK: small={} big={} options".format(len(small_peek), len(big_peek)))
                origin = carla.Location(x=wp_loc.x, y=wp_loc.y, z=wp_loc.z + 2.0)
                world.debug.draw_point(origin, size=0.4,
                    color=carla.Color(255, 255, 0), life_time=120.0)
                for idx, w in enumerate(big_peek):
                    d = w.transform.location.distance(dest_loc)
                    chosen_marker = " <-- CHOSEN" if w is target else ""
                    print("  opt{}: {} dist={:.1f}{}".format(
                        idx, w.transform.location, d, chosen_marker))
                    color = fork_colors[idx % len(fork_colors)]
                    opt_loc = carla.Location(
                        x=w.transform.location.x, y=w.transform.location.y,
                        z=w.transform.location.z + 2.0)
                    world.debug.draw_point(opt_loc, size=0.35, color=color, life_time=120.0)
                    world.debug.draw_line(origin, opt_loc, thickness=0.15,
                        color=color, life_time=120.0)

                # Jump directly to the chosen big-step waypoint
                current_wp = target
                path.append(current_wp.transform.location)
            else:
                current_wp = small_peek[0]
                path.append(current_wp.transform.location)

        # Trim waypoints past the goal
        trimmed = deque()
        for wp_loc in path:
            trimmed.append(wp_loc)
            if wp_loc.distance(dest_loc) < self.GOAL_TOLERANCE:
                break
        path = trimmed

        final_wp = carla_map.get_waypoint(dest_loc)
        if not path or path[-1].distance(final_wp.transform.location) > self.GOAL_TOLERANCE:
            path.append(final_wp.transform.location)
        print("[Planner GREEDY] path length:", len(path))
        return path

    # ---- A* -----------------------------------------------------------------
    def _build_path_astar(self, source, destination):
        """
        A* over the CARLA road topology graph.

        WHY NOT JUST wp.next() ?
        ------------------------
        In CARLA, `waypoint.next()` inside junctions (roundabouts especially)
        often only returns the continuation lane — exit branches are missed
        because they live in different road segments connected via the
        junction object, not via `next()`. That makes raw-`next()` A* think
        the only way out of a roundabout is to keep going around.

        SOLUTION
        --------
        Use carla's GlobalRoutePlanner, which builds a proper topology graph
        from `Map.get_topology()` (returns (entry_wp, exit_wp) pairs across
        all roads INCLUDING junction connections) and runs A* on that.

        This is exactly the same algorithm we wrote, just on a graph that
        actually models the road network correctly.
        """
        path = deque([])
        world = self.vehicle.get_world()
        carla_map = world.get_map()
        source_loc = self._extract_location(source)
        dest_loc = self._coerce_location(destination)

        # Snap goal to nearest road waypoint
        goal_wp = carla_map.get_waypoint(dest_loc)
        start_wp = carla_map.get_waypoint(source_loc)
        # FIX: prefer same-direction lane. Opposite lane_id signs = opposite
        # travel directions in CARLA. If the user-given goal coordinate snaps
        # to the opposite-direction lane (very common for off-road or roughly-
        # specified destinations), GRP will route the LONG way around to enter
        # that lane the right way. Mirror to the matching-direction lane.
        if goal_wp.lane_id * start_wp.lane_id < 0:
            candidate = goal_wp.get_left_lane()
            if candidate is not None and \
                    candidate.lane_id * start_wp.lane_id > 0:
                print("[Planner A*] goal snapped to opposite lane, mirroring")
                goal_wp = candidate
        goal_loc = goal_wp.transform.location

        # Try to import GlobalRoutePlanner from CARLA's PythonAPI/agents folder
        grp = self._get_global_route_planner(carla_map)
        if grp is None:
            print("[Planner A*] GlobalRoutePlanner unavailable, falling back to greedy")
            return self._build_path_greedy(source, destination)

        # GRP returns list of (waypoint, RoadOption) tuples
        try:
            route = grp.trace_route(source_loc, goal_loc)
        except Exception as e:
            print("[Planner A*] trace_route failed:", e)
            return self._build_path_greedy(source, destination)

        if not route:
            print("[Planner A*] empty route, falling back to greedy")
            return self._build_path_greedy(source, destination)

        # GRP returns waypoints already spaced at ~STEP_DISTANCE (we pass
        # STEP_DISTANCE as sampling_resolution to GRP). Use them directly,
        # dropping consecutive duplicates that GRP emits at junction
        # transitions (these are what made the car loop in earlier runs).
        # NOTE: we do NOT append the original goal_loc at the end. GRP already
        # routes to the snapped goal_wp; appending the raw user destination
        # creates a "teleport-target" off-road waypoint, which made the car
        # cut through buildings on the last segment.
        prev_loc = None
        DEDUP_THRESHOLD = 0.5  # meters - drop waypoints closer than this to prev
        for wp, _ in route:
            loc = wp.transform.location
            if prev_loc is not None and loc.distance(prev_loc) < DEDUP_THRESHOLD:
                continue
            path.append(loc)
            prev_loc = loc
        print("[Planner A*] GRP route waypoints={}, path length={}".format(
            len(route), len(path)))
        return path

    def _get_global_route_planner(self, carla_map):
        """
        Locate and instantiate carla's GlobalRoutePlanner. Tries multiple
        possible locations; returns None if it can't find it.
        """
        # Try import paths in order
        candidates = []
        carla_root = os.environ.get('CARLA_ROOT', None)
        here = os.path.dirname(os.path.abspath(__file__))
        if carla_root:
            candidates.append(os.path.join(carla_root, 'PythonAPI', 'carla'))
        candidates.append(os.path.join(here, '..', 'carla'))                 # examples/../carla
        candidates.append(os.path.join(here, '..', '..', 'PythonAPI', 'carla'))
        for c in candidates:
            if os.path.isdir(c) and c not in sys.path:
                sys.path.append(c)
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
        except ImportError:
            try:
                # Older CARLA versions also need the DAO
                from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
                from agents.navigation.global_route_planner import GlobalRoutePlanner
                dao = GlobalRoutePlannerDAO(carla_map, self.STEP_DISTANCE)
                grp = GlobalRoutePlanner(dao)
                grp.setup()
                return grp
            except ImportError:
                return None
        # Newer CARLA (>= 0.9.12) takes (map, sampling_resolution) directly
        try:
            grp = GlobalRoutePlanner(carla_map, self.STEP_DISTANCE)
            return grp
        except TypeError:
            return None

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    def _draw_path(self, color=None, label_prefix=''):
        """Draw red dots + green lines for the current path in CARLA's debug renderer."""
        if self.vehicle is None or len(self.path) == 0:
            return
        world = self.vehicle.get_world()
        wps = list(self.path)
        if color is None:
            color = carla.Color(255, 0, 0)
        line_color = carla.Color(0, 255, 0)

        for i, p in enumerate(wps):
            loc = carla.Location(x=p.x, y=p.y, z=p.z + 1.0)
            world.debug.draw_point(
                loc, size=0.12,
                color=color,
                life_time=self.DRAW_LIFETIME)
            world.debug.draw_string(
                loc,
                "{}{}".format(label_prefix, i),
                draw_shadow=False,
                color=carla.Color(255, 255, 255),
                life_time=self.DRAW_LIFETIME)
        for a, b in zip(wps, wps[1:]):
            world.debug.draw_line(
                carla.Location(x=a.x, y=a.y, z=a.z + 1.0),
                carla.Location(x=b.x, y=b.y, z=b.z + 1.0),
                thickness=0.08,
                color=line_color,
                life_time=self.DRAW_LIFETIME)

    # -------------------------------------------------------------------------
    # Internal: publish current path to Knowledge so Executor can read it
    # -------------------------------------------------------------------------
    def _publish_path_locations(self):
        # Pure pursuit needs the rest of the path, not just the head.
        self.knowledge.update_data('path_locations', list(self.path))

    # -------------------------------------------------------------------------
    # Coercion helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_location(maybe):
        if isinstance(maybe, carla.Location):
            return maybe
        return carla.Location(x=maybe.x, y=maybe.y, z=maybe.z)

    @staticmethod
    def _extract_location(source):
        """source may be a Transform (from set_destination) or a vector."""
        if hasattr(source, 'location'):
            return source.location
        return Planner._coerce_location(source)