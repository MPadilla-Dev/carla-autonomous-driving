#!/usr/bin/env python
"""
ai_parser.py
=============
Monitor (reads sensors -> Knowledge) + Analyser (interprets Knowledge -> sets flags).

WHAT'S IN HERE
--------------
- Monitor reads:
    * vehicle location/rotation              (every tick)
    * lane invasion events                   (callback)
    * lidar point cloud                      (callback, for M2)
    * traffic light status                   (every tick, for M3)
    * speed limit                            (every tick, for M3 polish)

- Analyser computes:
    * target_speed                           (M3 - 0 at red lights, else cruise)
    * obstacle_threat / threat_direction     (M2 - from lidar)
    * Triggers HEALING status and asks Planner to build an escape path.

PRESENTATION TIPS
-----------------
- Toggle TRAFFIC_LIGHT_ENABLED to compare driving-with-vs-without traffic-light awareness.
- Toggle OBSTACLE_AVOIDANCE_ENABLED for the M2 head-to-head demo.
- Tunables for lidar threat detection are at the top of Analyser.
"""

import glob
import os
import sys
import math
import weakref

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
# Monitor - sensor reading
# =============================================================================
class Monitor(object):
    """
    Spawns sensors and ticks them into Knowledge. Keeps refs to actors so
    they can be cleaned up via destroy_sensors() at shutdown.
    """

    # ---- TUNABLE: LIDAR PARAMETERS ------------------------------------------
    LIDAR_RANGE = 20.0            # meters
    LIDAR_ROTATION_FREQ = 20.0    # Hz
    LIDAR_CHANNELS = 32
    LIDAR_POINTS_PER_SECOND = 56000
    LIDAR_HEIGHT = 2.5            # meters above vehicle origin

    def __init__(self, knowledge, vehicle):
        self.vehicle = vehicle
        self.knowledge = knowledge
        self.sensors = []
        weak_self = weakref.ref(self)

        self.knowledge.update_data('location', self.vehicle.get_transform().location)
        self.knowledge.update_data('rotation', self.vehicle.get_transform().rotation)

        world = self.vehicle.get_world()
        bp_lib = world.get_blueprint_library()

        # ---- Lane invasion sensor (already in original) --------------------
        ln_bp = bp_lib.find('sensor.other.lane_invasion')
        ln_sensor = world.spawn_actor(ln_bp, carla.Transform(), attach_to=self.vehicle)
        ln_sensor.listen(lambda event: Monitor._on_invasion(weak_self, event))
        self.sensors.append(ln_sensor)

        # ---- Lidar (new, for M2 obstacle detection) ------------------------
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(self.LIDAR_RANGE))
        lidar_bp.set_attribute('rotation_frequency', str(self.LIDAR_ROTATION_FREQ))
        lidar_bp.set_attribute('channels', str(self.LIDAR_CHANNELS))
        lidar_bp.set_attribute('points_per_second', str(self.LIDAR_POINTS_PER_SECOND))
        lidar_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=self.LIDAR_HEIGHT))
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_tf, attach_to=self.vehicle)
        lidar_sensor.listen(lambda data: Monitor._on_lidar(weak_self, data))
        self.sensors.append(lidar_sensor)

        # ---- Initialize default knowledge keys ------------------------------
        self.knowledge.update_data('lidar_points', None)
        self.knowledge.update_data('at_lights', False)
        self.knowledge.update_data('traffic_light_state', None)
        self.knowledge.update_data('speed_limit', 30.0)

    # -------------------------------------------------------------------------
    # Tick - run every frame
    # -------------------------------------------------------------------------
    def update(self, time_elapsed):
        tf = self.vehicle.get_transform()
        self.knowledge.update_data('location', tf.location)
        self.knowledge.update_data('rotation', tf.rotation)

        # Traffic light state (M3)
        try:
            at_tl = self.vehicle.is_at_traffic_light()
        except AttributeError:
            at_tl = False
        self.knowledge.update_data('at_lights', at_tl)

        if at_tl:
            tl = self.vehicle.get_traffic_light()
            if tl is not None:
                self.knowledge.update_data('traffic_light_state', tl.get_state())
            else:
                self.knowledge.update_data('traffic_light_state', None)
        else:
            self.knowledge.update_data('traffic_light_state', None)

        # Speed limit (km/h, what the road sign says)
        try:
            self.knowledge.update_data('speed_limit', self.vehicle.get_speed_limit())
        except AttributeError:
            pass

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def destroy_sensors(self):
        for s in self.sensors:
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.destroy()
            except Exception:
                pass
        self.sensors = []

    # -------------------------------------------------------------------------
    # Sensor callbacks (static so weakref doesn't pin self)
    # -------------------------------------------------------------------------
    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.knowledge.update_data('lane_invasion', event.crossed_lane_markings)

    @staticmethod
    def _on_lidar(weak_self, sensor_data):
        """
        Parse the raw bytes into an Nx4 array of (x, y, z, intensity)
        and stash on Knowledge for the Analyser to consume.

        Lidar coordinates are sensor-local: +x forward, +y right, +z up.
        """
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(sensor_data.raw_data, dtype=np.float32)
        if points.size == 0:
            self.knowledge.update_data('lidar_points', None)
            return
        points = np.reshape(points, (-1, 4))
        self.knowledge.update_data('lidar_points', points)


# =============================================================================
# Analyser - turn raw data into decisions
# =============================================================================
class Analyser(object):
    """
    Per-tick:
      1. Adjust target_speed from speed limits and red lights (M3).
      2. Scan lidar for incoming-collision threats (M2).
      3. If a threat is detected and we're DRIVING, switch to HEALING and
         hand the planner a request to build an escape path.
    """

    # ---- TUNABLE: FEATURE TOGGLES (great for demo A/B) ----------------------
    OBSTACLE_AVOIDANCE_ENABLED = True
    TRAFFIC_LIGHT_ENABLED = True
    USE_POSTED_SPEED_LIMIT = False   # if True, target_speed follows road signs

    # ---- TUNABLE: SPEED ------------------------------------------------------
    NORMAL_TARGET_SPEED = 50.0    # km/h - cruise speed when no other constraint
    HEALING_TARGET_SPEED = 18.0   # km/h - slow during evasive maneuver

    # ---- TUNABLE: LIDAR THREAT DETECTION (M2) -------------------------------
    # Define a danger zone in front of the car (in lidar-local coords).
    # Any cluster of points in this zone for >= THREAT_FRAMES_TO_TRIGGER frames
    # counts as a threat.
    THREAT_FORWARD_MIN = 0.5      # meters - ignore points right at the bumper
    THREAT_FORWARD_MAX = 9.0      # meters - how far ahead to look
    THREAT_LATERAL_HALF_WIDTH = 3.5  # meters - how wide the cone is
    THREAT_HEIGHT_MIN = -0.3      # meters - ignore ground returns
    THREAT_HEIGHT_MAX = 2.0       # meters - ignore overhead structures
    THREAT_MIN_POINTS = 8         # how many points qualify as a real object
    THREAT_FRAMES_TO_TRIGGER = 3  # consecutive ticks before flagging

    # ---- TUNABLE: HEALING DURATION ------------------------------------------
    HEALING_FRAMES = 60           # ticks to stay in HEALING before re-planning

    def __init__(self, knowledge):
        self.knowledge = knowledge
        # State for threat-detection debouncing
        self._threat_frames = 0
        self._healing_frames_left = 0
        # Set by Autopilot so we can request an escape path or re-plan
        self.planner = None
        self.original_destination = None
        self.last_threat_direction = 'right'

    def attach_planner(self, planner):
        """Autopilot calls this once at construction so we can talk to Planner."""
        self.planner = planner

    def remember_destination(self, destination):
        """Cache the high-level goal so we can re-plan after HEALING ends."""
        self.original_destination = destination

    # -------------------------------------------------------------------------
    # Tick
    # -------------------------------------------------------------------------
    def update(self, time_elapsed):
        # 1) Speed control (M3)
        self._update_target_speed()

        # 2) Healing maintenance
        if self.knowledge.get_status() == Status.HEALING:
            self._healing_frames_left -= 1
            if self._healing_frames_left <= 0:
                # Re-plan back to original goal
                if self.planner is not None and self.original_destination is not None:
                    self.knowledge.update_status(Status.DRIVING)
                    veh_tf = carla.Transform(self.knowledge.get_location())
                    self.planner.make_plan(veh_tf, self.original_destination)
                else:
                    self.knowledge.update_status(Status.DRIVING)
            return  # don't re-trigger HEALING during HEALING

        # 3) Threat detection (M2)
        if self.OBSTACLE_AVOIDANCE_ENABLED:
            self._update_threat_detection()

    # -------------------------------------------------------------------------
    # Speed control
    # -------------------------------------------------------------------------
    def _update_target_speed(self):
        if self.knowledge.get_status() == Status.HEALING:
            self.knowledge.update_data('target_speed', self.HEALING_TARGET_SPEED)
            return

        if self.TRAFFIC_LIGHT_ENABLED:
            at_tl = self.knowledge.retrieve_data('at_lights', False)
            tl_state = self.knowledge.retrieve_data('traffic_light_state', None)
            if at_tl and tl_state == carla.TrafficLightState.Red:
                self.knowledge.update_data('target_speed', 0.0)
                return

        # Otherwise: posted speed limit (capped at NORMAL_TARGET_SPEED) or just NORMAL.
        if self.USE_POSTED_SPEED_LIMIT:
            posted = self.knowledge.retrieve_data('speed_limit', self.NORMAL_TARGET_SPEED)
            if posted is None or posted <= 0:
                posted = self.NORMAL_TARGET_SPEED
            target = min(self.NORMAL_TARGET_SPEED, posted)
        else:
            target = self.NORMAL_TARGET_SPEED

        self.knowledge.update_data('target_speed', target)

    # -------------------------------------------------------------------------
    # Threat detection
    # -------------------------------------------------------------------------
    def _update_threat_detection(self):
        points = self.knowledge.retrieve_data('lidar_points', None)
        if points is None or len(points) == 0:
            self._threat_frames = 0
            self.knowledge.update_data('obstacle_threat', False)
            return

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        in_zone = (
            (x >= self.THREAT_FORWARD_MIN) &
            (x <= self.THREAT_FORWARD_MAX) &
            (np.abs(y) <= self.THREAT_LATERAL_HALF_WIDTH) &
            (z >= self.THREAT_HEIGHT_MIN) &
            (z <= self.THREAT_HEIGHT_MAX)
        )
        n_in_zone = int(np.count_nonzero(in_zone))

        if n_in_zone >= self.THREAT_MIN_POINTS:
            self._threat_frames += 1
            # Determine threat direction (+y = right, -y = left in lidar local)
            mean_y = float(np.mean(y[in_zone]))
            self.last_threat_direction = 'right' if mean_y > 0 else 'left'
        else:
            self._threat_frames = 0

        threat_now = self._threat_frames >= self.THREAT_FRAMES_TO_TRIGGER
        self.knowledge.update_data('obstacle_threat', threat_now)
        self.knowledge.update_data('threat_direction', self.last_threat_direction)

        if threat_now and self.knowledge.get_status() == Status.DRIVING:
            self._enter_healing()

    def _enter_healing(self):
        """Switch to HEALING and ask Planner for an escape path."""
        self.knowledge.update_status(Status.HEALING)
        self._healing_frames_left = self.HEALING_FRAMES
        # Reset threat counter so we need fresh evidence to re-trigger after healing
        self._threat_frames = 0
        if self.planner is not None:
            self.planner.build_escape_path(self.last_threat_direction)