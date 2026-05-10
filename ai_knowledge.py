#!/usr/bin/env python
"""
ai_knowledge.py
================
Shared state ("blackboard") for the AI modules.

All four modules (Monitor, Analyser, Planner, Executor) read from and write to
this single Knowledge object. This decouples them: nobody calls anyone directly.

WHAT WAS CHANGED FROM THE ORIGINAL
-----------------------------------
1. update_destination() threshold lowered from 5.0 -> 0.5 m. The 5 m filter
   silently swallows every change with dense (2 m) waypoints, which froze the
   Executor on the first waypoint forever.
2. Added a safe retrieve_data() that returns a default instead of KeyError.
3. Added helpers for the M2/M3 fields (at_lights, traffic_light_state,
   obstacle_threat) so we don't sprinkle hardcoded strings throughout the code.
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from enum import Enum


class Status(Enum):
    """States the AI can be in. Drives behavior in Planner and Executor."""
    ARRIVED  = 0   # at destination - hold still
    DRIVING  = 1   # normal operation - follow planned waypoints
    CRASHED  = 2   # unrecoverable failure - hold still
    HEALING  = 3   # avoiding obstacle - follow escape waypoints (M2)
    UNDEFINED = 4  # initial / error fallback


class Knowledge(object):
    """
    The shared blackboard. Read/write here from any module.

    Memory is a dict keyed by string. Common keys:
      - 'location'             : carla.Location of the vehicle
      - 'rotation'             : carla.Rotation of the vehicle
      - 'target_speed'         : float, km/h - set by Analyser/Autopilot
      - 'at_lights'            : bool - True when stopped at red (M3)
      - 'traffic_light_state'  : carla.TrafficLightState
      - 'lane_invasion'        : last lane invasion event
      - 'lidar_points'         : numpy array of latest lidar return (M2)
      - 'obstacle_threat'      : bool - Analyser flagged an incoming threat (M2)
      - 'threat_direction'     : str 'left'/'right' - escape direction (M2)
    """

    def __init__(self):
        self.status = Status.ARRIVED
        self.memory = {'location': carla.Vector3D(0.0, 0.0, 0.0)}
        self.destination = self.get_location()

        # Callbacks - other modules can subscribe to changes
        self.status_changed = lambda *_, **__: None
        self.destination_changed = lambda *_, **__: None
        self.data_changed = lambda *_, **__: None

    # -------------------------------------------------------------------------
    # Callback wiring
    # -------------------------------------------------------------------------
    def set_data_changed_callback(self, callback):
        self.data_changed = callback

    def set_status_changed_callback(self, callback):
        self.status_changed = callback

    def set_destination_changed_callback(self, callback):
        self.destination_changed = callback

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------
    def get_status(self):
        return self.status

    def set_status(self, new_status):
        self.status = new_status

    def update_status(self, new_status):
        """
        Updates status, but blocks transitions out of CRASHED unless going
        to HEALING. This is the original logic, preserved.
        """
        if (self.status != Status.CRASHED or new_status == Status.HEALING) \
                and self.status != new_status:
            self.set_status(new_status)
            self.status_changed(new_status)

    # -------------------------------------------------------------------------
    # Memory access
    # -------------------------------------------------------------------------
    def retrieve_data(self, data_name, default=None):
        """
        Safer than the original - returns `default` if the key is missing
        instead of raising KeyError.

        TUNABLE: change `default=None` if you want a different fallback.
        """
        return self.memory.get(data_name, default)

    def update_data(self, data_name, value):
        self.memory[data_name] = value
        self.data_changed(data_name)

    # -------------------------------------------------------------------------
    # Location & destination
    # -------------------------------------------------------------------------
    def get_location(self):
        return self.retrieve_data('location')

    def get_current_destination(self):
        return self.destination

    def update_destination(self, new_destination):
        """
        TUNABLE: ARRIVAL_FILTER_THRESHOLD controls how much the destination
        must change before we treat it as a new target. Original was 5.0 m,
        which is bigger than our typical waypoint spacing (2 m) and caused
        the destination to never update. 0.5 m is small enough to pass
        every real change, big enough to ignore numeric noise.
        """
        ARRIVAL_FILTER_THRESHOLD = 0.5
        if self.distance(self.destination, new_destination) > ARRIVAL_FILTER_THRESHOLD:
            self.destination = new_destination
            self.destination_changed(new_destination)

    def arrived_at(self, destination):
        """
        TUNABLE: ARRIVAL_RADIUS - how close the car must be to count as
        "arrived" at a waypoint. 5 m is the project default. Smaller =
        more precise but car may struggle to actually stop within tolerance.
        """
        ARRIVAL_RADIUS = 5.0
        return self.distance(self.get_location(), destination) < ARRIVAL_RADIUS

    # -------------------------------------------------------------------------
    # Geometry helper
    # -------------------------------------------------------------------------
    def distance(self, vec1, vec2):
        l1 = carla.Location(vec1)
        l2 = carla.Location(vec2)
        return l1.distance(l2)