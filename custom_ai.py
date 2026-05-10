#!/usr/bin/env python
"""
custom_ai.py
=============
The Autopilot - wires together Knowledge, Monitor, Analyser, Planner, Executor.

Autopilot.update() is called every tick from the main script. It runs the
modules in order: sense -> analyse -> plan -> act.

PRESENTATION TIPS
-----------------
- The initial target_speed gets set here. Analyser overrides it per-tick,
  so this just seeds it before the first tick.
- Pass `vehicle` to Planner so it can access world / map for waypoint-based
  path generation and visualization.
"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import ai_knowledge as data
import ai_control as control
import ai_parser as parser


class Autopilot(object):

    # ---- TUNABLE: INITIAL TARGET SPEED --------------------------------------
    # Analyser usually overwrites this immediately, but if Analyser is
    # disabled this will be the cruise speed.
    INITIAL_TARGET_SPEED = 35.0   # km/h

    def __init__(self, vehicle):
        self.vehicle = vehicle

        # Shared state
        self.knowledge = data.Knowledge()
        self.knowledge.update_data('target_speed', self.INITIAL_TARGET_SPEED)
        self.knowledge.set_status_changed_callback(self._on_status_changed)

        # Modules
        self.analyser = parser.Analyser(self.knowledge)
        self.monitor = parser.Monitor(self.knowledge, self.vehicle)
        self.planner = control.Planner(self.knowledge, self.vehicle)
        self.executor = control.Executor(self.knowledge, self.vehicle)

        # Wire Analyser <-> Planner so Analyser can request escape paths
        self.analyser.attach_planner(self.planner)

        # Time bookkeeping
        self.prev_time = int(round(time.time() * 1000))

        # User-supplied callbacks
        self.route_finished = lambda *_, **__: None
        self.crashed = lambda *_, **__: None

    # -------------------------------------------------------------------------
    # Status callback dispatcher
    # -------------------------------------------------------------------------
    def _on_status_changed(self, new_status):
        if new_status == data.Status.ARRIVED:
            self.route_finished(self)
        if new_status == data.Status.CRASHED:
            self.crashed(self)

    def set_route_finished_callback(self, callback):
        self.route_finished = callback

    def set_crash_callback(self, callback):
        self.crashed = callback

    def get_vehicle(self):
        return self.vehicle

    # -------------------------------------------------------------------------
    # Per-tick update
    # -------------------------------------------------------------------------
    def update(self):
        ctime = int(round(time.time() * 1000))
        delta_time = ctime - self.prev_time
        self.prev_time = ctime

        self.monitor.update(delta_time)
        self.analyser.update(delta_time)
        self.planner.update(delta_time)
        self.executor.update(delta_time)

        return self.knowledge.get_status()

    # -------------------------------------------------------------------------
    # Public: set destination
    # -------------------------------------------------------------------------
    def set_destination(self, destination):
        """
        Plans a route to `destination`. The Analyser remembers it so that
        if a HEALING detour happens, we can re-plan back to the same goal.
        """
        self.analyser.remember_destination(destination)
        self.planner.make_plan(self.vehicle.get_transform(), destination)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def destroy(self):
        """Clean up sensors. Call this from the test script's finally block."""
        try:
            self.monitor.destroy_sensors()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Convenience for the telemetry HUD
    # -------------------------------------------------------------------------
    def get_telemetry(self):
        """Returns a dict of values for the dashboard to render."""
        loc = self.knowledge.get_location()
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        return {
            'status': self.knowledge.get_status().name,
            'speed_kmh': speed_kmh,
            'target_speed': self.knowledge.retrieve_data('target_speed', 0.0),
            'at_lights': self.knowledge.retrieve_data('at_lights', False),
            'tl_state': str(self.knowledge.retrieve_data('traffic_light_state', None)),
            'obstacle_threat': self.knowledge.retrieve_data('obstacle_threat', False),
            'threat_direction': self.knowledge.retrieve_data('threat_direction', '-'),
            'speed_limit': self.knowledge.retrieve_data('speed_limit', 0.0),
            'location': (loc.x if loc else 0.0,
                         loc.y if loc else 0.0,
                         loc.z if loc else 0.0),
        }