#!/usr/bin/env python
"""
ai_modtest_hud.py
==================
Same as ai_modtest.py but with a pygame window showing:
  - Live chase camera view from behind the hero vehicle
  - Telemetry overlay: speed, target speed, throttle, steer, brake
  - Status text (DRIVING / HEALING / ARRIVED / CRASHED)
  - Event log (last few messages: arrivals, healing triggers, collisions)

Designed for screen-recording: open OBS/Game Bar/etc, point at the pygame
window, run this script. The pygame window IS your demo video.

Usage:
  python ai_modtest_hud.py -m 1
  python ai_modtest_hud.py -m 2
  python ai_modtest_hud.py -m 3
"""

import glob
from http import client
import os
import sys
from unittest import result
from xmlrpc import client

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pygame
import numpy as np
import weakref

import random
import time
import argparse

import custom_ai as ai


# ---- TUNABLE: WINDOW / CAMERA -----------------------------------------------
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Chase camera offset relative to the vehicle (x=forward, y=right, z=up).
# Negative x = behind the car. Positive z = above.
CAMERA_OFFSET_X = -8.0   # 6m behind
CAMERA_OFFSET_Z = 7.0    # 3m above
CAMERA_PITCH = -25.0     # tilt down 15 deg

# ---- TUNABLE: HUD COLORS ----------------------------------------------------
COLOR_TEXT = (255, 255, 255)
COLOR_DIM = (180, 180, 190)
COLOR_OK = (90, 220, 110)
COLOR_WARN = (250, 200, 70)
COLOR_BAD = (240, 90, 90)
COLOR_PANEL_BG = (0, 0, 0, 160)   # translucent black panel

# ---- TUNABLE: EVENT LOG SIZE ------------------------------------------------
MAX_LOG_LINES = 6


# =============================================================================
# Camera manager - attaches RGB camera, copies frames to pygame surface
# =============================================================================
class CameraView(object):
    def __init__(self, vehicle, width, height):
        self.surface = None
        world = vehicle.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        bp.set_attribute('fov', '90')
        cam_tf = carla.Transform(
            carla.Location(x=CAMERA_OFFSET_X, z=CAMERA_OFFSET_Z),
            carla.Rotation(pitch=CAMERA_PITCH))
        self.sensor = world.spawn_actor(bp, cam_tf, attach_to=vehicle)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda img: CameraView._on_frame(weak_self, img))

    @staticmethod
    def _on_frame(weak_self, image):
        self = weak_self()
        if not self:
            return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        arr = arr[:, :, :3]               # drop alpha
        arr = arr[:, :, ::-1]              # BGRA -> RGB
        # pygame wants (W, H, 3) — surfarray uses (W, H), so we transpose
        self.surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def destroy(self):
        try:
            self.sensor.stop()
            self.sensor.destroy()
        except Exception:
            pass


# =============================================================================
# HUD overlay - text panels on top of the camera view
# =============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_big = pygame.font.SysFont("consolas", 26, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.events = []
        self._last_status = None

    def log_event(self, text):
        stamp = time.strftime("%H:%M:%S")
        self.events.append("[{}] {}".format(stamp, text))
        if len(self.events) > MAX_LOG_LINES:
            self.events = self.events[-MAX_LOG_LINES:]

    def render(self, display, vehicle, autopilot):
        # --- gather data ----------------------------------------------------
        ctrl = vehicle.get_control()
        velocity = vehicle.get_velocity()
        speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        tele = autopilot.get_telemetry()
        status = tele.get('status', 'UNKNOWN')
        if status != self._last_status:
            self.log_event("Status: " + status)
            self._last_status = status

        # --- top-left panel: status + speed ---------------------------------
        self._panel(display, 10, 10, 280, 130)
        status_color = {
            'DRIVING': COLOR_OK, 'ARRIVED': COLOR_OK,
            'HEALING': COLOR_WARN, 'CRASHED': COLOR_BAD,
        }.get(status, COLOR_DIM)
        self._blit(display, self.font_small, "STATUS", (20, 18), COLOR_DIM)
        self._blit(display, self.font_big, status, (20, 32), status_color)
        self._blit(display, self.font_small, "SPEED (km/h)", (20, 70), COLOR_DIM)
        self._blit(display, self.font_big,
                   "{:5.1f} / {:4.1f}".format(speed_kmh, tele.get('target_speed', 0.0)),
                   (20, 86), COLOR_TEXT)

        # --- top-right panel: controls --------------------------------------
        self._panel(display, self.width - 290, 10, 280, 130)
        rx = self.width - 280
        self._blit(display, self.font_small, "CONTROLS", (rx, 18), COLOR_DIM)
        self._blit(display, self.font,
                   "throttle: {:.2f}".format(ctrl.throttle), (rx, 40), COLOR_TEXT)
        self._blit(display, self.font,
                   "brake:    {:.2f}".format(ctrl.brake), (rx, 62), COLOR_TEXT)
        self._blit(display, self.font,
                   "steer:    {:+.2f}".format(ctrl.steer), (rx, 84), COLOR_TEXT)
        # Steer bar
        bar_x = rx
        bar_y = 110
        bar_w = 250
        pygame.draw.rect(display, (50, 50, 60), (bar_x, bar_y, bar_w, 8))
        mid = bar_x + bar_w // 2
        pygame.draw.line(display, (200, 200, 200), (mid, bar_y - 2), (mid, bar_y + 10), 1)
        sx = int(mid + (ctrl.steer * bar_w / 2))
        pygame.draw.circle(display, (90, 160, 230), (sx, bar_y + 4), 6)

        # --- bottom-left: event log ------------------------------------------
        log_h = MAX_LOG_LINES * 18 + 20
        self._panel(display, 10, self.height - log_h - 10, 460, log_h)
        self._blit(display, self.font_small, "EVENTS",
                   (20, self.height - log_h), COLOR_DIM)
        for i, msg in enumerate(self.events):
            self._blit(display, self.font_small, msg,
                       (20, self.height - log_h + 18 + i * 18), COLOR_TEXT)

        # --- bottom-right: extras --------------------------------------------
        self._panel(display, self.width - 290, self.height - 90, 280, 80)
        rx = self.width - 280
        ry = self.height - 82
        tl = str(tele.get('tl_state', 'None'))
        tl_color = (COLOR_BAD if 'Red' in tl
                    else COLOR_WARN if 'Yellow' in tl
                    else COLOR_OK if 'Green' in tl else COLOR_DIM)
        self._blit(display, self.font_small, "Traffic light:", (rx, ry), COLOR_DIM)
        self._blit(display, self.font, tl, (rx + 110, ry - 2), tl_color)

        threat = tele.get('obstacle_threat', False)
        direction = tele.get('threat_direction', '-')
        if threat:
            self._blit(display, self.font, "THREAT from " + direction,
                       (rx, ry + 22), COLOR_BAD)
        else:
            self._blit(display, self.font, "Scan: clear",
                       (rx, ry + 22), COLOR_OK)
        self._blit(display, self.font_small,
                   "limit: {:.0f} km/h".format(tele.get('speed_limit', 0)),
                   (rx, ry + 50), COLOR_DIM)

    # ---- helpers --------------------------------------------------------
    def _panel(self, display, x, y, w, h):
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(COLOR_PANEL_BG)
        display.blit(surf, (x, y))

    def _blit(self, display, font, text, pos, color):
        s = font.render(text, True, color)
        display.blit(s, pos)


# =============================================================================
# Helpers from original test
# =============================================================================
def get_dist(point1, point2):
    return point1.location.distance(point2)


def get_start_point(world, coord):
    points = world.get_map().get_spawn_points()
    index = 0
    ti = -1
    td = get_dist(points[0], coord)
    for point in points:
        ti += 1
        d = get_dist(point, coord)
        if d < td:
            td = d
            index = ti
    return world.get_map().get_waypoint(points[index].location)


# =============================================================================
# Main
# =============================================================================
def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-m', '--milestone-number', metavar='M',
                           default=1, type=int)
    argparser.add_argument('--no-record', action='store_true',
                           help='Skip CARLA recorder')
    argparser.add_argument('--label', default=None,
                           help='Recording filename label')
    args = argparser.parse_args()

    actor_list = []
    autopilot = None
    camera = None
    display = None
    client = None

    pygame.init()
    display = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA AI Demo")
    clock = pygame.time.Clock()
    hud = HUD(WINDOW_WIDTH, WINDOW_HEIGHT)

    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        world = client.load_world('Town03')
        # ---- Synchronous mode -----------------------------------------------
        # In async mode, lidar runs at its own rate and our main loop ticks
        # separately, causing intermittent threat detection (lidar misses
        # frames or repeats them). Sync mode ties them together: one
        # world.tick() = one simulation step + one lidar frame.
        original_settings = world.get_settings()
        sync_settings = world.get_settings()
        sync_settings.synchronous_mode = True
        sync_settings.fixed_delta_seconds = 0.05   # 20 Hz
        world.apply_settings(sync_settings)

        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints
                      if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]

        # ---- Recorder ------------------------------------------------------
        if not args.no_record:
            rec_dir = os.path.join(
                "C:\\Projects\\CARLA\\CARLA_0.9.13\\WindowsNoEditor",
                "PythonAPI", "examples", "presentation")
            os.makedirs(rec_dir, exist_ok=True)
            label = args.label or "m{}_hud_{}.log".format(
                args.milestone_number, time.strftime("%H%M%S"))
            rec_path = os.path.join(rec_dir, label)
            print("Recording to:", rec_path)
            client.start_recorder(rec_path, True)

        # ---- Spawn ---------------------------------------------------------
        def try_spawn(transform, vid=""):
            if vid:
                bp = next((b for b in blueprints if b.id == vid), None)
                if bp is None:
                    return None
                if bp.has_attribute('color'):
                    bp.set_attribute('color',
                        random.choice(bp.get_attribute('color').recommended_values))
                bp.set_attribute('role_name', 'autopilot')
                v = world.try_spawn_actor(bp, transform)
                if v is not None:
                    actor_list.append(v)
                return v
            return None

        ex1 = [carla.Vector3D(42.5959, -4.3443, 1.8431),
               carla.Vector3D(22, -4, 1.8431),
               carla.Vector3D(9, -22, 1.8431)]
        ex2 = [carla.Vector3D(42.5959, -4.3443, 1.8431),
               carla.Vector3D(-30, 167, 1.8431)]
        ex3 = [carla.Vector3D(42.5959, -4.3443, 1.8431),
               carla.Vector3D(22, -4, 1.8431),
               carla.Vector3D(9, -22, 1.8431)]
        milestones = [ex1, ex2, ex3]
        ms = max(0, min(args.milestone_number - 1, len(milestones) - 1))
        ex = milestones[ms]
        end = ex[-1]
        destination = ex[1]

        # Draw goals
        for i, point in enumerate(ex):
            loc = carla.Location(x=point.x, y=point.y, z=point.z + 2.0)
            world.debug.draw_point(loc, size=0.3,
                                   color=carla.Color(0, 0, 255), life_time=120.0)
            world.debug.draw_string(loc, "GOAL {}".format(i),
                                    color=carla.Color(255, 255, 255),
                                    life_time=120.0)

        start = get_start_point(world, ex[0])
        vehicle = try_spawn(start.transform, "vehicle.nissan.micra")
        if vehicle is None:
            print("Spawn failed")
            return
        world.tick()   # sync mode: let CARLA actually place the vehicle
        hud.log_event("Hero spawned")

        # ---- Camera --------------------------------------------------------
        camera = CameraView(vehicle, WINDOW_WIDTH, WINDOW_HEIGHT)

        # ---- Autopilot -----------------------------------------------------
        autopilot = ai.Autopilot(vehicle)

        # Route advancement
        running = [True]
        route_idx = [1]
        quit_timer = [0.0]

        def route_finished(ap):
            pos = ap.get_vehicle().get_transform().location
            hud.log_event("Arrived at waypoint")
            
            if route_idx[0] >= len(ex) - 1:
                if quit_timer[0] == 0.0: # Prevent spamming the timer
                    hud.log_event("EXERCISE FINISHED")
                    quit_timer[0] = time.time() + 3.0  # Set timer for 3 seconds
            else:
                route_idx[0] += 1
                ap.set_destination(ex[route_idx[0]])
                hud.log_event("Next goal: {}".format(route_idx[0]))

        autopilot.set_destination(destination)
        autopilot.set_route_finished_callback(route_finished)

        # ---- Malicious actor (M2 / -m 3) -----------------------------------
        if ms == 2:
            # Static obstacle 15m ahead
            ahead_wps = start.next(10.0)
            if ahead_wps:
                obstacle_tf = ahead_wps[0].transform
                obstacle_tf.location.z += 0.5
                obstacle = try_spawn(obstacle_tf, "vehicle.nissan.micra")
                if obstacle is not None:
                    c = carla.VehicleControl()
                    c.brake = 1.0
                    c.hand_brake = True
                    obstacle.apply_control(c)
                    world.tick()
                    hud.log_event("Static obstacle 15m ahead")

            # hud.log_event("Spawning malicious actor")
            # mal_spawn = start.get_right_lane()
            # if mal_spawn is not None:
            #     mal = try_spawn(mal_spawn.transform, "vehicle.nissan.micra")
            #     if mal is not None:
            #         cbp = world.get_blueprint_library().find('sensor.other.collision')
            #         csensor = world.spawn_actor(cbp, carla.Transform(), attach_to=mal)
            #         actor_list.append(csensor)

            #         def _on_mal_collision(event):
            #             kind = event.other_actor.type_id
            #             hud.log_event("MAL collision: " + kind)
            #             if kind.split('.')[0] == 'vehicle':
            #                 hud.log_event("TEST FAILED")

            #         csensor.listen(lambda e: _on_mal_collision(e))
            #         c = carla.VehicleControl()
            #         c.throttle = 1.0
            #         c.steer = -0.07
            #         mal.apply_control(c)

        # ---- Hero collision sensor (logs hits) -----------------------------
        hcbp = world.get_blueprint_library().find('sensor.other.collision')
        hero_collision = world.spawn_actor(hcbp, carla.Transform(), attach_to=vehicle)
        actor_list.append(hero_collision)

        def _on_hero_collision(event):
            kind = event.other_actor.type_id
            if kind.split('.')[0] == 'vehicle':
                hud.log_event("HERO HIT VEHICLE: " + kind)
            else:
                hud.log_event("Hero grazed: " + kind)
        hero_collision.listen(lambda e: _on_hero_collision(e))

        # ---- Main loop -----------------------------------------------------
        ctr = 0
        while running[0]:
            # Advance simulation by one fixed step. In sync mode this also
            # produces exactly one lidar frame, perfectly aligned with our tick.
            if quit_timer[0] > 0.0 and time.time() > quit_timer[0]:
                running[0] = False
            world.tick()

            # pygame events (window close, ESC to quit)
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    running[0] = False
                elif evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE:
                    running[0] = False

            status = autopilot.update()
            if status is None:
                ctr += 1
                if ctr > 3:
                    running[0] = False
            else:
                ctr = 0

            # Render: camera first (full background), HUD on top
            display.fill((0, 0, 0))
            camera.render(display)
            hud.render(display, vehicle, autopilot)
            pygame.display.flip()

    finally:
        # Restore async mode so the server isn't stuck waiting for ticks
        # on the next run / from manual_control / etc.
        try:
            if 'original_settings' in dir() or 'original_settings' in locals():
                world.apply_settings(original_settings)
        except Exception:
            pass

        if autopilot is not None:
            try:
                autopilot.destroy()
            except Exception:
                pass
        if camera is not None:
            camera.destroy()
        print('destroying actors')
        for actor in actor_list:
            try:
                actor.destroy()
            except Exception:
                pass
        if client is not None and not args.no_record:
            try:
                client.stop_recorder()
            except Exception:
                pass
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    main()