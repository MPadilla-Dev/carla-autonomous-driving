#!/usr/bin/env python
"""
race_demo.py
=============
Head-to-head race between two AIs. Spawns two cars at adjacent spawn points,
both head to the same finish line, prints who wins.

THREE OPPONENT MODES (--mode flag):

    --mode self_vs_self     (default)
        Both cars use our Autopilot. The OPPONENT is configured "naive":
        greedy planner, aim-at-next steering. Our HERO uses A* + pure pursuit.
        Reliable, educational, always works.

    --mode hero_vs_carla_tm
        Our hero uses our full Autopilot. Opponent is on CARLA's built-in
        Traffic Manager autopilot. The TM doesn't follow a destination, so
        the "race" is just: hero reaches goal, we time it. We also report
        the TM car's distance traveled for context.

    --mode hero_vs_behavior_agent
        Our hero uses our Autopilot. Opponent uses CARLA's BehaviorAgent
        (agents/navigation/behavior_agent.py). Both head to the same goal.
        Best apples-to-apples race, but requires the CARLA agents/ folder
        to be importable.

USAGE
-----
    python race_demo.py --mode self_vs_self --hud
    python race_demo.py --mode hero_vs_carla_tm
    python race_demo.py --mode hero_vs_behavior_agent

PRESENTATION TIP
----------------
self_vs_self is the most reliable for a live demo. The narrative is "we
built the same architecture twice with different planner+controller choices,
and you can see it matters". For a recording, hero_vs_behavior_agent is
the most impressive if it's working in your install.
"""

import glob
import os
import sys
import argparse
import time
import random
from datetime import datetime

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import custom_ai as ai
import ai_control as control_mod
import ai_parser as parser_mod


# =============================================================================
# Pre-built race routes (start, finish). Pick one or define your own.
# =============================================================================
RACE_ROUTES = {
    'short': {
        'start':  carla.Vector3D(42.5959, -4.3443, 1.8431),
        'finish': carla.Vector3D(9, -22, 1.8431),
    },
    'long': {
        'start':  carla.Vector3D(42.5959, -4.3443, 1.8431),
        'finish': carla.Vector3D(-30, 167, 1.8431),
    },
}

# Distance considered "finished" (m)
FINISH_TOLERANCE = 6.0


# =============================================================================
# Helpers
# =============================================================================
def get_nearest_spawn(world, target_vec):
    spawns = world.get_map().get_spawn_points()
    return min(spawns, key=lambda sp: sp.location.distance(
        carla.Location(x=target_vec.x, y=target_vec.y, z=target_vec.z)))


def configure_autopilot_naive(autopilot):
    """Make this Autopilot use the 'before' configuration: greedy + aim."""
    autopilot.planner.PLANNER_MODE = 'greedy'
    autopilot.executor.STEER_MODE = 'aim'


def configure_autopilot_full(autopilot):
    """Make this Autopilot use the 'after' configuration: A* + pure pursuit."""
    autopilot.planner.PLANNER_MODE = 'astar'
    autopilot.executor.STEER_MODE = 'pure_pursuit'


def reached(vehicle, finish_loc):
    return vehicle.get_transform().location.distance(finish_loc) < FINISH_TOLERANCE


def make_recording_path(label):
    here = os.path.dirname(os.path.abspath(__file__))
    rec_dir = os.path.join(here, "recordings")
    os.makedirs(rec_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(rec_dir, "{}_{}.log".format(stamp, label))


# =============================================================================
# Main
# =============================================================================
def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--mode',
        choices=['self_vs_self', 'hero_vs_carla_tm', 'hero_vs_behavior_agent'],
        default='self_vs_self')
    argparser.add_argument('--route', choices=list(RACE_ROUTES.keys()),
                           default='long')
    argparser.add_argument('--town', default='Town03')
    argparser.add_argument('--no-record', action='store_true')
    argparser.add_argument('--hud', action='store_true')
    argparser.add_argument('--max-seconds', type=float, default=180.0,
                           help='Race timeout (default 180s)')
    args = argparser.parse_args()

    actor_list = []
    hero_ai = None
    opp_ai = None
    behavior_agent = None
    hud = None
    client = None

    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        world = client.load_world(args.town)

        # ---- Recorder -------------------------------------------------------
        if not args.no_record:
            rec_path = make_recording_path("race_" + args.mode)
            print("Recording to:", rec_path)
            print(client.start_recorder(rec_path, True))

        route = RACE_ROUTES[args.route]
        start_vec = route['start']
        finish_vec = route['finish']
        finish_loc = carla.Location(x=finish_vec.x, y=finish_vec.y, z=finish_vec.z)

        # ---- Visualize start/finish -----------------------------------------
        world.debug.draw_point(
            carla.Location(x=start_vec.x, y=start_vec.y, z=start_vec.z + 2.5),
            size=0.5, color=carla.Color(0, 255, 0), life_time=120.0)
        world.debug.draw_string(
            carla.Location(x=start_vec.x, y=start_vec.y, z=start_vec.z + 3.0),
            "START", draw_shadow=True,
            color=carla.Color(255, 255, 255), life_time=120.0)
        world.debug.draw_point(
            carla.Location(x=finish_vec.x, y=finish_vec.y, z=finish_vec.z + 2.5),
            size=0.5, color=carla.Color(255, 0, 0), life_time=120.0)
        world.debug.draw_string(
            carla.Location(x=finish_vec.x, y=finish_vec.y, z=finish_vec.z + 3.0),
            "FINISH", draw_shadow=True,
            color=carla.Color(255, 255, 255), life_time=120.0)

        # ---- Vehicle blueprints ---------------------------------------------
        bp_lib = world.get_blueprint_library()
        hero_bp = bp_lib.find('vehicle.nissan.micra')
        hero_bp.set_attribute('color', '255,0,0')   # red
        opp_bp = bp_lib.find('vehicle.nissan.micra')
        opp_bp.set_attribute('color', '0,0,255')    # blue

        # ---- Spawn ----------------------------------------------------------
        start_sp = get_nearest_spawn(world, start_vec)
        # Hero in left lane, opponent in right lane (or adjacent)
        hero_wp = world.get_map().get_waypoint(start_sp.location)
        right_wp = hero_wp.get_right_lane()

        hero_tf = hero_wp.transform
        opp_tf = right_wp.transform if right_wp is not None else hero_wp.next(8.0)[0].transform

        hero = world.try_spawn_actor(hero_bp, hero_tf)
        if hero is None:
            print("Failed to spawn HERO")
            return
        actor_list.append(hero)

        opp = world.try_spawn_actor(opp_bp, opp_tf)
        if opp is None:
            print("Failed to spawn OPPONENT")
            return
        actor_list.append(opp)

        print("HERO spawned at", hero_tf.location)
        print("OPPONENT spawned at", opp_tf.location)

        # ---- Configure HERO (always our full AI) ----------------------------
        hero_ai = ai.Autopilot(hero)
        configure_autopilot_full(hero_ai)
        hero_ai.set_destination(finish_vec)
        hero_finish_time = [None]

        def hero_finished(_):
            hero_finish_time[0] = time.time()
            print(">>> HERO finished")

        hero_ai.set_route_finished_callback(hero_finished)

        # ---- Configure OPPONENT based on mode -------------------------------
        opp_finish_time = [None]
        opp_total_distance = [0.0]
        opp_prev_loc = [opp_tf.location]

        if args.mode == 'self_vs_self':
            opp_ai = ai.Autopilot(opp)
            configure_autopilot_naive(opp_ai)
            opp_ai.set_destination(finish_vec)

            def opp_finished(_):
                opp_finish_time[0] = time.time()
                print(">>> OPPONENT finished")
            opp_ai.set_route_finished_callback(opp_finished)

        elif args.mode == 'hero_vs_carla_tm':
            tm = client.get_trafficmanager()
            tm.set_synchronous_mode(False)
            opp.set_autopilot(True, tm.get_port())

        elif args.mode == 'hero_vs_behavior_agent':
            try:
                # Try to import CARLA's BehaviorAgent
                carla_root = os.environ.get('CARLA_ROOT', None)
                candidates = []
                if carla_root:
                    candidates.append(os.path.join(carla_root, 'PythonAPI', 'carla'))
                # Look up two levels (examples folder -> carla root)
                here = os.path.dirname(os.path.abspath(__file__))
                candidates.append(os.path.join(here, '..', 'carla'))
                candidates.append(os.path.join(here, '..', '..', 'PythonAPI', 'carla'))
                for c in candidates:
                    if os.path.isdir(c) and c not in sys.path:
                        sys.path.append(c)
                from agents.navigation.behavior_agent import BehaviorAgent
                behavior_agent = BehaviorAgent(opp, behavior='normal')
                behavior_agent.set_destination(finish_loc)
            except Exception as e:
                print("BehaviorAgent unavailable, falling back to TM:", e)
                args.mode = 'hero_vs_carla_tm'
                tm = client.get_trafficmanager()
                opp.set_autopilot(True, tm.get_port())

        # ---- Optional HUD ---------------------------------------------------
        if args.hud:
            try:
                from telemetry_hud import TelemetryHUD
                hud = TelemetryHUD(title="HERO Telemetry")
            except Exception as e:
                print("HUD unavailable:", e)

        # ---- Race loop ------------------------------------------------------
        race_start = time.time()
        deadline = race_start + args.max_seconds
        print("\n=== RACE STARTED ===\n")

        while True:
            now = time.time()
            if now > deadline:
                print(">>> Race timed out at", args.max_seconds, "s")
                break

            # HERO update
            hero_ai.update()

            # OPPONENT update
            if args.mode == 'self_vs_self':
                opp_ai.update()
            elif args.mode == 'hero_vs_behavior_agent' and behavior_agent is not None:
                if not behavior_agent.done():
                    ctrl = behavior_agent.run_step()
                    opp.apply_control(ctrl)
                else:
                    if opp_finish_time[0] is None:
                        opp_finish_time[0] = now
                        print(">>> OPPONENT (BehaviorAgent) finished")

            # Check finish-line proximity (independent of internal callbacks,
            # since CARLA-controlled cars don't fire our callbacks)
            if hero_finish_time[0] is None and reached(hero, finish_loc):
                hero_finish_time[0] = now
                print(">>> HERO crossed finish")
            if opp_finish_time[0] is None and reached(opp, finish_loc):
                opp_finish_time[0] = now
                print(">>> OPPONENT crossed finish")

            # Track opponent distance for TM mode (no goal, so we report
            # how far it travelled)
            curr_loc = opp.get_transform().location
            opp_total_distance[0] += curr_loc.distance(opp_prev_loc[0])
            opp_prev_loc[0] = curr_loc

            # HUD
            if hud is not None:
                hud.update(hero_ai.get_telemetry())
                hud.render()

            if hero_finish_time[0] is not None and opp_finish_time[0] is not None:
                break

            time.sleep(0.05)

        # ---- Results --------------------------------------------------------
        print("\n=== RACE RESULTS ===")
        if hero_finish_time[0] is not None:
            print("HERO: {:.2f} s".format(hero_finish_time[0] - race_start))
        else:
            print("HERO: did not finish")
        if opp_finish_time[0] is not None:
            print("OPPONENT: {:.2f} s".format(opp_finish_time[0] - race_start))
        else:
            print("OPPONENT: did not finish (distance traveled: {:.1f} m)".format(
                opp_total_distance[0]))

        if hero_finish_time[0] and opp_finish_time[0]:
            winner = "HERO" if hero_finish_time[0] < opp_finish_time[0] else "OPPONENT"
            margin = abs(hero_finish_time[0] - opp_finish_time[0])
            print("WINNER: {} by {:.2f} s".format(winner, margin))
        elif hero_finish_time[0]:
            print("WINNER: HERO (opponent did not finish)")
        elif opp_finish_time[0]:
            print("WINNER: OPPONENT (hero did not finish)")
        print("====================\n")

    finally:
        if hud is not None:
            try: hud.close()
            except Exception: pass

        for inst in (hero_ai, opp_ai):
            if inst is not None:
                try: inst.destroy()
                except Exception: pass

        if client is not None and not args.no_record:
            try:
                print("Stopping recorder")
                client.stop_recorder()
            except Exception: pass

        print("destroying actors")
        for actor in actor_list:
            try: actor.destroy()
            except Exception: pass
        print("done.")


if __name__ == '__main__':
    main()