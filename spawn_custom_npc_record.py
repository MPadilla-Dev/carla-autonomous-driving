#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

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

import argparse
import random
import time

import custom_ai as ai

def main():
    actor_list = []
    vai_list = []
    collision_sensors = [] # ADD THIS
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=6,
        type=int,
        help='number of vehicles (default: 4)')
    argparser.add_argument(
        '-d', '--delay',
        metavar='D',
        default=2.0,
        type=float,
        help='delay in seconds between spawns (default: 2.0)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    args = argparser.parse_args()

    actor_list = []
    vai_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10)

    try:

        world = client.get_world()
        world = client.load_world('Town03')
        # Start Recorder
        client.start_recorder(os.path.abspath("roundabout_test.log"), True)
        
        # Tracking dictionaries
        crash_counts = {}
        heal_counts = {}
        prev_status = {}
        
        blueprints = world.get_blueprint_library().filter('vehicle.nissan.micra')
        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]

        #This is definition of a callback function that will be called when the autopilot arrives at destination
        def route_finished(autopilot):
            print("Vehicle arrived at destination")
            #After vehicle has arrived we set a random spawn point as a new destination
            #TODO: Make an 'intelligent' list of targets where cars could go (has annotated waypoints so you could use that)
            autopilot.set_destination(random.choice(world.get_map().get_spawn_points()).location)
            # controller.set_destination(random.choice(world.get_map().get_spawn_points()))
            #TODO, BONUS: Make fixed exit and entry points (for example parking lots), 
            #so that cars are removed from simulation when they enter those and new ones are created from random points.
            #use try_spawn_random_vehicle_at(random.choice(spawn_points)) to spawn new cars
            

        #Function to spawn new vehicles
        def try_spawn_random_vehicle_at(transform):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                vid = vehicle.id
                crash_counts[vid] = 0
                heal_counts[vid] = 0
                prev_status[vid] = None
                actor_list.append(vehicle)
                
                # Attach Collision Sensor
                cbp = world.get_blueprint_library().find('sensor.other.collision')
                csensor = world.spawn_actor(cbp, carla.Transform(), attach_to=vehicle)
                collision_sensors.append(csensor)
                actor_list.append(csensor)
                
                def on_collision(event, v_id=vid):
                    crash_counts[v_id] += 1
                    hit_target = event.other_actor.type_id
                    print(f"[CRASH] Actor {v_id} hit {hit_target}. Total crashes for {v_id}: {crash_counts[v_id]}")
                
                csensor.listen(on_collision)

                autopilot = ai.Autopilot(vehicle)
                autopilot.set_route_finished_callback(route_finished)
                autopilot.set_destination(random.choice(world.get_map().get_spawn_points()).location)
                vai_list.append(autopilot)

                print('spawned %r (ID: %d) at %s' % (vehicle.type_id, vid, transform.location))
                return True
            return False

        # Get all spawn points, but filter for the roundabout bounding box (-50 to 50 on X and Y)
        # Get points near the roundabout that face inwards
        all_points = world.get_map().get_spawn_points()
        spawn_points = []
        for p in all_points:
            # Bound to the roundabout area
            if -60 <= p.location.x <= 60 and -60 <= p.location.y <= 60:
                # Vector pointing to the center (0,0)
                vx, vy = -p.location.x, -p.location.y
                # Vehicle's forward vector
                fx = p.rotation.get_forward_vector().x
                fy = p.rotation.get_forward_vector().y
                
                # Dot product > 0 means the vehicle is facing towards the center
                if (vx * fx + vy * fy) > 0:
                    spawn_points.append(p)
                    
        random.shuffle(spawn_points)
        # spawn_points = [p for p in all_points if -50 <= p.location.x <= 50 and -50 <= p.location.y <= 50]
        # random.shuffle(spawn_points)	

        print('found %d spawn points.' % len(spawn_points))

        count = args.number_of_vehicles

        for spawn_point in spawn_points:
            if try_spawn_random_vehicle_at(spawn_point):
                count -= 1
            if count <= 0:
                break

        print('spawned %d vehicles, press Ctrl+C to exit.' % args.number_of_vehicles)

        # Infinite loop to update car statuses
        while True:
            for controller in vai_list:
                status = controller.update()
                vid = controller.get_vehicle().id
                
                # Check for transition into HEALING
                if status and status.name == 'HEALING' and prev_status[vid] != 'HEALING':
                    heal_counts[vid] += 1
                    print(f"[HEALING] Actor {vid} triggered avoidance! Total for {vid}: {heal_counts[vid]}")
                
                if status:
                    prev_status[vid] = status.name
                    
            time.sleep(0.05)
                    

    finally:
        print('\nCleaning up...')
        client.stop_recorder()
        
        # Stop AI modules (Lidar/Lane sensors)
        for controller in vai_list:
            try: controller.destroy()
            except Exception: pass
            
        # Batch 1: Destroy attached collision sensors FIRST
        client.apply_batch([carla.command.DestroyActor(x.id) for x in collision_sensors])
        
        # Batch 2: Destroy the vehicles SECOND
        client.apply_batch([carla.command.DestroyActor(x.id) for x in actor_list if x not in collision_sensors])


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
