#!/usr/bin/env python

import glob
import os
import sys
from collections import deque
import math
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

# Executor is responsible for moving the vehicle around
# In this implementation it only needs to match the steering and speed so that we arrive at provided waypoints
# BONUS TODO: implement different speed limits so that planner would also provide speed target speed in addition to direction
class Executor(object):
  def __init__(self, knowledge, vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
    self.target_pos = knowledge.get_location()
    


  def apply_stop(self):
    control = carla.VehicleControl()
    control.throttle = 0.0
    control.brake = 1.0
    control.steer = 0.0
    control.hand_brake = False
    self.vehicle.apply_control(control)

  def update(self, time_elapsed):
    status = self.knowledge.get_status()
    if status == Status.DRIVING or status == Status.HEALING:
        dest = self.knowledge.get_current_destination()
        self.update_control(dest, [1], time_elapsed)
    else:
        self.apply_stop()

  ### Working code for M1
  # #Update the executor at some intervals to steer the car in desired direction
  # def update(self, time_elapsed):
  #   status = self.knowledge.get_status()
  #   #TODO: this needs to be able to handle
  #   if status == Status.DRIVING:
  #     dest = self.knowledge.get_current_destination()
  #     self.update_control(dest, [1], time_elapsed)

  # TODO: Take into account that exiting the crash site could also be done in reverse, so there might need to be additional data passed between planner and executor, or there needs to be some way to tell this that it is ok to drive in reverse during HEALING and CRASHED states. An example is additional_vars, that could be a list with parameters that can tell us which things we can do (for example going in reverse)
  def update_control(self, destination, additional_vars, delta_time):
    vehicle_transform = self.vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_velocity = self.vehicle.get_velocity()

    current_speed = 3.6 * math.sqrt(
        vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)

    try:
      target_speed = self.knowledge.retrieve_data('target_speed')
    except KeyError:
      target_speed = 40  # km/h default; Milestone 3 sets this properly via knowledge

    dest_loc = carla.Location(x=destination.x, y=destination.y, z=destination.z)
    distance = vehicle_location.distance(dest_loc)

    # Build 2D forward and target vectors
    fwd = vehicle_transform.get_forward_vector()
    fwd_arr = np.array([fwd.x, fwd.y])
    tgt_arr = np.array([dest_loc.x - vehicle_location.x,
                        dest_loc.y - vehicle_location.y])

    fwd_norm = np.linalg.norm(fwd_arr)
    tgt_norm = np.linalg.norm(tgt_arr)

    steer = 0.0
    if fwd_norm > 0.001 and tgt_norm > 0.001:
      fwd_unit = fwd_arr / fwd_norm
      tgt_unit = tgt_arr / tgt_norm

      # Angle between forward and target directions
      dot = float(np.clip(np.dot(fwd_unit, tgt_unit), -1.0, 1.0))
      angle = math.acos(dot)

      # Cross product z-component determines left/right
      cross = fwd_unit[0] * tgt_unit[1] - fwd_unit[1] * tgt_unit[0]
      if cross < 0:
        angle = -angle

      # Map angle to [-1, 1]: 90 degrees = full lock
      steer = max(-1.0, min(1.0, angle / (math.pi / 2)))

    # print(f"dest: ({dest_loc.x:.1f},{dest_loc.y:.1f}) dist: {distance:.1f} speed: {current_speed:.1f}")
    if current_speed < target_speed - 5:
      throttle = 0.7
      brake = 0.0
    elif current_speed < target_speed:
      throttle = 0.3
      brake = 0.0
    else:
      throttle = 0.0
      brake = 0.2

    control = carla.VehicleControl()
    control.throttle = throttle
    control.steer = steer
    control.brake = brake
    control.hand_brake = False
    self.vehicle.apply_control(control)

# Planner is responsible for creating a plan for moving around
# In our case it creates a list of waypoints to follow so that vehicle arrives at destination
# Alternatively this can also provide a list of waypoints to try avoid crashing or 'uncrash' itself
class Planner(object):
  def __init__(self, knowledge, vehicle):
    self.knowledge = knowledge
    self.path = deque([])
    self.vehicle = vehicle  # needed for world access


  # Create a map of waypoints to follow to the destination and save it
  def make_plan(self, source, destination):
    self.path = self.build_path(source,destination)
    self.update_plan()
    self.knowledge.update_destination(self.get_current_destination())
    self.draw_path(life_time=60.0)  # ← add this
  
  # Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    self.update_plan()
    self.knowledge.update_destination(self.get_current_destination())
  
  #Update internal state to make sure that there are waypoints to follow and that we have not arrived yet
  def update_plan(self):
    if len(self.path) == 0:
      return
    
    if self.knowledge.arrived_at(self.path[0]):
      self.path.popleft()
    
    if len(self.path) == 0:
      self.knowledge.update_status(Status.ARRIVED)
    else:
      self.knowledge.update_status(Status.DRIVING)

  #get current destination 
  def get_current_destination(self):
    status = self.knowledge.get_status()
    #if we are driving, then the current destination is next waypoint
    if status == Status.DRIVING:
      #TODO: Take into account traffic lights and other cars
      return self.path[0]
    if status == Status.ARRIVED:
      return self.knowledge.get_location()
    if status == Status.HEALING:
      #TODO: Implement crash handling. Probably needs to be done by following waypoint list to exit the crash site.
      #Afterwards needs to remake the path.
      return self.knowledge.get_location()
    if status == Status.CRASHED:
      #TODO: implement function for crash handling, should provide map of wayoints to move towards to for exiting crash state. 
      #You should use separate waypoint list for that, to not mess with the original path. 
      return self.knowledge.get_location()
    #otherwise destination is same as current position
    return self.knowledge.get_location()

  #TODO: Implementation
  def build_path(self, source, destination):
    self.path = deque([])

    if self.vehicle is None:
        # Fallback: no map access, use direct line
        self.path.append(destination)
        return self.path

    carla_map = self.vehicle.get_world().get_map()

    # source comes in as a Transform from custom_ai.py's set_destination
    if hasattr(source, 'location'):
        source_loc = source.location
    else:
        source_loc = carla.Location(x=source.x, y=source.y, z=source.z)

    dest_loc = carla.Location(x=destination.x, y=destination.y, z=destination.z)

    current_wp = carla_map.get_waypoint(source_loc)
    step_distance = 2.0       # meters between waypoints
    max_steps = 1000          # safety cap

    for _ in range(max_steps):
        wp_loc = current_wp.transform.location
        if wp_loc.distance(dest_loc) < 5.0:
            break

        next_wps = current_wp.next(step_distance)
        if not next_wps:
            break  # dead end

        # Greedy: pick the next waypoint closest to destination
        current_wp = min(next_wps,
                         key=lambda w: w.transform.location.distance(dest_loc))

        self.path.append(current_wp.transform.location)


    # Final destination
    self.path.append(dest_loc)
    return self.path
  
  ### new
  def draw_path(self, life_time=10.0):
    if self.vehicle is None or len(self.path) == 0:
        return
    world = self.vehicle.get_world()
    waypoints = list(self.path)

    for i, wp in enumerate(waypoints):
        loc = carla.Location(x=wp.x, y=wp.y, z=wp.z + 1.0)
        world.debug.draw_point(
            loc, size=0.15,
            color=carla.Color(255, 0, 0),
            life_time=life_time)
        world.debug.draw_string(
            loc, str(i),
            draw_shadow=False,
            color=carla.Color(255, 255, 255),
            life_time=life_time)

    for a, b in zip(waypoints, waypoints[1:]):
        world.debug.draw_line(
            carla.Location(x=a.x, y=a.y, z=a.z + 1.0),
            carla.Location(x=b.x, y=b.y, z=b.z + 1.0),
            thickness=0.1,
            color=carla.Color(0, 255, 0),
            life_time=life_time)