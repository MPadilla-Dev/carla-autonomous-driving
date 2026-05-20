#!/usr/bin/env python
import carla
import pygame
import numpy as np
import weakref
import time
import os
import sys

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

class CameraView(object):
    def __init__(self, vehicle, width, height):
        self.surface = None
        world = vehicle.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        bp.set_attribute('fov', '90')
        cam_tf = carla.Transform(carla.Location(x=-6.0, z=6.0), carla.Rotation(pitch=-20.0))
        self.sensor = world.spawn_actor(bp, cam_tf, attach_to=vehicle)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda img: CameraView._on_frame(weak_self, img))

    @staticmethod
    def _on_frame(weak_self, image):
        self = weak_self()
        if not self: return
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()

class HUD(object):
    def __init__(self):
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_big = pygame.font.SysFont("consolas", 26, bold=True)

    def render(self, display, vehicle):
        ctrl = vehicle.get_control()
        vel = vehicle.get_velocity()
        speed_kmh = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2)**0.5

        # Speed Panel
        pygame.draw.rect(display, (0, 0, 0, 160), (10, 10, 250, 90))
        display.blit(self.font.render("REPLAY MODE", True, (250, 200, 70)), (20, 20))
        display.blit(self.font_big.render(f"{speed_kmh:.1f} km/h", True, (255, 255, 255)), (20, 50))

        # Controls Panel
        rx = WINDOW_WIDTH - 290
        pygame.draw.rect(display, (0, 0, 0, 160), (rx, 10, 280, 130))
        display.blit(self.font.render("CONTROLS", True, (180, 180, 190)), (rx + 10, 18))
        display.blit(self.font.render(f"throttle: {ctrl.throttle:.2f}", True, (255, 255, 255)), (rx + 10, 40))
        display.blit(self.font.render(f"brake:    {ctrl.brake:.2f}", True, (255, 255, 255)), (rx + 10, 62))
        display.blit(self.font.render(f"steer:    {ctrl.steer:+.2f}", True, (255, 255, 255)), (rx + 10, 84))

def main():
    pygame.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Replay Viewer")
    hud = HUD()

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 1. Use absolute path so the server finds the file
    log_path = os.path.abspath("roundabout_test.log")
    print(f"Starting replay from: {log_path}")
    
    # client.replay_file returns a string with info about the replay (good for debugging)
    print(client.replay_file(log_path, 0, 0, 0))
    
    # 2. Smart wait loop: check every 0.5s for up to 5 seconds
    vehicles = []
    print("Waiting for vehicles to spawn in replay...")
    for _ in range(10):
        time.sleep(0.5)
        vehicles = world.get_actors().filter('vehicle.*')
        if vehicles:
            break
            
    if not vehicles:
        print("No vehicles found! Make sure you let the recording script run for at least a few seconds before pressing Ctrl+C.")
        return

    # Pick the first vehicle to follow
    hero = vehicles[0]

    # Pick the first vehicle to follow
    hero = vehicles[0]
    print(f"Attached camera to {hero.type_id}")
    camera = CameraView(hero, WINDOW_WIDTH, WINDOW_HEIGHT)

    running = True
    try:
        while running:
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT or (evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE):
                    running = False

            display.fill((0, 0, 0))
            camera.render(display)
            hud.render(display, hero)
            pygame.display.flip()

    finally:
        camera.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()