# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import gym
import os
import numpy as np
import cv2
from utils import *
from gym import error, spaces
from gym.utils import seeding
import logging
import pygame, pygame.mixer
from pygame.locals import *
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

nb_actions = 4
dimensions = 2
action_names = ['forward', 'backward', 'leftturn', 'rightturn']
action_specs = [[0.2, 0.], [-0.2, 0.], [0., 1.0], [0, -1.0]]
robot_diameter = 10
robot_width = 0
screen_size = screen_width, screen_height = 640, 480
game_fps_limit = 60

class Color:
  black = 0, 0, 0
  white = 255, 255, 255
  red = 255, 0, 0
  green = 0, 255, 0
  blue = 0, 0, 255
  pink = 255, 192, 203
  orange = 255, 165, 0

class JamesEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, is_human_input = True):
    self._seed()
    self.is_human_input = is_human_input
    self.screen = pygame.display.set_mode(screen_size)
    self.clock = pygame.time.Clock()
    
    pygame.display.set_caption("James' truncated domain")

    self.world = Box2D.b2World()
    self.robot_pos = (100, 100)
    self.human_pos = (-1, -1)
    self.orange_pole_pos = (200, 200)
    self.green_pole_pos = (400, 200)

    self.prev_reward = None

    high = np.array([np.inf]*2) 
    self.observation_space = spaces.Box(-high, high)

    self.action_space = spaces.Discrete(nb_actions)

    self._reset()


  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _close(self):
    pygame.quit()
    pass

  def _reset(self):
    pass

  def _create_human_input(self):
    pass

  def _step(self, action):
    info = action_names[action]+" action has taken. "
    state = None
    reward = 0.0
    done = False

    if self.is_human_input:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          done = True

      keys = pygame.key.get_pressed()
      if keys[pygame.K_q]:
        print("-1 !!!")
      elif keys[pygame.K_w]:
        print("0 !!!")
      elif keys[pygame.K_e]:
        print("1 !!!")

      if pygame.mouse.get_pressed()[0]:
        human_pos = pygame.mouse.get_pos()
        print(human_pos)
        self.human_pos = human_pos
      else:
        self.human_pos = (-1, -1)

      self.clock.tick(game_fps_limit)
    return state, reward, done, info

  def _render(self, mode='human', close=False):
    self.screen.fill(Color.white)
    pygame.draw.circle(self.screen, Color.orange, self.orange_pole_pos, robot_diameter, robot_width)
    pygame.draw.circle(self.screen, Color.green, self.green_pole_pos, robot_diameter, robot_width)
    pygame.draw.circle(self.screen, Color.black, self.robot_pos, robot_diameter, robot_width)
    if self.human_pos[0] != -1:
      pygame.draw.circle(self.screen, Color.pink, self.human_pos, robot_diameter, robot_width)

    pygame.display.flip()
    

if __name__ == "__main__":
  env = JamesEnv()
  s = env.reset()
  total_reward = 0
  steps = 0
  while True:
    a = np.random.randint(nb_actions)
    s, r, done, info = env.step(a)
    env.render()
    total_reward += r
    steps += 1
    if done:
      break
  env.close()
