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
action_names = ['forward', 'stay', 'leftturn', 'rightturn']
action_specs = [[5., 0.], [0., 0.], [0., np.pi / 9.], [0, - np.pi / 9.]]
robot_diameter = 10
robot_width = 0
heading_length = 8
heading_width = 2
screen_size = screen_width, screen_height = 600, 500
boundary = {'min':[50, 50], 'max':[550, 450]}
game_fps_limit = 15

ball_detect_range = np.pi / 4.0 # 45 degrees
ball_state_bit_size = 5
default_ball_distance = 50
ball_diameter = 15

class Color:
  black = 0, 0, 0
  white = 255, 255, 255
  red = 255, 0, 0
  green = 0, 255, 0
  blue = 0, 0, 255
  pink = 255, 192, 203
  orange = 255, 165, 0

class Robot:
  def __init__(self, x, y, heading):
    self.pos = (x, y)
    self.y = y
    self.heading = heading
    self.diameter = robot_diameter

  def get_pose(self):
    return (self.pos, self.heading)

  def draw(self, screen):
    xx = (self.diameter + heading_length)* np.cos(self.heading)
    yy = (self.diameter + heading_length)* np.sin(self.heading)
    pygame.draw.circle(screen, Color.black, self.pos, self.diameter, robot_width)
    pygame.draw.line(screen, Color.black, self.pos, (self.pos[0] + xx, self.pos[1] + yy), heading_width)

  def step(self, action):
    # update upon action
    x = self.pos[0] + action_specs[action][0] * np.cos(self.heading)
    y = self.pos[1] + action_specs[action][0] * np.sin(self.heading)
    heading = self.heading + action_specs[action][1]

    # check the boundary
    if x < boundary['min'][0]:
      x = boundary['min'][0]
    if x > boundary['max'][0]:
      x = boundary['max'][0]
    if y < boundary['min'][1]:
      y = boundary['min'][1]
    if y > boundary['max'][1]:
      y = boundary['max'][1]

    # update the current status
    self.pos = (int(x), int(y))
    self.heading = heading

def reposition_angle(angle):
  a = angle
  if a < 0:
    a = np.pi * 2.0 + a
  a = a % (2 * np.pi)
  if a > np.pi:
    a = a - np.pi * 2 
  return a

def overlap((a, b), (c, d)):
  return (a >= c and a <= d) or (b >= c and b <= d) or (c >= a and c <= b) or (d >= a and d <= b)

class Ball:
  def __init__(self, name, robot):
    robot_pos, robot_heading = robot.get_pose()

    # first initialize a pink in front of a robot
    x = robot_pos[0] + default_ball_distance * np.cos(robot_heading)
    y = robot_pos[1] + default_ball_distance * np.sin(robot_heading)

    self.name = name
    self.pos = (x, y)
    self.present = True

  def set_position(self, pos):
    if pos[0] < 0:
      self.present = False
    else:
      self.present = True
    self.pos = pos

  def draw(self, screen):
    if self.present:
      pygame.draw.circle(screen, Color.pink, self.pos, ball_diameter, 0)

  def visualize(self, robot):
    bits = np.zeros(ball_state_bit_size)

    if self.pos[0] >= 0:
      robot_pos, robot_heading = robot.get_pose()
      x = self.pos[0] - robot_pos[0]
      y = self.pos[1] - robot_pos[1]
      angle = np.arctan2(y, x)
      
      xx = self.pos[0] + float(ball_diameter) / 2.0 * np.cos(angle + np.pi / 2.0) - robot_pos[0]
      yy = self.pos[1] + float(ball_diameter) / 2.0 * np.sin(angle + np.pi / 2.0) - robot_pos[1]

      angle_plus = np.arctan2(yy, xx)
      angle_minus = angle - (angle_plus - angle)

      angle_plus = reposition_angle(angle_plus - robot_heading)
      angle_minus = reposition_angle(angle_minus - robot_heading)

      bits = np.zeros(ball_state_bit_size)

      start_angle = - ball_detect_range / 2.0
      interval = ball_detect_range / float(ball_state_bit_size)
      # inclusive
      for i in range(ball_state_bit_size):
        s_angle = start_angle + i * interval
        e_angle = start_angle + (i + 1) * interval
        if overlap((angle_minus, angle_plus), (s_angle, e_angle)):
          bits[i] = 1

      #print(self.name, angle_plus, angle_minus, start_angle, bits)
    return bits

def follow_reward(state, action):
    # If no signal is firing in state, all actions except staying still are +1
    if sum(state) == 0:
        if action != 'stay':
            return 1.0
        else:
            return -1.0

    # If all signal bits are firing then we are as close to the ball as we should be
    # Any further movement actions should be -1
    if sum(state) == len(state):
        if action != 'stay':
            return -1.0
        else:
            return 1.0

    mid = int(len(state) / 2)
    # If more signal appears to the left than the right
    if sum(state[:mid]) > sum(state[mid:]):
        if action == 'leftturn':
            return 1.0
        else:
            return -1.0
    # If more signal appears to the right than the left
    if sum(state[:mid]) < sum(state[mid:]):
        if action == 'rightturn':
            return 1.0
        else:
            return -1.0

    #If there is only signal in the center of the robot's field of view
    if sum(state[mid-1:mid+2]) > sum(state[:mid-1]) + sum(state[mid+2:]):
        if action == 'forward':
            return 1.0
        else:
            return -1.0

    print 'State-action pair did not fall into any valid return condition!'
    return 0.0

class JamesEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, is_human_input = True):
    self._seed()
    self.is_human_input = is_human_input
    self.screen = pygame.display.set_mode(screen_size)
    self.clock = pygame.time.Clock()
    
    pygame.display.set_caption("James' truncated domain")

    self.world = Box2D.b2World()
    self.robot = Robot(100, 100, np.pi/2)
    self.human_pos = (-1, -1)

    self.pink_ball = Ball('pink', self.robot)
    #self.orange_pole_pos = (200, 200)
    #self.green_pole_pos = (400, 200)

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

    self.robot.step(action)

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

      self.pink_ball.set_position(self.human_pos)
      state = self.pink_ball.visualize(self.robot)
      reward = follow_reward(state, action_names[action])

      self.clock.tick(game_fps_limit)

    return state, reward, done, info

  def _render(self, mode='human', close=False):
    self.screen.fill(Color.white)
    self.robot.draw(self.screen)
    self.pink_ball.draw(self.screen)

    pygame.display.flip()
    

if __name__ == "__main__":
  env = JamesEnv()
  s = env.reset()
  total_reward = 0
  extended_actions = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [2, 2, 2], [3, 3, 3]]
  sub_steps = 0 
  steps = 0
  ea = 0
  while True:
    if sub_steps == 0:
      ea = np.random.randint(len(extended_actions))

    a = extended_actions[ea][sub_steps]
    sub_steps += 1
    if sub_steps == len(extended_actions[ea]) - 1:
      sub_steps = 0

    s, r, done, info = env.step(a)
    print(s, r)
    env.render()
    total_reward += r
    steps += 1
    if done:
      break
  env.close()
