# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import gym
import os
import numpy as np
import cv2
from utils import *
from gym import error, spaces
from gym.utils import seeding
import logging

class JamesEnv(gym.Env):


