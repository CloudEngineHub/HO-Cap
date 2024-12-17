# Standard library imports
import os
import sys
import gc
import time
import math
import json
import shutil
import pickle as pkl
import argparse
import itertools
import multiprocessing
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
import logging
import concurrent.futures

# Third-party libraries
from ruamel.yaml import YAML
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d, CubicSpline
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import open3d as o3d
import open3d.core as o3c
import trimesh
import pyrender
import av
import torch

yaml = YAML()
yaml.default_flow_style = False
yaml.indent(mapping=2, sequence=4, offset=2)
