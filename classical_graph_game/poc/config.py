

import gamms
import numpy as np

MAP_FILE = "/Users/jmalegaonkar/infogamms/maps/map1.bson"
VIS_ENGINE = gamms.visual.Engine.PYGAME
NUM_AGENTS = 4
MRX_SENSOR_RANGE = 30.0  # meters
MRX_FOV = 2 * np.pi  # 360 degrees 
MAP_SENSOR_RANGE = 20.0  # meters
MAP_FOV = 2 * np.pi  # 360 degrees
CATCH_DISTANCE = 2.0  # meters