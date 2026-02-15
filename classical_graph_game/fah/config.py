import gamms
import math
# Visualization
VIS_ENGINE = gamms.visual.Engine.PYGAME
MAP_FILE = "back_of_fah.osm"
# Agent settings
NUM_AGENTS = 1
CATCH_DISTANCE = 10.0  # meters to catch Mr. X

# Sensor ranges
MAP_SENSOR_RANGE = 50.0  # How far agents can see the map
MAP_FOV = math.pi / 3  # 120 degree FOV
MRX_SENSOR_RANGE = 30.0  # Detection range for Mr. X
MRX_FOV = math.pi / 3  # 120 degree FOV

# Frontier search parameters
FRONTIER_ASSIGNMENT_INTERVAL = 20  # Steps between frontier reassignments
MIN_FRONTIER_DISTANCE = 15.0  # Minimum distance between agent targets