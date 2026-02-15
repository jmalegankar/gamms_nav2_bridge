import gamms
import gamms.osm
import random
import numpy as np
import strategy
import config

# === Setup === #

print("\n" + "="*60)
print("INITIALIZING FRONTIER SEARCH PURSUIT GAME")
print("="*60 + "\n")

# Create GAMMS context
ctx = gamms.create_context(vis_engine=config.VIS_ENGINE, logger_config={'level': 'ERROR'})

# Load OSM graph
print("[Setup] Loading OSM map...")


# Use gamms.osm directly
G = gamms.osm.graph_from_xml(
    config.MAP_FILE,
    resolution=5.0,      # meters between nodes on long edges
    bidirectional=True,   # make graph bidirectional
    retain_all=False,     # don't keep isolated nodes
    tolerance=8.25      # intersection consolidation tolerance
)

total_nodes = len(G.nodes)
total_edges = len(G.edges)

print(f"[Setup] Loaded graph: {total_nodes} nodes, {total_edges} edges")
    

# Attach to gamms - no conversion needed!
print("[Setup] Attaching graph to gamms...")
ctx.graph.attach_networkx_graph(G)

# Verify
all_nodes = list(ctx.graph.graph.get_nodes())
if all_nodes:
    sample = ctx.graph.graph.get_node(all_nodes[0])
    print(f"[Setup] Verified: sample node at ({sample.x:.1f}, {sample.y:.1f})")

ctx.visual.set_graph_visual(
    width=1920, 
    height=1080,
    node_size=2.0,
    edge_width=1.0,
    node_color=(80, 80, 80),
    edge_color=(120, 120, 120),
)

# Get all node IDs
all_nodes = list(ctx.graph.graph.get_nodes())

if not all_nodes:
    print("ERROR: No nodes in graph!")
    exit(1)

# Pick a random starting node for all agents
start_node = random.choice(all_nodes)
start_node_obj = ctx.graph.graph.get_node(start_node)
print(f"\n[Setup] All agents starting at node {start_node} at ({start_node_obj.x:.1f}, {start_node_obj.y:.1f})")

# Pick Mr. X start node (far from agents)
nodes_by_dist = []
for nid in all_nodes:
    node = ctx.graph.graph.get_node(nid)
    dist = np.sqrt((node.x - start_node_obj.x)**2 + (node.y - start_node_obj.y)**2)
    nodes_by_dist.append((nid, dist))

nodes_by_dist.sort(key=lambda x: x[1], reverse=True)

# Pick from top 25% farthest nodes
far_nodes = [nid for nid, _ in nodes_by_dist[:max(1, len(nodes_by_dist)//4)]]
mrx_start = random.choice(far_nodes)

mrx_node = ctx.graph.graph.get_node(mrx_start)
mrx_dist = np.sqrt((mrx_node.x - start_node_obj.x)**2 + (mrx_node.y - start_node_obj.y)**2)
print(f"[Setup] Mr. X starting at node {mrx_start} at ({mrx_node.x:.1f}, {mrx_node.y:.1f}) - {mrx_dist:.1f}m away")

# Create Mr. X
ctx.agent.create_agent(name='mr_x', start_node_id=mrx_start)
ctx.visual.set_agent_visual(name='mr_x', color=(128, 0, 128), size=6)
ctx.sensor.create_sensor(
    sensor_id='mrx_neighbor_sensor',
    sensor_type=gamms.sensor.SensorType.NEIGHBOR
)
ctx.agent.get_agent('mr_x').register_sensor('mrx_neighbor_sensor', ctx.sensor.get_sensor('mrx_neighbor_sensor'))


# Custom sensor to detect Mr. X
@ctx.sensor.custom(name="MRXDetection")
class MrXDetectionSensor(gamms.typing.ISensor):
    def __init__(self, ctx, sensor_id, sensor_range, fov, mrx_name):
        self.ctx = ctx
        self._sensor_id = sensor_id
        self._sensor_range = sensor_range
        self._fov = fov
        self._mrx_name = mrx_name
        self._owner = None
        self._data = {}
    
    @property
    def sensor_id(self):
        return self._sensor_id
    
    @property
    def type(self):
        return gamms.typing.SensorType.CUSTOM
    
    @property
    def data(self):
        return self._data
    
    def set_owner(self, owner):
        self._owner = owner
    
    def update(self, data):
        pass
    
    def sense(self, node_id):
        """Detect if Mr. X is within range and FOV"""
        if not self._owner:
            self._data = {}
            return
        
        agent = self.ctx.agent.get_agent(self._owner)
        mrx = self.ctx.agent.get_agent(self._mrx_name)
        
        agent_node = self.ctx.graph.graph.get_node(agent.current_node_id)
        mrx_node = self.ctx.graph.graph.get_node(mrx.current_node_id)
        
        # Calculate distance
        dx = mrx_node.x - agent_node.x
        dy = mrx_node.y - agent_node.y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > self._sensor_range:
            self._data = {}
            return
        
        # Calculate angle
        angle_to_mrx = np.arctan2(dy, dx)
        orientation = agent.orientation
        if orientation == (0.0, 0.0):
            orientation = (1.0, 0.0)
        
        agent_angle = np.arctan2(orientation[1], orientation[0])
        relative_angle = angle_to_mrx - agent_angle
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
        
        # Check FOV
        if abs(relative_angle) <= self._fov / 2:
            self._data = {
                'detected': True,
                'node_id': mrx.current_node_id,
                'distance': distance,
                'angle': relative_angle,
                'position': (mrx_node.x, mrx_node.y)
            }
        else:
            self._data = {}


# Create agents
print(f"\n[Setup] Creating {config.NUM_AGENTS} agents:")

agent_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 255)]
agent_names = []

for i in range(config.NUM_AGENTS):
    agent_name = f'agent_{i}'
    agent_names.append(agent_name)
    
    # All agents start at the same node
    ctx.agent.create_agent(name=agent_name, start_node_id=start_node)
    
    # Random orientation
    angle = random.uniform(0, 2 * np.pi)
    agent = ctx.agent.get_agent(agent_name)
    agent.orientation = (np.cos(angle), np.sin(angle))
    
    # Neighbor sensor
    ctx.sensor.create_sensor(
        sensor_id=f'neighbor_sensor_{i}',
        sensor_type=gamms.sensor.SensorType.NEIGHBOR
    )
    agent.register_sensor(f'neighbor_sensor_{i}', ctx.sensor.get_sensor(f'neighbor_sensor_{i}'))
    
    # Range sensor (map awareness)
    ctx.sensor.create_sensor(
        sensor_id=f'map_sensor_{i}',
        sensor_type=gamms.sensor.SensorType.RANGE,
        sensor_range=config.MAP_SENSOR_RANGE,
        fov=config.MAP_FOV
    )
    agent.register_sensor(f'map_sensor_{i}', ctx.sensor.get_sensor(f'map_sensor_{i}'))
    
    # Mr. X detection sensor
    mrx_sensor = MrXDetectionSensor(ctx, f'mrx_sensor_{i}', config.MRX_SENSOR_RANGE, config.MRX_FOV, 'mr_x')
    ctx.sensor.add_sensor(mrx_sensor)
    agent.register_sensor(f'mrx_sensor_{i}', mrx_sensor)
    
    # Visualization
    ctx.visual.set_agent_visual(name=agent_name, color=agent_colors[i % len(agent_colors)], size=6)
    print(f"  ✓ {agent_name}")

# Initialize strategy
strategies = strategy.map_strategy(ctx, agent_names, start_node)

for agent in ctx.agent.create_iter():
    if agent.name in strategies:
        agent.register_strategy(strategies[agent.name])


# === Main Game Loop === #
print("\n" + "="*60)
print("GAME START - FRONTIER SEARCH MODE")
print("="*60 + "\n")

step = 0
max_steps = 100000

while not ctx.is_terminated() and step < max_steps:
    step += 1
    
    # Agent actions
    for agent in ctx.agent.create_iter():
        state = agent.get_state()
        
        if agent.name == 'mr_x':
            # Mr. X moves randomly
            sensor_data = state['sensor']
            if 'mrx_neighbor_sensor' in sensor_data:
                _, neighbors = sensor_data['mrx_neighbor_sensor']
                
                current_node = agent.current_node_id
                valid_moves = [n for n in neighbors if n != current_node]
                
                if valid_moves:
                    state['action'] = random.choice(valid_moves)
                else:
                    state['action'] = agent.current_node_id
            else:
                state['action'] = agent.current_node_id
        else:
            # Agent strategy (frontier search or pursuit)
            if agent.strategy:
                agent.strategy(state)
            else:
                state['action'] = agent.current_node_id
    
    # Update states
    for agent in ctx.agent.create_iter():
        agent.set_state()
    
    # Visualization
    ctx.visual.simulate()
    
    # Check win condition (caught Mr. X)
    mrx = ctx.agent.get_agent('mr_x')
    mrx_node_obj = ctx.graph.graph.get_node(mrx.current_node_id)
    
    for agent in ctx.agent.create_iter():
        if agent.name == 'mr_x':
            continue
        agent_node_obj = ctx.graph.graph.get_node(agent.current_node_id)
        dist = np.sqrt((agent_node_obj.x - mrx_node_obj.x)**2 + (agent_node_obj.y - mrx_node_obj.y)**2)
        
        if dist <= config.CATCH_DISTANCE:
            print(f"\n{'='*60}")
            print(f"🎯 GAME OVER - {agent.name} caught Mr. X!")
            print(f"Steps: {step}")
            print(f"Explored: {len(strategy.visited_nodes)} nodes ({100*len(strategy.visited_nodes)/total_nodes:.1f}%)")
            print(f"{'='*60}\n")
            ctx.terminate()
            break
    
    # Progress updates
    if step % 100 == 0:
        explored_pct = 100 * len(strategy.visited_nodes) / total_nodes
        frontier_count = len(strategy.get_all_frontiers())
        print(f"Step {step}: Explored {len(strategy.visited_nodes)}/{total_nodes} nodes ({explored_pct:.1f}%) | "
              f"{frontier_count} frontiers" +
              (f" | Mr. X spotted at node {strategy.mrx_last_seen[0]}" if strategy.mrx_last_seen else " | Searching..."))

if step >= max_steps:
    print(f"\n{'='*60}")
    print(f"TIMEOUT - Max steps reached")
    print(f"{'='*60}\n")

ctx.terminate()