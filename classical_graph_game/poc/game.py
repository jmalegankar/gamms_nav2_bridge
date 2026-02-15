import gamms
import cbor2
import struct
import random
import numpy as np
import strategy
import config

# === Utility Functions === #

# Load BSON graph file into GAMMS context
def load_bson_graph(ctx, map_file):
    """Load a BSON format graph directly into gamms, ensuring bidirectional edges"""
    graph = ctx.graph.graph
    
    with open(map_file, 'rb') as fp:
        # Read width and height
        width = struct.unpack('f', fp.read(4))[0]
        height = struct.unpack('f', fp.read(4))[0]
        
        print(f"Loading map with dimensions: {width:.1f} x {height:.1f}")
        
        nodes_loaded = 0
        edges_loaded = 0
        edges_to_add = []  # Collect edges first
        
        # Read CBOR-encoded nodes and edges
        while True:
            try:
                optype, item = cbor2.load(fp)
                
                if optype == 0:  # Node
                    graph.add_node(item)
                    nodes_loaded += 1
                elif optype == 1:  # Edge
                    edges_to_add.append(item)
                else:
                    raise ValueError(f"Unknown operation type: {optype}")
                    
            except EOFError:
                break
        
        # Add edges bidirectionally
        print(f"Loaded {nodes_loaded} nodes, adding {len(edges_to_add)} edges bidirectionally...")
        
        for edge_data in edges_to_add:
            source = edge_data['source']
            target = edge_data['target']
            edge_id = edge_data['id']
            
            # Add forward edge
            graph.add_edge(edge_data)
            edges_loaded += 1
            
            # Add reverse edge (with new ID)
            reverse_edge = {
                'id': edge_id + 100000,  # Offset to avoid ID collision
                'source': target,
                'target': source,
                **{k: v for k, v in edge_data.items() if k not in ['id', 'source', 'target']}
            }
            graph.add_edge(reverse_edge)
            edges_loaded += 1
        
        print(f"Total: {nodes_loaded} nodes and {edges_loaded} edges (bidirectional)")
    
    return nodes_loaded, edges_loaded

# Get clustered start nodes for agents
def get_clustered_start_nodes(ctx, num_agents, cluster_radius=25.0):
    """Get nodes in a cluster for agent starting positions"""
    graph = ctx.graph.graph
    
    # Get all nodes
    all_node_ids = list(graph.get_nodes())
    if len(all_node_ids) < num_agents:
        raise ValueError(f"Not enough nodes for {num_agents} agents")
    
    # Pick a random center node
    center_node_id = random.choice(all_node_ids)
    center_node = graph.get_node(center_node_id)
    center_pos = (center_node.x, center_node.y)
    
    print(f"  Cluster center: node {center_node_id} at ({center_pos[0]:.1f}, {center_pos[1]:.1f})")
    
    # Find all nodes within cluster_radius
    nearby_nodes = []
    for nid in all_node_ids:
        node = graph.get_node(nid)
        dist = np.sqrt((node.x - center_pos[0])**2 + (node.y - center_pos[1])**2)
        if dist <= cluster_radius:
            nearby_nodes.append(nid)
    
    if len(nearby_nodes) < num_agents:
        print(f"  Warning: Only {len(nearby_nodes)} nodes in cluster, using all available")
        # Expand radius if needed
        cluster_radius *= 1.5
        nearby_nodes = []
        for nid in all_node_ids:
            node = graph.get_node(nid)
            dist = np.sqrt((node.x - center_pos[0])**2 + (node.y - center_pos[1])**2)
            if dist <= cluster_radius:
                nearby_nodes.append(nid)
    
    # Select num_agents nodes from cluster
    selected = random.sample(nearby_nodes, min(num_agents, len(nearby_nodes)))
    
    for i, nid in enumerate(selected):
        node = graph.get_node(nid)
        dist = np.sqrt((node.x - center_pos[0])**2 + (node.y - center_pos[1])**2)
        print(f"  Agent {i}: node {nid} (distance from center: {dist:.1f}m)")
    
    return selected

# Check win condition
def check_win_condition(ctx, total_nodes, total_edges):
    """Check if game is won (caught Mr. X or fully explored)"""
    mrx = ctx.agent.get_agent('mr_x')
    mrx_node = ctx.graph.graph.get_node(mrx.current_node_id)
    
    # Check if any agent caught Mr. X
    for agent in ctx.agent.create_iter():
        if agent.name == 'mr_x':
            continue
        agent_node = ctx.graph.graph.get_node(agent.current_node_id)
        dist = np.sqrt((agent_node.x - mrx_node.x)**2 + (agent_node.y - mrx_node.y)**2)
        if dist <= config.CATCH_DISTANCE:
            return True, f"Agent {agent.name} caught Mr. X!"
    
    # Check if map fully explored
    if len(strategy.global_nodes) >= total_nodes and len(strategy.global_edges) >= total_edges:
        return True, "Map fully explored!"
    
    return False, None


# === Setup === #

# Create GAMMS context
ctx = gamms.create_context(vis_engine=config.VIS_ENGINE, logger_config={'level': 'ERROR'})
      
# Load graph from BSON file and set visual parameters
total_nodes, total_edges = load_bson_graph(ctx, config.MAP_FILE)
ctx.visual.set_graph_visual(
    width=1920, 
    height=1080,
    node_size=1.0,
    edge_width=1.0,
    node_color=(80, 80, 80),
    edge_color=(120, 120, 120),
)


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


# Get clustered start nodes for agents
agent_start_nodes = get_clustered_start_nodes(ctx, config.NUM_AGENTS, cluster_radius=25.0)

# Create Mr. X at random node 
all_nodes = list(ctx.graph.graph.get_nodes())

# Try to place Mr. X far from agents
agent_center_pos = (0, 0)
if agent_start_nodes:
    positions = [ctx.graph.graph.get_node(nid) for nid in agent_start_nodes]
    agent_center_pos = (
        sum(n.x for n in positions) / len(positions),
        sum(n.y for n in positions) / len(positions)
    )

# Sort nodes by distance from agent cluster
nodes_by_dist = []
for nid in all_nodes:
    node = ctx.graph.graph.get_node(nid)
    dist = np.sqrt((node.x - agent_center_pos[0])**2 + (node.y - agent_center_pos[1])**2)
    nodes_by_dist.append((nid, dist))

nodes_by_dist.sort(key=lambda x: x[1], reverse=True)

# Pick from top 25% farthest nodes
far_nodes = [nid for nid, _ in nodes_by_dist[:len(nodes_by_dist)//4]]
mrx_start = random.choice(far_nodes) if far_nodes else random.choice(all_nodes)

mrx_node = ctx.graph.graph.get_node(mrx_start)
mrx_dist = np.sqrt((mrx_node.x - agent_center_pos[0])**2 + (mrx_node.y - agent_center_pos[1])**2)
print(f"\n[Setup] Mr. X starting at node {mrx_start} (distance from agents: {mrx_dist:.1f}m)")

ctx.agent.create_agent(name='mr_x', start_node_id=mrx_start)
ctx.visual.set_agent_visual(name='mr_x', color=(128, 0, 128), size=2)
ctx.sensor.create_sensor(
        sensor_id='mrx_neighbor_sensor',
        sensor_type=gamms.sensor.SensorType.NEIGHBOR
    )
ctx.agent.get_agent('mr_x').register_sensor('mrx_neighbor_sensor', ctx.sensor.get_sensor('mrx_neighbor_sensor'))
    

print(f"\n[Setup] Creating {config.NUM_AGENTS} agents:")

agent_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]
agent_names = []

for i in range(config.NUM_AGENTS):
    agent_name = f'agent_{i}'
    agent_names.append(agent_name)
    
    ctx.agent.create_agent(name=agent_name, start_node_id=agent_start_nodes[i])
    
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
    
    # Range sensor
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
    ctx.visual.set_agent_visual(name=agent_name, color=agent_colors[i], size=2)
    print(f"  {agent_name} at node {agent_start_nodes[i]}")

strategies = strategy.map_strategy(ctx, agent_names, total_nodes, total_edges)
    
for agent in ctx.agent.create_iter():
    if agent.name in strategies:
        agent.register_strategy(strategies[agent.name])


    


# === Main Game Loop === #
print("\n" + "="*60)
print("GAME START")
print("="*60 + "\n")
step = 0
max_steps = 100000

while not ctx.is_terminated() and step < max_steps:
    step += 1
    
    # Agent actions
    for agent in ctx.agent.create_iter():
        state = agent.get_state()
        
        if agent.name == 'mr_x':
            # Mr. X moves randomly to a valid neighbor
            sensor_data = state['sensor']
            if 'mrx_neighbor_sensor' in sensor_data:
                _, neighbors = sensor_data['mrx_neighbor_sensor']
                
                current_node = agent.current_node_id
                valid_moves = [n for n in neighbors if n != current_node]
                
                if valid_moves:
                    state['action'] = random.choice(valid_moves)
                else:
                    # Only stay put if there are no other options
                    state['action'] = agent.current_node_id
            else:
                state['action'] = agent.current_node_id
        else:
            # Agent strategy
            if agent.strategy:
                agent.strategy(state)
            else:
                state['action'] = agent.current_node_id
    
    # Update states
    for agent in ctx.agent.create_iter():
        agent.set_state()
    
    # Visualization
    ctx.visual.simulate()
    
    # Check win condition
    won, message = check_win_condition(ctx, total_nodes, total_edges)
    if won:
        print(f"\n{'='*60}")
        print(f"GAME OVER - {message}")
        print(f"Steps: {step}")
        print(f"Explored: {len(strategy.global_nodes)}/{total_nodes} nodes, "
                f"{len(strategy.global_edges)}/{total_edges} edges")
        print(f"{'='*60}\n")
        break
    
    # Progress updates
    if step % 100 == 0:
        print(f"Step {step}: Explored {len(strategy.global_nodes)}/{total_nodes} nodes, "
                f"{len(strategy.global_edges)}/{total_edges} edges" +
                (f" | Mr. X at node {strategy.mrx_last_seen[0]}" if strategy.mrx_last_seen else ""))

if step >= max_steps:
    print(f"\n{'='*60}")
    print(f"TIMEOUT - Max steps reached")
    print(f"{'='*60}\n")

ctx.terminate()

# while ctx.is_terminated() is False:
#     ctx.visual.simulate()