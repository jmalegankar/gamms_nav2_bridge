import random
import numpy as np
from collections import deque

# ============================================================================
# SHARED KNOWLEDGE
# ============================================================================

global_nodes = {}  # {node_id: (x, y)}
global_edges = set()  # {(source, target)}
visited_nodes = set()  # Nodes any agent has visited
mrx_last_seen = None  # (node_id, step) or None
mrx_history = []  # [(node_id, step), ...] - track movement pattern
agent_territories = {}  # {agent_name: (center_x, center_y)}
agent_recent_nodes = {}  # {agent_name: [node_id, ...]} - last 5 nodes visited


# ============================================================================
# HELPERS
# ============================================================================

def get_moves(state, agent_name):
    """Get valid moves from neighbor sensor"""
    sensor_data = state['sensor']
    agent_idx = agent_name.split('_')[1]
    neighbor_key = f'neighbor_sensor_{agent_idx}'
    current_node = state['curr_pos']
    
    if neighbor_key not in sensor_data:
        return []
    
    _, neighbors = sensor_data[neighbor_key]
    if not neighbors:
        return []
    
    return [n for n in neighbors if n != current_node]


def distance_l2(pos1, pos2):
    """Euclidean distance"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def get_known_neighbors(node_id):
    """Get neighbors from global edges"""
    return [t for s, t in global_edges if s == node_id]


def bfs_path(start, goal):
    """BFS shortest path using known edges"""
    if start == goal:
        return [start]
    if start not in global_nodes or goal not in global_nodes:
        return None
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        neighbors = get_known_neighbors(current)
        
        for neighbor in neighbors:
            if neighbor == goal:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


# ============================================================================
# UPDATE KNOWLEDGE
# ============================================================================

def update_knowledge(state, agent_name, step):
    """Update shared knowledge from sensors"""
    global mrx_last_seen, mrx_history
    
    current_node = state['curr_pos']
    visited_nodes.add(current_node)
    
    # Track recent nodes for this agent (prevent oscillation)
    if agent_name not in agent_recent_nodes:
        agent_recent_nodes[agent_name] = []
    
    agent_recent_nodes[agent_name].append(current_node)
    
    # Keep only last 5 nodes
    if len(agent_recent_nodes[agent_name]) > 5:
        agent_recent_nodes[agent_name] = agent_recent_nodes[agent_name][-5:]
    
    sensor_data = state['sensor']
    agent_idx = agent_name.split('_')[1]
    
    # Update from map sensor
    map_key = f'map_sensor_{agent_idx}'
    if map_key in sensor_data:
        _, map_data = sensor_data[map_key]
        
        if 'nodes' in map_data and isinstance(map_data['nodes'], dict):
            for nid, node in map_data['nodes'].items():
                if nid not in global_nodes:
                    global_nodes[nid] = (node.x, node.y)
        
        if 'edges' in map_data:
            for edge in map_data['edges']:
                if hasattr(edge, 'source') and hasattr(edge, 'target'):
                    global_edges.add((edge.source, edge.target))
                    global_edges.add((edge.target, edge.source))
    
    # Check for Mr. X
    mrx_key = f'mrx_sensor_{agent_idx}'
    if mrx_key in sensor_data:
        _, mrx_data = sensor_data[mrx_key]
        if mrx_data and mrx_data.get('detected'):
            mrx_node = mrx_data['node_id']
            mrx_last_seen = (mrx_node, step)
            mrx_history.append((mrx_node, step))
            
            # Keep only recent history (last 10 sightings)
            if len(mrx_history) > 10:
                mrx_history = mrx_history[-10:]
            
            print(f"🎯 [{agent_name}] SPOTTED MR. X at node {mrx_node}!")


# ============================================================================
# TERRITORY ASSIGNMENT (Voronoi-like)
# ============================================================================

def assign_territories(agent_names, ctx):
    """Assign each agent to a territory based on map bounds"""
    if agent_territories:
        return  # Already assigned
    
    if not global_nodes:
        return  # Not enough info yet
    
    # Find map bounds
    xs = [pos[0] for pos in global_nodes.values()]
    ys = [pos[1] for pos in global_nodes.values()]
    
    if not xs or not ys:
        return
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Divide into quadrants
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    
    # Assign territories based on starting positions
    for name in agent_names:
        agent = ctx.agent.get_agent(name)
        node = agent.current_node_id
        
        if node in global_nodes:
            x, y = global_nodes[node]
            
            # Assign to closest quadrant center
            if x < mid_x and y < mid_y:
                agent_territories[name] = ((min_x + mid_x)/2, (min_y + mid_y)/2)  # SW
            elif x >= mid_x and y < mid_y:
                agent_territories[name] = ((mid_x + max_x)/2, (min_y + mid_y)/2)  # SE
            elif x < mid_x and y >= mid_y:
                agent_territories[name] = ((min_x + mid_x)/2, (mid_y + max_y)/2)  # NW
            else:
                agent_territories[name] = ((mid_x + max_x)/2, (mid_y + max_y)/2)  # NE


# ============================================================================
# SMARTER FRONTIER SELECTION
# ============================================================================

def get_frontier_score(node_id):
    """
    Score a node by how "frontier-like" it is.
    Higher score = more unexplored neighbors = better frontier
    """
    if node_id not in global_nodes:
        return 1000  # Unknown nodes are highest priority
    
    neighbors = get_known_neighbors(node_id)
    
    # Count unexplored neighbors
    unexplored_count = sum(1 for n in neighbors if n not in visited_nodes)
    
    # Count unknown neighbors (not even in global_nodes)
    unknown_count = sum(1 for n in neighbors if n not in global_nodes)
    
    # Score: unknown > unexplored > explored
    score = unknown_count * 10 + unexplored_count * 2
    
    return score


def explore(state, agent_name, ctx):
    """
    Smart exploration with frontier selection and anti-oscillation
    """
    current_node = state['curr_pos']
    moves = get_moves(state, agent_name)
    
    if not moves:
        return current_node
    
    # Assign territories if not done
    assign_territories([a.name for a in ctx.agent.create_iter() if a.name != 'mr_x'], ctx)
    
    # Get recent nodes for this agent
    recent_nodes = agent_recent_nodes.get(agent_name, [])
    
    # Score each move
    best_move = None
    best_score = -float('inf')
    
    for move in moves:
        score = 0
        
        # 1. Frontier score (most important)
        frontier_score = get_frontier_score(move)
        score += frontier_score * 10
        
        # 2. Penalize recently visited nodes (ANTI-OSCILLATION)
        if move in recent_nodes:
            # Heavy penalty based on how recently visited
            recency_index = len(recent_nodes) - recent_nodes.index(move)
            penalty = recency_index * 50  # More recent = bigger penalty
            score -= penalty
        
        # 3. Territory preference (stay in your region)
        if agent_name in agent_territories and move in global_nodes:
            territory_center = agent_territories[agent_name]
            move_pos = global_nodes[move]
            dist_to_territory = distance_l2(move_pos, territory_center)
            score -= dist_to_territory * 0.5
        
        # 4. Teammate repulsion (don't cluster)
        if move in global_nodes:
            move_pos = global_nodes[move]
            
            for other_agent in ctx.agent.create_iter():
                if other_agent.name == agent_name or other_agent.name == 'mr_x':
                    continue
                
                other_node = other_agent.current_node_id
                if other_node in global_nodes:
                    other_pos = global_nodes[other_node]
                    dist = distance_l2(move_pos, other_pos)
                    
                    if dist < 20:
                        score -= (20 - dist) * 2
        
        # 5. Random tie-breaker
        score += random.random() * 0.1
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move if best_move else random.choice(moves)


# ============================================================================
# PREDICTIVE PURSUIT
# ============================================================================

def predict_mrx_position():
    """
    Predict Mr. X's next position based on movement history.
    Returns predicted node_id or None.
    """
    if len(mrx_history) < 2:
        return None
    
    # Get last two positions
    pos1_node, time1 = mrx_history[-2]
    pos2_node, time2 = mrx_history[-1]
    
    if pos1_node not in global_nodes or pos2_node not in global_nodes:
        return None
    
    pos1 = global_nodes[pos1_node]
    pos2 = global_nodes[pos2_node]
    
    # Calculate movement vector
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    
    # Predict next position (extrapolate)
    predicted_x = pos2[0] + dx
    predicted_y = pos2[1] + dy
    
    # Find closest known node to predicted position
    best_node = None
    best_dist = float('inf')
    
    for node_id, pos in global_nodes.items():
        dist = distance_l2(pos, (predicted_x, predicted_y))
        if dist < best_dist:
            best_dist = dist
            best_node = node_id
    
    return best_node


def pursue(state, agent_name, ctx):
    """
    Smart pursuit with prediction and role assignment
    """
    current_node = state['curr_pos']
    moves = get_moves(state, agent_name)
    
    if not moves or not mrx_last_seen:
        return explore(state, agent_name, ctx)
    
    mrx_node, last_seen_step = mrx_last_seen
    
    # Try to predict Mr. X's next position
    predicted_node = predict_mrx_position()
    
    # Decide target: use prediction if available and recent
    target_node = predicted_node if predicted_node else mrx_node
    
    # Try BFS pathfinding
    path = bfs_path(current_node, target_node)
    
    if path and len(path) > 1:
        next_step = path[1]
        if next_step in moves:
            return next_step
    
    # No path - greedy movement toward target
    if current_node not in global_nodes or target_node not in global_nodes:
        return random.choice(moves)
    
    current_pos = global_nodes[current_node]
    target_pos = global_nodes[target_node]
    
    best_move = None
    best_dist = float('inf')
    
    for move in moves:
        if move in global_nodes:
            move_pos = global_nodes[move]
            dist = distance_l2(move_pos, target_pos)
            
            # Small random variation so agents approach from different angles
            dist += random.random() * 0.3
            
            if dist < best_dist:
                best_dist = dist
                best_move = move
    
    return best_move if best_move else random.choice(moves)


# ============================================================================
# MAIN STRATEGY
# ============================================================================

def agent_strategy(state, agent_name, ctx, step):
    """Main strategy: explore with frontier selection → predictive pursuit"""
    global mrx_last_seen
    update_knowledge(state, agent_name, step)
    
    if mrx_last_seen is None:
        next_node = explore(state, agent_name, ctx)
    else:
        mrx_node, last_seen_step = mrx_last_seen
        
        # Give up if sighting too old
        if step - last_seen_step > 50:
            mrx_last_seen = None
            next_node = explore(state, agent_name, ctx)
        else:
            next_node = pursue(state, agent_name, ctx)
    
    state['action'] = next_node


# ============================================================================
# STRATEGY MAPPER
# ============================================================================

def map_strategy(ctx, agent_names, total_nodes, total_edges):
    """Create strategy functions for each agent"""
    
    step_state = {'count': 0, 'last': None}
    last_agent = agent_names[-1] if agent_names else None
    
    def make_wrapper(name):
        def wrapper(state):
            if step_state['last'] == last_agent or step_state['last'] is None:
                step_state['count'] += 1
            step_state['last'] = name
            
            agent_strategy(state, name, ctx, step_state['count'])
        
        return wrapper
    
    print(f"\n✓ IMPROVED Strategy initialized")
    print(f"  • Smarter frontier selection (unexplored neighbors)")
    print(f"  • Predictive pursuit (estimate Mr. X movement)")
    print(f"  • Voronoi territories (agents spread to regions)")
    print(f"  • Map: {total_nodes} nodes, {total_edges} edges\n")
    
    return {name: make_wrapper(name) for name in agent_names}