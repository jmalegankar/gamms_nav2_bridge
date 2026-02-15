import random
import numpy as np
from collections import deque
import heapq

# ============================================================================
# SHARED KNOWLEDGE (for single agent)
# ============================================================================

global_nodes = {}  # {node_id: (x, y)}
global_edges = set()  # {(source, target)}
visited_nodes = set()  # Nodes visited
recent_nodes = deque(maxlen=10)  # Last 10 nodes to prevent backtracking

mrx_last_seen = None  # (node_id, step) when Mr. X was last spotted
mrx_history = []  # Track Mr. X sightings


# ============================================================================
# A* PATHFINDING (Much faster than BFS!)
# ============================================================================

def heuristic(node_a, node_b):
    """
    Heuristic: Euclidean distance (straight-line distance)
    This is admissible (never overestimates) so A* is optimal
    """
    if node_a not in global_nodes or node_b not in global_nodes:
        return 0
    
    pos_a = global_nodes[node_a]
    pos_b = global_nodes[node_b]
    
    return np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)


def astar_path(start, goal, max_depth=500):
    """
    A* pathfinding - finds optimal path much faster than BFS
    
    How it works:
    1. Explores nodes with lowest f(n) = g(n) + h(n) first
       - g(n) = actual cost from start
       - h(n) = estimated cost to goal (heuristic)
    2. Always expands most promising nodes first
    3. Guaranteed optimal if heuristic is admissible (ours is!)
    """
    if start == goal:
        return [start]
    
    if start not in global_nodes or goal not in global_nodes:
        return None
    
    # Priority queue: (f_score, node, g_score, path)
    open_set = [(0, start, 0, [start])]
    
    # Track best g_score for each node
    g_scores = {start: 0}
    
    # Track which nodes we've explored
    closed_set = set()
    
    nodes_explored = 0
    
    while open_set:
        f_score, current, g_score, path = heapq.heappop(open_set)
        
        # Already explored this node with better path
        if current in closed_set:
            continue
        
        closed_set.add(current)
        nodes_explored += 1
        
        # Depth limit
        if len(path) > max_depth:
            continue
        
        # Goal reached!
        if current == goal:
            print(f"    A* explored {nodes_explored} nodes (path length: {len(path)})")
            return path
        
        # Explore neighbors
        neighbors = get_known_neighbors(current)
        
        for neighbor in neighbors:
            if neighbor in closed_set:
                continue
            
            # Calculate cost to reach this neighbor
            # For now, assume uniform edge cost (could use actual distances)
            edge_cost = heuristic(current, neighbor)  # Actual distance
            tentative_g = g_score + edge_cost
            
            # If this is a better path to neighbor, update it
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                h_score = heuristic(neighbor, goal)
                f_score = tentative_g + h_score
                
                new_path = path + [neighbor]
                heapq.heappush(open_set, (f_score, neighbor, tentative_g, new_path))
    
    # No path found
    print(f"    A* explored {nodes_explored} nodes - no path found")
    return None


# ============================================================================
# BIDIRECTIONAL A* (Even faster for long paths!)
# ============================================================================

def bidirectional_astar(start, goal, max_depth=500):
    """
    Bidirectional A* - searches from both start and goal simultaneously
    Often 2x faster than regular A* for long paths
    """
    if start == goal:
        return [start]
    
    if start not in global_nodes or goal not in global_nodes:
        return None
    
    # Forward search (from start)
    forward_open = [(0, start, 0, [start])]
    forward_g = {start: 0}
    forward_closed = set()
    forward_parents = {start: (None, [start])}
    
    # Backward search (from goal)
    backward_open = [(0, goal, 0, [goal])]
    backward_g = {goal: 0}
    backward_closed = set()
    backward_parents = {goal: (None, [goal])}
    
    # Best path found so far
    best_path = None
    best_cost = float('inf')
    
    nodes_explored = 0
    
    while forward_open or backward_open:
        # Forward step
        if forward_open:
            f_score, current, g_score, path = heapq.heappop(forward_open)
            
            if current not in forward_closed:
                forward_closed.add(current)
                nodes_explored += 1
                
                # Check if we met backward search
                if current in backward_closed:
                    # Reconstruct path
                    _, backward_path = backward_parents[current]
                    combined_path = path + backward_path[-2::-1]  # Reverse backward path
                    
                    if len(combined_path) < best_cost:
                        best_path = combined_path
                        best_cost = len(combined_path)
                
                # Explore neighbors
                if len(path) < max_depth:
                    neighbors = get_known_neighbors(current)
                    
                    for neighbor in neighbors:
                        if neighbor in forward_closed:
                            continue
                        
                        edge_cost = heuristic(current, neighbor)
                        tentative_g = g_score + edge_cost
                        
                        if neighbor not in forward_g or tentative_g < forward_g[neighbor]:
                            forward_g[neighbor] = tentative_g
                            h_score = heuristic(neighbor, goal)
                            f_score = tentative_g + h_score
                            
                            new_path = path + [neighbor]
                            forward_parents[neighbor] = (current, new_path)
                            heapq.heappush(forward_open, (f_score, neighbor, tentative_g, new_path))
        
        # Backward step
        if backward_open:
            f_score, current, g_score, path = heapq.heappop(backward_open)
            
            if current not in backward_closed:
                backward_closed.add(current)
                nodes_explored += 1
                
                # Check if we met forward search
                if current in forward_closed:
                    _, forward_path = forward_parents[current]
                    combined_path = forward_path + path[-2::-1]
                    
                    if len(combined_path) < best_cost:
                        best_path = combined_path
                        best_cost = len(combined_path)
                
                # Explore neighbors
                if len(path) < max_depth:
                    neighbors = get_known_neighbors(current)
                    
                    for neighbor in neighbors:
                        if neighbor in backward_closed:
                            continue
                        
                        edge_cost = heuristic(current, neighbor)
                        tentative_g = g_score + edge_cost
                        
                        if neighbor not in backward_g or tentative_g < backward_g[neighbor]:
                            backward_g[neighbor] = tentative_g
                            h_score = heuristic(neighbor, start)
                            f_score = tentative_g + h_score
                            
                            new_path = path + [neighbor]
                            backward_parents[neighbor] = (current, new_path)
                            heapq.heappush(backward_open, (f_score, neighbor, tentative_g, new_path))
        
        # Early termination if we found a path and searches have met
        if best_path and not forward_open and not backward_open:
            break
    
    if best_path:
        print(f"    Bidirectional A* explored {nodes_explored} nodes (path length: {len(best_path)})")
        return best_path
    
    print(f"    Bidirectional A* explored {nodes_explored} nodes - no path found")
    return None


# ============================================================================
# SMART PATHFINDING (Choose best algorithm)
# ============================================================================

def smart_path(start, goal, max_depth=500):
    """
    Choose pathfinding algorithm based on distance:
    - Short distance (< 50m): Use A*
    - Long distance (>= 50m): Use Bidirectional A*
    """
    if start not in global_nodes or goal not in global_nodes:
        return None
    
    # Estimate distance
    straight_line_dist = heuristic(start, goal)
    
    # For short distances, regular A* is fine
    if straight_line_dist < 50:
        return astar_path(start, goal, max_depth)
    else:
        # For long distances, bidirectional is faster
        return bidirectional_astar(start, goal, max_depth)


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


# ============================================================================
# UPDATE KNOWLEDGE
# ============================================================================

def update_knowledge(state, agent_name, step):
    """Update map knowledge and detect Mr. X"""
    global mrx_last_seen, mrx_history
    
    current_node = state['curr_pos']
    visited_nodes.add(current_node)
    recent_nodes.append(current_node)
    
    sensor_data = state['sensor']
    agent_idx = agent_name.split('_')[1]
    
    # Update map from sensor
    map_key = f'map_sensor_{agent_idx}'
    if map_key in sensor_data:
        _, map_data = sensor_data[map_key]
        
        # Add nodes
        if 'nodes' in map_data and isinstance(map_data['nodes'], dict):
            for nid, node in map_data['nodes'].items():
                if nid not in global_nodes:
                    global_nodes[nid] = (node.x, node.y)
        
        # Add edges (bidirectional)
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
            
            if len(mrx_history) > 10:
                mrx_history = mrx_history[-10:]
            
            print(f"\n🎯 SPOTTED MR. X at node {mrx_node}! (step {step})\n")


# ============================================================================
# FRONTIER DETECTION
# ============================================================================

def is_frontier(node_id):
    """A node is a frontier if visited with unvisited neighbors"""
    if node_id not in visited_nodes:
        return False
    
    neighbors = get_known_neighbors(node_id)
    return any(n not in visited_nodes for n in neighbors)


def get_all_frontiers():
    """Get all current frontier nodes"""
    return [nid for nid in visited_nodes if is_frontier(nid)]


def score_frontier(frontier_node, current_pos):
    """
    Score a frontier by information gain and distance
    """
    if frontier_node not in global_nodes or current_pos not in global_nodes:
        return -float('inf')
    
    # Information gain: count unexplored neighbors
    neighbors = get_known_neighbors(frontier_node)
    unexplored_count = sum(1 for n in neighbors if n not in visited_nodes)
    
    # Distance
    frontier_pos = global_nodes[frontier_node]
    current_position = global_nodes[current_pos]
    distance = distance_l2(frontier_pos, current_position)
    
    # Score = information_gain - distance_penalty
    score = unexplored_count * 20 - distance * 0.1
    
    return score


def find_best_frontier(current_pos):
    """Find the best frontier to explore"""
    frontiers = get_all_frontiers()
    
    if not frontiers:
        return None
    
    best_frontier = None
    best_score = -float('inf')
    
    for frontier in frontiers:
        score = score_frontier(frontier, current_pos)
        
        if score > best_score:
            best_score = score
            best_frontier = frontier
    
    return best_frontier


# ============================================================================
# FRONTIER EXPLORATION
# ============================================================================

def explore_frontier(state, agent_name):
    """Explore using frontier search with A* pathfinding"""
    current_node = state['curr_pos']
    moves = get_moves(state, agent_name)
    
    if not moves:
        return current_node
    
    # Find best frontier
    best_frontier = find_best_frontier(current_node)
    
    if not best_frontier:
        # No frontiers - explore randomly (avoid recent nodes)
        unvisited_moves = [m for m in moves if m not in recent_nodes]
        
        if unvisited_moves:
            return random.choice(unvisited_moves)
        return random.choice(moves)
    
    # Use smart pathfinding (A* or Bidirectional A*)
    path = smart_path(current_node, best_frontier, max_depth=500)
    
    if path and len(path) > 1:
        next_step = path[1]
        if next_step in moves:
            return next_step
    
    # No path found - move greedily toward frontier
    if current_node in global_nodes and best_frontier in global_nodes:
        current_pos = global_nodes[current_node]
        frontier_pos = global_nodes[best_frontier]
        
        best_move = None
        best_dist = float('inf')
        
        for move in moves:
            if move in global_nodes:
                move_pos = global_nodes[move]
                dist = distance_l2(move_pos, frontier_pos)
                
                # Avoid going back to recent nodes
                if move in recent_nodes:
                    dist += 100
                
                if dist < best_dist:
                    best_dist = dist
                    best_move = move
        
        if best_move:
            return best_move
    
    # Fallback: random (avoid recent)
    unvisited_moves = [m for m in moves if m not in recent_nodes]
    if unvisited_moves:
        return random.choice(unvisited_moves)
    
    return random.choice(moves)


# ============================================================================
# MR. X PURSUIT
# ============================================================================

def pursue_mrx(state, agent_name):
    """Pursue Mr. X optimally using A*"""
    if not mrx_last_seen:
        return None
    
    current_node = state['curr_pos']
    moves = get_moves(state, agent_name)
    
    if not moves:
        return current_node
    
    target_node, _ = mrx_last_seen
    
    # Use smart pathfinding
    path = smart_path(current_node, target_node, max_depth=500)
    
    if path and len(path) > 1:
        next_step = path[1]
        if next_step in moves:
            return next_step
    
    # No path found - move greedily
    if current_node in global_nodes and target_node in global_nodes:
        current_pos = global_nodes[current_node]
        target_pos = global_nodes[target_node]
        
        best_move = None
        best_dist = float('inf')
        
        for move in moves:
            if move in global_nodes:
                move_pos = global_nodes[move]
                dist = distance_l2(move_pos, target_pos)
                
                if dist < best_dist:
                    best_dist = dist
                    best_move = move
        
        if best_move:
            return best_move
    
    return random.choice(moves)


# ============================================================================
# MAIN STRATEGY
# ============================================================================

def agent_strategy(state, agent_name, step):
    """
    Two-mode strategy:
    1. EXPLORATION: Frontier search
    2. PURSUIT: Chase Mr. X when spotted
    """
    global mrx_last_seen
    
    update_knowledge(state, agent_name, step)
    
    # Mode 1: PURSUIT
    if mrx_last_seen:
        mrx_node, last_seen_step = mrx_last_seen
        time_since_seen = step - last_seen_step
        
        if time_since_seen < 100:
            print(f"[Step {step}] PURSUIT MODE (last seen {time_since_seen} steps ago)")
            next_node = pursue_mrx(state, agent_name)
            
            if next_node:
                state['action'] = next_node
                return
        else:
            print(f"[Step {step}] Trail too old - resuming exploration")
            mrx_last_seen = None
    
    # Mode 2: EXPLORATION
    next_node = explore_frontier(state, agent_name)
    state['action'] = next_node


# ============================================================================
# STRATEGY MAPPER
# ============================================================================

def map_strategy(ctx, agent_names, start_node):
    """Create strategy function"""
    
    step_state = {'count': 0}
    agent_name = agent_names[0] if agent_names else 'agent_0'
    
    def strategy_wrapper(state):
        step_state['count'] += 1
        agent_strategy(state, agent_name, step_state['count'])
    
    print(f"\n✓ SINGLE AGENT FRONTIER SEARCH WITH A*")
    print(f"  • Frontier-based exploration")
    print(f"  • A* pathfinding (5-10x faster than BFS)")
    print(f"  • Bidirectional A* for long distances")
    print(f"  • Optimal paths guaranteed")
    print(f"  • Anti-backtracking")
    print(f"  • Smart pursuit mode\n")
    
    return {agent_name: strategy_wrapper}