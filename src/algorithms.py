import collections
import heapq
import numpy as np

def get_neighbors(point, w, h, block_size, snake_body):
    neighbors = []
    directions = [(block_size, 0), (-block_size, 0), (0, block_size), (0, -block_size)]
    for dx, dy in directions:
        next_point = (point[0] + dx, point[1] + dy)
        if (0 <= next_point[0] < w and 
            0 <= next_point[1] < h and 
            next_point not in snake_body[1:]): # Allow head to move to tail
            neighbors.append(next_point)
    return neighbors

def a_star(start, target, w, h, block_size, snake_body):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - target[0]) + abs(start[1] - target[1])}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in get_neighbors(current, w, h, block_size, snake_body):
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + abs(neighbor[0] - target[0]) + abs(neighbor[1] - target[1])
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def is_dead_end(start, w, h, block_size, snake_body):
    visited = set()
    queue = collections.deque([start])
    visited.add(start)
    count = 0
    required_space = len(snake_body)
    
    while queue:
        current = queue.popleft()
        count += 1
        if count >= required_space:
            return False
        for neighbor in get_neighbors(current, w, h, block_size, snake_body):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return True

def get_hamiltonian_cycle(w, h, block_size):
    """
    Generates a simple Hamiltonian cycle for a grid (assuming w and h are even multiples of block_size).
    This is a basic zig-zag pattern that covers the entire grid.
    """
    rows = h // block_size
    cols = w // block_size
    cycle = []
    
    # Zig-zag pattern
    for i in range(rows):
        if i % 2 == 0:
            for j in range(cols):
                cycle.append((j * block_size, i * block_size))
        else:
            for j in range(cols - 1, -1, -1):
                cycle.append((j * block_size, i * block_size))
    
    # The cycle needs to connect back, but for simplicity in reward shaping, 
    # we just provide the ordered list of points.
    return cycle

def get_next_hamiltonian_step(current_point, cycle):
    """
    Returns the next point in the Hamiltonian cycle after the current point.
    """
    try:
        idx = cycle.index(current_point)
        return cycle[(idx + 1) % len(cycle)]
    except ValueError:
        return None
