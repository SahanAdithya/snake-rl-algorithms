import collections
import heapq

def get_neighbors(point, w, h, block_size, snake_body):
    neighbors = []
    # Right, Left, Down, Up
    directions = [(block_size, 0), (-block_size, 0), (0, block_size), (0, -block_size)]
    for dx, dy in directions:
        next_point = (point[0] + dx, point[1] + dy)
        if (0 <= next_point[0] < w and 
            0 <= next_point[1] < h and 
            next_point not in snake_body[1:]): # Allow head to move to tail if it's moving
            neighbors.append(next_point)
    return neighbors

def a_star(start, target, w, h, block_size, snake_body):
    """
    Returns a list of points representing the shortest path from start to target.
    """
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
    
    return None # No path found

def is_dead_end(start, w, h, block_size, snake_body):
    """
    Uses BFS to check if the snake is in a region it cannot escape from.
    Essentially checks the size of the reachable area.
    """
    visited = set()
    queue = collections.deque([start])
    visited.add(start)
    count = 0
    
    # We only care if the reachable space is less than the snake length
    # but for a quick check, let's just use a threshold or the snake size.
    max_cells = (w // block_size) * (h // block_size)
    required_space = len(snake_body)
    
    while queue:
        current = queue.popleft()
        count += 1
        if count >= required_space:
            return False # Enough space to survive for now

        for neighbor in get_neighbors(current, w, h, block_size, snake_body):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return True # Reachable space is too small
