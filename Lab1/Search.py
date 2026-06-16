import time
import heapq
from collections import deque
from typing import Optional, List, Dict, Set


class SearchResult:
    def __init__(
        self,
        found: bool,
        path: Optional[List],
        steps: int,
        expanded: int,
        time_s: float,
        max_fringe: int = 0,
    ):
        self.found = found
        self.path = path or []
        self.steps = steps
        self.expanded = expanded
        self.time_s = time_s
        self.max_fringe = max_fringe


def ReconstructPath(came_from: Dict, goal_state):
    path = []
    cur = goal_state
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur, None)
    path.reverse()
    return path


def BreadthFirstSearch(start, goal, neighbors_fn):
    t0 = time.perf_counter()
    queue = deque([start])
    came_from = {start: None}
    visited: Set = {start}
    expanded = 0

    while queue:
        state = queue.popleft()
        expanded += 1
        if state == goal:
            path = ReconstructPath(came_from, state)
            return SearchResult(
                True, path, len(path) - 1, expanded, time.perf_counter() - t0
            )
        for nb in neighbors_fn(state):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = state
                queue.append(nb)
    return SearchResult(False, None, 0, expanded, time.perf_counter() - t0)


def DepthFirstSearch(start, goal, neighbors_fn):
    t0 = time.perf_counter()
    stack = [(start, 0, None)]
    came_from = {start: None}
    visited_depth = {start: 0}
    expanded = 0
    max_fringe = 1

    while stack:
        state, depth, parent = stack.pop()
        expanded += 1
        if state == goal:
            path = ReconstructPath(came_from, state)
            return SearchResult(
                True,
                path,
                len(path) - 1,
                expanded,
                time.perf_counter() - t0,
                max_fringe,
            )
        neighbors = list(neighbors_fn(state))
        for nb in reversed(neighbors):
            nd = depth + 1
            if nb in visited_depth and visited_depth[nb] <= nd:
                continue
            visited_depth[nb] = nd
            came_from[nb] = state
            stack.append((nb, nd, state))
        if len(stack) > max_fringe:
            max_fringe = len(stack)

    return SearchResult(False, None, 0, expanded, time.perf_counter() - t0, max_fringe)


def AStarSearch(start, goal, neighbors_fn, heuristic_fn):
    t0 = time.perf_counter()
    open_heap = []
    g_score = {start: 0}
    f_score = {start: heuristic_fn(start)}
    counter = 0
    heapq.heappush(open_heap, (f_score[start], g_score[start], counter, start))
    came_from = {start: None}
    closed: Set = set()
    expanded = 0

    while open_heap:
        _, g, _, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        expanded += 1
        if state == goal:
            path = ReconstructPath(came_from, state)
            return SearchResult(
                True, path, len(path) - 1, expanded, time.perf_counter() - t0
            )
        closed.add(state)

        for nb_item in neighbors_fn(state):
            if isinstance(nb_item, tuple) and len(nb_item) == 2:
                nb, cost = nb_item
            else:
                nb, cost = nb_item, 1

            tentative_g = g + cost
            if nb in closed and tentative_g >= g_score.get(nb, float("inf")):
                continue
            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb] = state
                g_score[nb] = tentative_g
                f = tentative_g + heuristic_fn(nb)
                counter += 1
                heapq.heappush(open_heap, (f, tentative_g, counter, nb))
    return SearchResult(False, None, 0, expanded, time.perf_counter() - t0)
