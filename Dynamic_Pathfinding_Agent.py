import pygame
import heapq          # for the priority queue (min-heap)
import math           # for Euclidean distance
import random         # for random obstacle spawning
import time           # for measuring execution time

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (R, G, B)
# ─────────────────────────────────────────────
WHITE        = (255, 255, 255)
BLACK        = (15,  15,  15)
GRAY         = (45,  45,  55)
LIGHT_GRAY   = (80,  80,  95)
GRID_LINE    = (35,  35,  45)

# Cell colours
WALL_COLOR   = (30,  30,  38)   # dark walls
START_COLOR  = (52, 211, 153)   # green  – start node
GOAL_COLOR   = (251, 99,  64)   # orange – goal node
FRONTIER_COLOR = (250, 204,  21) # yellow – open / frontier
VISITED_COLOR  = (96, 165, 250)  # blue   – expanded / visited
PATH_COLOR     = (134, 239, 172) # bright green – final path
OBSTACLE_COLOR = (55,  55,  68)  # slightly lighter than wall

# UI panel colours
PANEL_BG     = (20,  20,  28)
BTN_ACTIVE   = (52, 211, 153)
BTN_INACTIVE = (45,  45,  58)
BTN_TEXT_ACT = (15,  15,  20)
BTN_TEXT_IN  = (180, 180, 200)
TEXT_COLOR   = (220, 220, 235)
ACCENT       = (52, 211, 153)

# ─────────────────────────────────────────────
#  LAYOUT CONSTANTS
# ─────────────────────────────────────────────
CELL_SIZE    = 28          # pixels per grid cell
PANEL_WIDTH  = 280         # width of the right-side control panel
FPS          = 60          # frames per second
VIZ_DELAY    = 0.018       # seconds between each visualisation step

# ─────────────────────────────────────────────
#  GRID CLASS  –  stores the map state
# ─────────────────────────────────────────────
class Grid:
    """
    Represents the 2-D grid world.
    Each cell can be: empty (0), wall (1), start (S), or goal (G).
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # 0 = free, 1 = wall
        self.cells = [[0] * cols for _ in range(rows)]
        self.start = (0, 0)
        self.goal  = (rows - 1, cols - 1)

    def in_bounds(self, r, c):
        """Return True if (r, c) is inside the grid."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, r, c):
        """Return True if cell (r, c) is not a wall."""
        return self.in_bounds(r, c) and self.cells[r][c] == 0

    def toggle_wall(self, r, c):
        """Flip a cell between wall and free (used by interactive editor)."""
        # Don't allow walling the start or goal
        if (r, c) == self.start or (r, c) == self.goal:
            return
        self.cells[r][c] = 1 - self.cells[r][c]

    def set_wall(self, r, c, value=1):
        """Explicitly set a cell as wall (1) or free (0)."""
        if (r, c) != self.start and (r, c) != self.goal:
            self.cells[r][c] = value

    def generate_random_maze(self, density=0.30):
        """
        Randomly place walls with the given density (0.0 – 1.0).
        Start and Goal cells are always kept free.
        """
        self.cells = [[0] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) != self.start and (r, c) != self.goal:
                    if random.random() < density:
                        self.cells[r][c] = 1

    def get_neighbours(self, r, c):
        """
        Return the 4 cardinal neighbours (up, down, left, right)
        that are inside the grid and not walls.
        Each move costs 1.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbours = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self.is_walkable(nr, nc):
                neighbours.append((nr, nc))
        return neighbours

    def spawn_dynamic_obstacle(self, current_path):
        """
        Spawn one random obstacle anywhere that isn't start, goal,
        or part of the current path.
        Returns the spawned cell or None.
        """
        path_set = set(current_path) if current_path else set()
        free_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (self.cells[r][c] == 0 and
                        (r, c) != self.start and
                        (r, c) != self.goal and
                        (r, c) not in path_set):
                    free_cells.append((r, c))
        if free_cells:
            chosen = random.choice(free_cells)
            self.cells[chosen[0]][chosen[1]] = 1
            return chosen
        return None


# ─────────────────────────────────────────────
#  HEURISTIC FUNCTIONS
# ─────────────────────────────────────────────
def manhattan_distance(a, b):
    """
    Manhattan Distance: |r1-r2| + |c1-c2|
    Admissible for 4-directional grid movement (cost = 1 per step).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a, b):
    """
    Euclidean Distance: sqrt((r1-r2)^2 + (c1-c2)^2)
    Admissible for any grid movement; slightly underestimates diagonal cost.
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# ─────────────────────────────────────────────
#  SEARCH ALGORITHMS
# ─────────────────────────────────────────────

def greedy_bfs(grid, heuristic_fn, visualise_cb=None):
    """
    Greedy Best-First Search
    ─────────────────────────
    f(n) = h(n)  — only the heuristic, ignores actual path cost.
    Uses a strict VISITED list: once a node is added to the
    frontier (seen), it is never re-inserted.
    This makes it fast but NOT optimal.

    Parameters:
        grid         – Grid object
        heuristic_fn – function(a, b) -> float
        visualise_cb – optional callback(frontier_set, visited_set)
                       called after each expansion for animation

    Returns:
        (path, nodes_expanded, path_cost)
        path is a list of (r,c) tuples from start to goal,
        or [] if no path exists.
    """
    start  = grid.start
    goal   = grid.goal

    # Priority queue entries: (h_value, node, parent_path)
    # We store the entire path so we can return it easily.
    open_heap = []
    heapq.heappush(open_heap, (heuristic_fn(start, goal), start, [start]))

    visited = set()   # strict visited list – never re-open
    visited.add(start)

    nodes_expanded = 0

    while open_heap:
        h_val, current, path = heapq.heappop(open_heap)

        nodes_expanded += 1

        # ── Visualisation callback ──
        if visualise_cb:
            frontier_set = {item[1] for item in open_heap}
            visualise_cb(frontier_set, set(path[:-1]))

        # ── Goal check ──
        if current == goal:
            cost = len(path) - 1   # each step costs 1
            return path, nodes_expanded, cost

        # ── Expand neighbours ──
        for neighbour in grid.get_neighbours(*current):
            if neighbour not in visited:
                visited.add(neighbour)
                new_path = path + [neighbour]
                h = heuristic_fn(neighbour, goal)
                heapq.heappush(open_heap, (h, neighbour, new_path))

    # No path found
    return [], nodes_expanded, 0


def astar_search(grid, heuristic_fn, visualise_cb=None):
    """
    A* Search
    ──────────
    f(n) = g(n) + h(n)
    g(n) = actual cost from start to n  (1 per step here)
    h(n) = heuristic estimate to goal

    Uses an EXPANDED list (closed set).  A node can be re-opened
    if a cheaper path is found later (important for admissibility
    with possibly inconsistent heuristics).

    Returns:
        (path, nodes_expanded, path_cost)
    """
    start = grid.start
    goal  = grid.goal

    # Each heap entry: (f, g, node, path)
    # Using g as a tiebreaker helps prefer deeper nodes when f is equal.
    open_heap = []
    g_start = 0
    h_start = heuristic_fn(start, goal)
    heapq.heappush(open_heap, (g_start + h_start, g_start, start, [start]))

    # best_g[node] = cheapest g-cost found so far to reach that node
    best_g = {start: 0}

    expanded = set()   # expanded / closed list
    nodes_expanded = 0

    while open_heap:
        f_val, g_val, current, path = heapq.heappop(open_heap)

        # Skip if we already expanded this node with a cheaper g
        if current in expanded:
            continue

        expanded.add(current)
        nodes_expanded += 1

        # ── Visualisation callback ──
        if visualise_cb:
            frontier_set = {item[2] for item in open_heap}
            visualise_cb(frontier_set, expanded)

        # ── Goal check ──
        if current == goal:
            return path, nodes_expanded, g_val

        # ── Expand neighbours ──
        for neighbour in grid.get_neighbours(*current):
            new_g = g_val + 1    # uniform cost = 1 per step
            # Only consider this path if it's better than any known path
            if new_g < best_g.get(neighbour, float('inf')):
                best_g[neighbour] = new_g
                new_f = new_g + heuristic_fn(neighbour, goal)
                new_path = path + [neighbour]
                heapq.heappush(open_heap, (new_f, new_g, neighbour, new_path))

    return [], nodes_expanded, 0


# ─────────────────────────────────────────────
#  BUTTON HELPER CLASS
# ─────────────────────────────────────────────
class Button:
    """Simple rectangular button for the Pygame UI."""

    def __init__(self, x, y, w, h, label, active=False):
        self.rect   = pygame.Rect(x, y, w, h)
        self.label  = label
        self.active = active

    def draw(self, surface, font):
        color     = BTN_ACTIVE   if self.active else BTN_INACTIVE
        txt_color = BTN_TEXT_ACT if self.active else BTN_TEXT_IN
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        # Subtle border
        pygame.draw.rect(surface, LIGHT_GRAY, self.rect, 1, border_radius=6)
        text_surf = font.render(self.label, True, txt_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


# ─────────────────────────────────────────────
#  SLIDER HELPER CLASS
# ─────────────────────────────────────────────
class Slider:
    """
    A simple horizontal slider.
    value is a float in [min_val, max_val].
    """

    def __init__(self, x, y, w, min_val, max_val, init_val, label):
        self.x       = x
        self.y       = y
        self.w       = w
        self.h       = 6
        self.min_val = min_val
        self.max_val = max_val
        self.value   = init_val
        self.label   = label
        self.dragging = False

    def draw(self, surface, font):
        # Track
        track_rect = pygame.Rect(self.x, self.y + 10, self.w, self.h)
        pygame.draw.rect(surface, LIGHT_GRAY, track_rect, border_radius=3)

        # Fill up to thumb
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_w = int(ratio * self.w)
        fill_rect = pygame.Rect(self.x, self.y + 10, fill_w, self.h)
        pygame.draw.rect(surface, ACCENT, fill_rect, border_radius=3)

        # Thumb
        thumb_x = self.x + fill_w
        pygame.draw.circle(surface, ACCENT, (thumb_x, self.y + 13), 8)

        # Label + value
        lbl = font.render(f"{self.label}: {self.value:.2f}", True, TEXT_COLOR)
        surface.blit(lbl, (self.x, self.y - 14))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            thumb_x = self.x + int((self.value - self.min_val) /
                                    (self.max_val - self.min_val) * self.w)
            if abs(event.pos[0] - thumb_x) <= 10 and abs(event.pos[1] - (self.y + 13)) <= 10:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = max(0, min(event.pos[0] - self.x, self.w))
            self.value = self.min_val + (rel_x / self.w) * (self.max_val - self.min_val)
            self.value = round(self.value, 2)


# ─────────────────────────────────────────────
#  MAIN APPLICATION CLASS
# ─────────────────────────────────────────────
class PathfinderApp:
    """
    Main application class.
    Manages the Pygame window, grid rendering, UI controls,
    user interaction, and search execution.
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Dynamic Pathfinding Agent  |  AI Assignment 2")

        # ── Default grid dimensions ──
        self.grid_rows = 20
        self.grid_cols = 25

        # Compute window size
        self.grid_pixel_w = self.grid_cols * CELL_SIZE
        self.grid_pixel_h = self.grid_rows * CELL_SIZE
        self.win_w = self.grid_pixel_w + PANEL_WIDTH
        self.win_h = max(self.grid_pixel_h, 700)

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock  = pygame.time.Clock()

        # ── Fonts ──
        self.font_sm  = pygame.font.SysFont("Consolas", 12)
        self.font_md  = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_lg  = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_hd  = pygame.font.SysFont("Consolas", 18, bold=True)

        # ── Grid & state ──
        self.grid = Grid(self.grid_rows, self.grid_cols)
        self.reset_search_state()

        # ── Mode flags ──
        self.placing_start    = False   # next click sets start
        self.placing_goal     = False   # next click sets goal
        self.drawing_wall     = False   # mouse-drag paints walls
        self.erasing_wall     = False   # mouse-drag erases walls
        self.dynamic_mode     = False   # dynamic obstacle spawning
        self.search_running   = False   # a search is in progress
        self.search_done      = False

        # ── Algorithm & heuristic selection ──
        self.algo      = "A*"           # "GBFS" or "A*"
        self.heuristic = "Manhattan"    # "Manhattan" or "Euclidean"

        # ── Build UI buttons ──
        px = self.grid_pixel_w + 15    # panel x start
        self._build_ui(px)

    # ──────────────────────────────────────────
    #  UI CONSTRUCTION
    # ──────────────────────────────────────────
    def _build_ui(self, px):
        """Create all Button and Slider objects for the right panel."""
        bw  = PANEL_WIDTH - 30   # button width
        bh  = 32                 # button height
        gap = 10

        y = 15

        # ── Algorithm selector ──
        self.btn_astar = Button(px, y,     bw // 2 - 4, bh, "A* Search",    active=True)
        self.btn_gbfs  = Button(px + bw // 2 + 4, y, bw // 2 - 4, bh, "Greedy BFS", active=False)
        y += bh + gap

        # ── Heuristic selector ──
        self.btn_manhattan  = Button(px, y,          bw // 2 - 4, bh, "Manhattan",  active=True)
        self.btn_euclidean  = Button(px + bw // 2 + 4, y, bw // 2 - 4, bh, "Euclidean", active=False)
        y += bh + gap * 2

        # ── Obstacle density slider ──
        self.slider_density = Slider(px, y + 14, bw, 0.05, 0.60, 0.30, "Density")
        y += 55

        # ── Map controls ──
        self.btn_gen_maze   = Button(px, y, bw, bh, "Generate Random Maze")
        y += bh + gap
        self.btn_clear      = Button(px, y, bw, bh, "Clear Grid")
        y += bh + gap * 2

        # ── Editing modes ──
        self.btn_set_start  = Button(px, y,          bw // 2 - 4, bh, "Set Start")
        self.btn_set_goal   = Button(px + bw // 2 + 4, y, bw // 2 - 4, bh, "Set Goal")
        y += bh + gap

        self.btn_draw_wall  = Button(px, y,          bw // 2 - 4, bh, "Draw Walls")
        self.btn_erase_wall = Button(px + bw // 2 + 4, y, bw // 2 - 4, bh, "Erase Walls")
        y += bh + gap * 2

        # ── Search controls ──
        self.btn_run     = Button(px, y, bw, bh, "▶  Run Search")
        y += bh + gap
        self.btn_dynamic = Button(px, y, bw, bh, "Dynamic Mode: OFF")
        y += bh + gap
        self.btn_reset   = Button(px, y, bw, bh, "Reset Search")
        y += bh + gap * 3

        # Store the y position where metrics start
        self.metrics_y = y

    # ──────────────────────────────────────────
    #  STATE MANAGEMENT
    # ──────────────────────────────────────────
    def reset_search_state(self):
        """Clear all visualisation data without touching the grid walls."""
        self.frontier_cells = set()   # currently in open list
        self.visited_cells  = set()   # already expanded
        self.final_path     = []      # final solution path
        self.agent_pos      = None    # agent's current position (dynamic mode)
        self.path_index     = 0       # how far along the path the agent is

        # Metrics
        self.nodes_expanded  = 0
        self.path_cost       = 0
        self.exec_time_ms    = 0.0
        self.status_msg      = "Ready"

    # ──────────────────────────────────────────
    #  ALGORITHM EXECUTION
    # ──────────────────────────────────────────
    def _get_heuristic_fn(self):
        if self.heuristic == "Manhattan":
            return manhattan_distance
        else:
            return euclidean_distance

    def run_search(self, start_override=None):
        """
        Run the selected algorithm and animate step-by-step.
        start_override lets the agent re-plan from its current position.
        """
        self.search_running = True
        self.search_done    = False

        # If re-planning mid-route, use agent's current cell as start
        if start_override:
            self.grid.start = start_override

        hfn       = self._get_heuristic_fn()
        t_start   = time.perf_counter()

        # Collect visualisation frames for smooth animation
        frames = []   # list of (frontier_set, visited_set)

        def collect_frame(frontier, visited):
            frames.append((set(frontier), set(visited)))

        if self.algo == "A*":
            path, expanded, cost = astar_search(self.grid, hfn,
                                                visualise_cb=collect_frame)
        else:
            path, expanded, cost = greedy_bfs(self.grid, hfn,
                                              visualise_cb=collect_frame)

        t_end = time.perf_counter()

        # ── Animate the frames ──
        for frontier, visited in frames:
            self.frontier_cells = frontier
            self.visited_cells  = visited
            self.draw_everything()
            pygame.display.flip()
            pygame.time.delay(int(VIZ_DELAY * 1000))
            # Allow the user to quit during animation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        # ── Store results ──
        self.final_path      = path
        self.nodes_expanded  = expanded
        self.path_cost       = cost
        self.exec_time_ms    = (t_end - t_start) * 1000
        self.agent_pos       = self.grid.start
        self.path_index      = 0

        if path:
            self.status_msg = f"Path found! Cost={cost}"
        else:
            self.status_msg = "No path found!"

        self.search_running = False
        self.search_done    = True

    # ──────────────────────────────────────────
    #  DYNAMIC MODE – agent walks & re-plans
    # ──────────────────────────────────────────
    def step_dynamic(self):
        """
        Move the agent one step forward along the current path.
        With a small probability, spawn a new obstacle; if it
        falls on the remaining path, trigger immediate re-planning.
        """
        if not self.final_path or self.path_index >= len(self.final_path) - 1:
            return   # already at goal or no path

        # Move agent one step
        self.path_index += 1
        self.agent_pos = self.final_path[self.path_index]

        # ── Maybe spawn an obstacle ──
        SPAWN_PROB = 0.15   # 15% chance per step
        if random.random() < SPAWN_PROB:
            remaining_path = self.final_path[self.path_index:]
            new_obs = self.grid.spawn_dynamic_obstacle(
                current_path=remaining_path
            )
            # Check if the new obstacle is on the remaining path
            if new_obs and new_obs in set(remaining_path):
                # Re-plan from current position
                old_start = self.grid.start
                self.status_msg = "Obstacle on path! Re-planning..."
                self.run_search(start_override=self.agent_pos)
                self.grid.start = old_start   # restore original start

    # ──────────────────────────────────────────
    #  DRAWING
    # ──────────────────────────────────────────
    def draw_everything(self):
        """Render the full frame: grid + panel."""
        self.screen.fill(BLACK)
        self._draw_grid()
        self._draw_panel()

    def _draw_grid(self):
        """Draw all grid cells with appropriate colours."""
        sr, sc = self.grid.start
        gr, gc = self.grid.goal

        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                x = c * CELL_SIZE
                y = r * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE - 1, CELL_SIZE - 1)

                # ── Determine cell colour ──
                if self.grid.cells[r][c] == 1:
                    color = WALL_COLOR
                elif (r, c) == (sr, sc):
                    color = START_COLOR
                elif (r, c) == (gr, gc):
                    color = GOAL_COLOR
                elif (r, c) in self.final_path:
                    color = PATH_COLOR
                elif (r, c) == self.agent_pos and self.dynamic_mode:
                    color = START_COLOR
                elif (r, c) in self.visited_cells:
                    color = VISITED_COLOR
                elif (r, c) in self.frontier_cells:
                    color = FRONTIER_COLOR
                else:
                    color = GRAY

                pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # Draw grid lines
        for r in range(self.grid.rows + 1):
            pygame.draw.line(self.screen, GRID_LINE,
                             (0, r * CELL_SIZE),
                             (self.grid_pixel_w, r * CELL_SIZE))
        for c in range(self.grid.cols + 1):
            pygame.draw.line(self.screen, GRID_LINE,
                             (c * CELL_SIZE, 0),
                             (c * CELL_SIZE, self.grid_pixel_h))

        # ── Highlight agent position in dynamic mode ──
        if self.dynamic_mode and self.agent_pos:
            ar, ac = self.agent_pos
            ax = ac * CELL_SIZE + CELL_SIZE // 2
            ay = ar * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, WHITE, (ax, ay), CELL_SIZE // 3)

    def _draw_panel(self):
        """Draw the right-side control panel."""
        panel_rect = pygame.Rect(self.grid_pixel_w, 0, PANEL_WIDTH, self.win_h)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, LIGHT_GRAY,
                         (self.grid_pixel_w, 0),
                         (self.grid_pixel_w, self.win_h), 1)

        # Title
        title = self.font_hd.render("Pathfinding Agent", True, ACCENT)
        self.screen.blit(title, (self.grid_pixel_w + 15, 0))  # covered by buttons below – adjust

        # ── Draw all buttons ──
        for btn in [self.btn_astar, self.btn_gbfs,
                    self.btn_manhattan, self.btn_euclidean,
                    self.btn_gen_maze, self.btn_clear,
                    self.btn_set_start, self.btn_set_goal,
                    self.btn_draw_wall, self.btn_erase_wall,
                    self.btn_run, self.btn_dynamic, self.btn_reset]:
            btn.draw(self.screen, self.font_md)

        # ── Density slider ──
        self.slider_density.draw(self.screen, self.font_sm)

        # ── Section labels ──
        px = self.grid_pixel_w + 15
        lbl_algo = self.font_sm.render("── Algorithm ──────────────", True, LIGHT_GRAY)
        self.screen.blit(lbl_algo, (px, self.btn_astar.rect.top - 16))
        lbl_heur = self.font_sm.render("── Heuristic ──────────────", True, LIGHT_GRAY)
        self.screen.blit(lbl_heur, (px, self.btn_manhattan.rect.top - 16))

        # ── Metrics ──
        my = self.metrics_y
        self._draw_label("── Metrics ──────────────", px, my);       my += 20
        self._draw_label(f"Nodes Expanded : {self.nodes_expanded}", px, my); my += 18
        self._draw_label(f"Path Cost      : {self.path_cost}",       px, my); my += 18
        self._draw_label(f"Exec Time (ms) : {self.exec_time_ms:.1f}", px, my); my += 28

        # ── Status ──
        self._draw_label(f"Status: {self.status_msg}", px, my, color=ACCENT); my += 28

        # ── Legend ──
        self._draw_legend(px, my)

        # ── Grid info ──
        info = self.font_sm.render(
            f"Grid: {self.grid.rows}×{self.grid.cols}  "
            f"Start:{self.grid.start}  Goal:{self.grid.goal}",
            True, LIGHT_GRAY)
        self.screen.blit(info, (px, self.win_h - 20))

    def _draw_label(self, text, x, y, color=TEXT_COLOR):
        surf = self.font_sm.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def _draw_legend(self, px, y):
        legend = [
            (START_COLOR,    "Start / Agent"),
            (GOAL_COLOR,     "Goal"),
            (FRONTIER_COLOR, "Frontier (Open List)"),
            (VISITED_COLOR,  "Visited (Expanded)"),
            (PATH_COLOR,     "Final Path"),
            (WALL_COLOR,     "Wall / Obstacle"),
        ]
        self._draw_label("── Legend ───────────────", px, y); y += 20
        for color, label in legend:
            pygame.draw.rect(self.screen, color, (px, y + 2, 14, 14), border_radius=3)
            self._draw_label(label, px + 20, y)
            y += 18

    # ──────────────────────────────────────────
    #  EVENT HANDLING
    # ──────────────────────────────────────────
    def handle_events(self):
        """Process all Pygame events each frame."""
        for event in pygame.event.get():

            # ── Quit ──
            if event.type == pygame.QUIT:
                return False

            # ── Slider ──
            self.slider_density.handle_event(event)

            # ── Mouse button down ──
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                # ─ Clicks inside the GRID area ─
                if mx < self.grid_pixel_w and my < self.grid_pixel_h:
                    r = my // CELL_SIZE
                    c = mx // CELL_SIZE
                    if self.grid.in_bounds(r, c):
                        if self.placing_start:
                            self.grid.start = (r, c)
                            self.placing_start = False
                            self.btn_set_start.active = False
                            self.reset_search_state()
                        elif self.placing_goal:
                            self.grid.goal = (r, c)
                            self.placing_goal = False
                            self.btn_set_goal.active = False
                            self.reset_search_state()
                        elif self.drawing_wall:
                            self.grid.set_wall(r, c, 1)
                        elif self.erasing_wall:
                            self.grid.set_wall(r, c, 0)

                # ─ Clicks inside the PANEL area ─
                else:
                    self._handle_panel_click(event.pos)

            # ── Mouse drag for wall drawing ──
            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:
                    mx, my = event.pos
                    if mx < self.grid_pixel_w and my < self.grid_pixel_h:
                        r = my // CELL_SIZE
                        c = mx // CELL_SIZE
                        if self.grid.in_bounds(r, c):
                            if self.drawing_wall:
                                self.grid.set_wall(r, c, 1)
                            elif self.erasing_wall:
                                self.grid.set_wall(r, c, 0)

        return True  # keep running

    def _handle_panel_click(self, pos):
        """Handle all button clicks in the right panel."""

        # ── Algorithm buttons (mutually exclusive) ──
        if self.btn_astar.is_clicked(pos):
            self.algo = "A*"
            self.btn_astar.active = True
            self.btn_gbfs.active  = False

        elif self.btn_gbfs.is_clicked(pos):
            self.algo = "GBFS"
            self.btn_gbfs.active  = True
            self.btn_astar.active = False

        # ── Heuristic buttons (mutually exclusive) ──
        elif self.btn_manhattan.is_clicked(pos):
            self.heuristic = "Manhattan"
            self.btn_manhattan.active = True
            self.btn_euclidean.active = False

        elif self.btn_euclidean.is_clicked(pos):
            self.heuristic = "Euclidean"
            self.btn_euclidean.active = True
            self.btn_manhattan.active = False

        # ── Map controls ──
        elif self.btn_gen_maze.is_clicked(pos):
            self.grid.generate_random_maze(density=self.slider_density.value)
            self.reset_search_state()

        elif self.btn_clear.is_clicked(pos):
            self.grid = Grid(self.grid_rows, self.grid_cols)
            self.reset_search_state()

        # ── Edit modes ──
        elif self.btn_set_start.is_clicked(pos):
            self.placing_start = not self.placing_start
            self.placing_goal  = False
            self.drawing_wall  = False
            self.erasing_wall  = False
            self.btn_set_start.active  = self.placing_start
            self.btn_set_goal.active   = False
            self.btn_draw_wall.active  = False
            self.btn_erase_wall.active = False

        elif self.btn_set_goal.is_clicked(pos):
            self.placing_goal  = not self.placing_goal
            self.placing_start = False
            self.drawing_wall  = False
            self.erasing_wall  = False
            self.btn_set_goal.active   = self.placing_goal
            self.btn_set_start.active  = False
            self.btn_draw_wall.active  = False
            self.btn_erase_wall.active = False

        elif self.btn_draw_wall.is_clicked(pos):
            self.drawing_wall  = not self.drawing_wall
            self.erasing_wall  = False
            self.placing_start = False
            self.placing_goal  = False
            self.btn_draw_wall.active  = self.drawing_wall
            self.btn_erase_wall.active = False
            self.btn_set_start.active  = False
            self.btn_set_goal.active   = False

        elif self.btn_erase_wall.is_clicked(pos):
            self.erasing_wall  = not self.erasing_wall
            self.drawing_wall  = False
            self.placing_start = False
            self.placing_goal  = False
            self.btn_erase_wall.active = self.erasing_wall
            self.btn_draw_wall.active  = False
            self.btn_set_start.active  = False
            self.btn_set_goal.active   = False

        # ── Search controls ──
        elif self.btn_run.is_clicked(pos):
            if not self.search_running:
                self.reset_search_state()
                self.run_search()

        elif self.btn_dynamic.is_clicked(pos):
            self.dynamic_mode = not self.dynamic_mode
            self.btn_dynamic.label  = f"Dynamic Mode: {'ON' if self.dynamic_mode else 'OFF'}"
            self.btn_dynamic.active = self.dynamic_mode

        elif self.btn_reset.is_clicked(pos):
            self.reset_search_state()

    # ──────────────────────────────────────────
    #  MAIN LOOP
    # ──────────────────────────────────────────
    def run(self):
        """
        The main game loop.
        Runs until the user closes the window.
        """
        dynamic_timer = 0   # counts frames between agent steps

        running = True
        while running:

            self.clock.tick(FPS)
            dynamic_timer += 1

            # ── Handle input ──
            running = self.handle_events()

            # ── Dynamic mode: step agent every ~20 frames ──
            if self.dynamic_mode and self.search_done and self.final_path:
                if dynamic_timer % 20 == 0:
                    self.step_dynamic()

            # ── Render ──
            self.draw_everything()
            pygame.display.flip()

        pygame.quit()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = PathfinderApp()
    app.run()