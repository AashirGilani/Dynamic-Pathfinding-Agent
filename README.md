# Dynamic-Pathfinding-Agent
A Pygame-based grid pathfinding visualiser that implements A* and Greedy Best-First Search with Manhattan and Euclidean heuristics. Features an interactive map editor, random maze generation, step-by-step search animation, and a dynamic mode where the agent navigates live and automatically re-plans when new obstacles appear on its path.

## Setup

```bash
pip install pygame
python dynamic_pathfinder.py
```

---

## Features

- **A\* Search** — optimal, uses `f(n) = g(n) + h(n)`
- **Greedy BFS** — fast but not optimal, uses `f(n) = h(n)`
- **Heuristics** — Manhattan Distance or Euclidean Distance (toggle in GUI)
- **Interactive grid** — draw/erase walls, move start & goal, generate random mazes
- **Dynamic Mode** — agent walks the path live; re-plans instantly if a new obstacle blocks the way
- **Metrics** — nodes expanded, path cost, execution time (ms)

## Colour Guide

| Colour | Meaning |
|---|---|
| 🟡 Yellow | Frontier (open list) |
| 🔵 Blue | Visited (expanded) |
| 💚 Green | Final path |
| ⚫ Dark | Wall |

---

## Author

**[Ali Aashir]** — Roll No: `24F-0535`
