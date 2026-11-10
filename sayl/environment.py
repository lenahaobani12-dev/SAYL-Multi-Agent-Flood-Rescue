"""
The flood environment: a grid that represents a neighbourhood in Jeddah.

What this file does
-------------------
* MAP. A 30 x 20 grid (one cell is about 50 m, so the map is roughly 1.5 km x 1 km).
  Every cell has a ground elevation in metres. A "wadi" (dry river valley) runs
  through the middle and slopes gently towards the downstream edge (right side),
  so it is the lowest part of the map. The Command Center sits on high ground in
  the top-left corner and the shelters are placed on the highest ground.
* ROADS. A street grid is laid over the map (``road_map()``). A road cell is
  CLOSED when the water on it is deeper than ``ROAD_BLOCK_DEPTH`` (0.6 m, deeper
  than a normal car can drive through). ``closed_road_map()`` and
  ``road_closure_rate()`` tell the other agents and the UI which roads are closed.
* WATER SPREAD. Every cell stores a water depth. Each tick:
    1. rain adds water to every cell        (rain_intensity 0..1),
    2. water flows from higher to lower water surfaces (4 neighbours),
    3. part of the water leaves the map through the outlet (downstream edge),
    4. a little water drains away or soaks into the ground everywhere.
  When the rain stops (rain_intensity = 0) steps 3 and 4 make the water recede.
* FORECAST. ``forecast(n)`` runs the same rules n ticks into the future on a copy,
  so the Flood Monitor can warn before a place is flooded.
* PATH FINDING. ``find_path(start, goal, mode)`` is an A* search that avoids closed
  roads (mode "ground") or prefers water (mode "boat"). Rescue agents use this route.
* INCIDENTS. Incidents (people asking for help) live here, so every agent sees them.

One tick = 1 simulated minute. The rain / drain numbers are scaled for the demo
(a flood that develops in about an hour of simulated time), not measured values.

Compatibility
-------------
``water_surface`` (used by the model statistics and the UI title) is the water
level at the deepest point of the map, in metres. ``depth(pos)``, ``depth_map()``,
``is_blocked(pos)`` and ``elevation`` work as before.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]

ROAD_BLOCK_DEPTH = 0.6   # metres of water that make a road impassable for cars
BOAT_MIN_DEPTH = 0.3     # metres of water a boat needs to float
WET_DEPTH = 0.02         # below this a cell is shown as dry on the map


@dataclass
class Incident:
    """One distress case. Created by the Distress Agent, handled by the others."""

    incident_id: int
    pos: Cell
    text: str
    created_tick: int
    severity: int = 1            # 1 (low) .. 5 (critical)
    people: int = 1              # People still awaiting pickup at the incident
    total_people: int = 1        # Original group size, including all rescue trips
    rescued_people: int = 0      # Picked up by rescue teams
    sheltered_people: int = 0    # Admitted by shelter agents
    staged_people: int = 0       # Delivered to temporary command-centre staging
    tags: List[str] = field(default_factory=list)
    status: str = "NEW"          # NEW -> VERIFYING -> VERIFIED -> RESCUING -> RESCUED -> SHELTERED
    drone: Optional[str] = None
    boat: Optional[str] = None
    shelter: Optional[str] = None
    rescued_tick: Optional[int] = None

    @property
    def open(self) -> bool:
        return self.status not in ("RESCUED", "SHELTERED")


class FloodEnvironment:
    # ---- water model constants (change them here to tune the demo) ----------
    RAIN_RATE = 0.02         # metres of water added per tick when rain_intensity = 1
    FLOW_K = 0.20            # share of the surface difference that flows per tick (< 0.25 = stable)
    OUTLET_FRAC = 0.25       # share of the water in the outlet column that leaves per tick
    DRAIN_RATE = 0.004       # metres per tick lost to drains / soaking into the ground
    INITIAL_LEVEL = 0.5      # the flood has already started: the lowest ground is under 0.5 m

    # ---- street grid ---------------------------------------------------------
    ROAD_EVERY_X = 7         # a north-south road every 7 columns
    ROAD_EVERY_Y = 6         # an east-west road every 6 rows

    def __init__(self, width: int = 30, height: int = 20, rain_intensity: float = 0.5,
                 seed: int = 7) -> None:
        self.width = width
        self.height = height
        self.rain_intensity = rain_intensity      # 0 (no rain) .. 1 (extreme)
        self.rng = random.Random(seed)
        self.tick = 0
        self.elevation = self._make_elevation(seed)
        self.roads = self._make_roads()
        # water depth in every cell (metres): the flood has already started in the wadi
        self.water = np.clip(self.INITIAL_LEVEL - self.elevation, 0.0, None)
        self.incidents: Dict[int, Incident] = {}
        self._next_incident = 1
        self.command_center_pos: Cell = (1, 1)

    # ------------------------------------------------------------------ terrain
    def _make_elevation(self, seed: int) -> np.ndarray:
        """Wadi in the middle, higher ground at the edges, gentle slope downstream, small bumps."""
        rs = np.random.RandomState(seed)
        ys = np.arange(self.height).reshape(-1, 1)
        xs = np.arange(self.width).reshape(1, -1)
        valley_center = self.height / 2 + 3 * np.sin(xs / 5.0)
        dist = np.abs(ys - valley_center)
        bumps = rs.uniform(0, 0.4, size=(self.height, self.width))
        bumps = self._smooth(bumps)                 # smooth bumps = no one-cell holes that trap water
        elevation = 0.25 * dist + bumps - 0.02 * xs  # the -0.02*x term is the slope towards the outlet
        return elevation.astype(float)

    @staticmethod
    def _smooth(a: np.ndarray) -> np.ndarray:
        """Average every cell with its 8 neighbours (edges use their own value)."""
        p = np.pad(a, 1, mode="edge")
        total = np.zeros_like(a)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                total += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
        return total / 9.0

    def _make_roads(self) -> np.ndarray:
        """Boolean map of the street grid (True = road)."""
        roads = np.zeros((self.height, self.width), dtype=bool)
        roads[2::self.ROAD_EVERY_Y, :] = True       # east-west streets
        roads[:, 3::self.ROAD_EVERY_X] = True       # north-south streets
        return roads

    # ------------------------------------------------------------------ water queries
    def depth(self, pos: Cell) -> float:
        x, y = pos
        return max(0.0, float(self.water[y, x]))

    def depth_map(self) -> np.ndarray:
        return self.water.copy()

    @property
    def water_level(self) -> float:
        """Water level at the deepest point of the map (metres)."""
        return float(self.water.max())

    @property
    def water_surface(self) -> float:           # name kept for the model / UI
        return self.water_level

    def wet_cells(self) -> int:
        return int((self.water > WET_DEPTH).sum())

    def wadi_mask(self) -> np.ndarray:
        """The lowest 20 % of the terrain (the wadi and its low ground)."""
        return self.elevation <= np.percentile(self.elevation, 20)

    # ------------------------------------------------------------------ roads
    def is_road(self, pos: Cell) -> bool:
        x, y = pos
        return bool(self.roads[y, x])

    def is_blocked(self, pos: Cell) -> bool:
        """True if the water at this cell is too deep for vehicles."""
        return self.depth(pos) >= ROAD_BLOCK_DEPTH

    def road_map(self) -> np.ndarray:
        return self.roads.copy()

    def closed_road_map(self) -> np.ndarray:
        """Boolean map: road cells that are closed by deep water."""
        return self.roads & (self.water >= ROAD_BLOCK_DEPTH)

    def closed_roads(self) -> List[Cell]:
        ys, xs = np.nonzero(self.closed_road_map())
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def road_closure_rate(self) -> float:
        """Share of the road network that is closed (0..1)."""
        return float(self.closed_road_map().sum()) / float(self.roads.sum())

    def in_bounds(self, pos: Cell) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    # ------------------------------------------------------------------ dynamics
    def _advance(self, water: np.ndarray, rain: float) -> np.ndarray:
        """One tick of the water model on a depth array. Returns the new array (the input is not changed)."""
        elev = self.elevation
        w = water + rain * self.RAIN_RATE                      # 1. rain

        # 2. flow: water moves towards neighbours whose water surface is lower
        s = elev + w
        p = np.pad(s, 1, mode="edge")                          # border cells see themselves: no flow out of the map
        drop_up = np.maximum(s - p[:-2, 1:-1], 0)
        drop_down = np.maximum(s - p[2:, 1:-1], 0)
        drop_left = np.maximum(s - p[1:-1, :-2], 0)
        drop_right = np.maximum(s - p[1:-1, 2:], 0)
        total = drop_up + drop_down + drop_left + drop_right
        out = np.minimum(w, self.FLOW_K * total)               # a cell cannot give more water than it has
        share = np.divide(out, total, out=np.zeros_like(out), where=total > 1e-12)
        to_up, to_down = drop_up * share, drop_down * share
        to_left, to_right = drop_left * share, drop_right * share
        new = w - out
        new[:-1, :] += to_up[1:, :]                            # the cell below sends water up
        new[1:, :] += to_down[:-1, :]                          # the cell above sends water down
        new[:, :-1] += to_left[:, 1:]
        new[:, 1:] += to_right[:, :-1]

        # 3. the downstream (right) edge is the outlet of the wadi
        new[:, -1] *= (1.0 - self.OUTLET_FRAC)
        # 4. drains and soaking into the ground
        new = np.maximum(new - self.DRAIN_RATE, 0.0)
        return new

    def step(self) -> None:
        self.tick += 1
        self.water = self._advance(self.water, self.rain_intensity)

    def forecast(self, ticks: int, rain: Optional[float] = None) -> np.ndarray:
        """Predicted depth map after ``ticks`` more ticks (same rules, current or given rain)."""
        w = self.water
        r = self.rain_intensity if rain is None else rain
        for _ in range(max(0, ticks)):
            w = self._advance(w, r)
        return w.copy()

    def total_water(self) -> float:
        """Total volume of water on the map (m of depth summed over the cells)."""
        return float(self.water.sum())

    # ------------------------------------------------------------------ path finding
    def find_path(self, start: Cell, goal: Cell, mode: str = "ground") -> List[Cell]:
        """A* shortest path from ``start`` to ``goal`` (8 directions). Returns the cells after
        ``start`` up to and including ``goal``; an empty list if there is no path.

        mode "ground": closed roads / deep water cannot be crossed; roads are cheaper than open ground.
        mode "boat":   any cell can be crossed, but moving through water (>= BOAT_MIN_DEPTH) is
                       much cheaper than over dry land (the boat has to be carried).
        """
        if mode not in ("ground", "boat"):
            raise ValueError("mode must be 'ground' or 'boat'")
        if not (self.in_bounds(start) and self.in_bounds(goal)):
            return []
        if start == goal:
            return []

        def cell_cost(c: Cell) -> float:
            x, y = c
            d = self.water[y, x]
            if mode == "ground":
                if d >= ROAD_BLOCK_DEPTH and c != goal:
                    return math.inf
                return 1.0 if self.roads[y, x] else 3.0
            return 1.0 if d >= BOAT_MIN_DEPTH else 4.0

        open_heap: List[Tuple[float, float, Cell]] = [(self.distance(start, goal), 0.0, start)]
        came: Dict[Cell, Cell] = {}
        best: Dict[Cell, float] = {start: 0.0}
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path[1:]
            if g > best.get(cur, math.inf):
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nxt = (cur[0] + dx, cur[1] + dy)
                    if not self.in_bounds(nxt):
                        continue
                    step = cell_cost(nxt) * (math.sqrt(2) if dx and dy else 1.0)
                    if math.isinf(step):
                        continue
                    ng = g + step
                    if ng < best.get(nxt, math.inf):
                        best[nxt] = ng
                        came[nxt] = cur
                        heapq.heappush(open_heap, (ng + self.distance(nxt, goal), ng, nxt))
        return []

    # ------------------------------------------------------------------ incidents
    def add_incident(self, text: str, pos: Optional[Cell] = None) -> Incident:
        if pos is None:
            pos = self.random_low_cell()
        inc = Incident(self._next_incident, pos, text, self.tick)
        self.incidents[inc.incident_id] = inc
        self._next_incident += 1
        return inc

    def open_incidents(self) -> List[Incident]:
        return [i for i in self.incidents.values() if i.open]

    def random_low_cell(self) -> Cell:
        """A random cell in the lower part of the terrain (where people get trapped)."""
        threshold = float(np.percentile(self.elevation, 30))
        candidates = [(x, y) for y in range(self.height) for x in range(self.width)
                      if self.elevation[y, x] <= threshold]
        return self.rng.choice(candidates)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def distance(a: Cell, b: Cell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def step_towards(pos: Cell, target: Cell, speed: int = 1) -> Cell:
        """Move up to ``speed`` cells towards the target (straight line)."""
        x, y = pos
        for _ in range(speed):
            if (x, y) == target:
                break
            x += (target[0] > x) - (target[0] < x)
            y += (target[1] > y) - (target[1] < y)
        return (x, y)
