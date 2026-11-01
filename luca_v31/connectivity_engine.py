"""
connectivity_engine.py  (LucaSO V1 -- Priority 5: Connectivity/Path Reasoning)
=================================================================================
Neither LUCA-ARC V13 nor SOSEP-02 has a general connectivity/path-graph
engine (confirmed absent in the LucaSO technical documentation's Chapter
15.3: "no algorithm in the codebase operates ON [the relation] graph
structure -- no isomorphism test, no graph-edit-distance computation, no
traversal-based reasoning"). fit_fill_holes (primitives.py) and
extract_holes (grid_world.py) handle ONE specific topological pattern
(enclosed background regions) via a border-touching flood-fill, but there
is no general BFS/shortest-path/reachability machinery, and nothing that
traces an explicit path between two points or objects.

This module adds that general machinery as a small set of Law-producing
fitters following the existing contract, plus reusable BFS/reachability
utilities other fitters (or a future module) can call directly.

Scope, stated plainly: this covers the connectivity/path patterns that
show up as recognisable ARC task families -- shortest-path tracing
between two marker objects, flood-fill from a marker, reachable-vs-
unreachable region painting, and adjacency-degree-based object
recolouring (which finally gives grid_world.relational_graph()'s
`touching` predicate a real consumer, closing a gap the technical
documentation flagged as "built but unused" in Chapter 3.5). It does not
implement a general graph-isomorphism or maze-solving-with-turns engine;
BFS shortest path is unweighted and 4-directional, matching the vast
majority of ARC path-tracing tasks.
"""

from __future__ import annotations
from .diagnostics import record_exception
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Set
from collections import deque, Counter

from .grid_world import Grid, grid_shape, guess_backgrounds, extract_objects, Entity, touching

Pair = Tuple[Grid, Grid]


@dataclass
class Law:
    name: str
    predict: Callable[[Grid], Grid]
    description: str
    constant_bits: int = 0
    op_count: int = 1


def _fits_all(predict, pairs: List[Pair]) -> bool:
    from .budget import check as _budget_check
    _budget_check()
    try:
        for x, y in pairs:
            if predict(x) != y:
                return False
        return True
    except Exception:
        record_exception('connectivity_engine.py:55')
        return False


# ------------------------------------------------------------ reusable BFS/connectivity utilities

DIRS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def bfs_shortest_path(g: Grid, start: Tuple[int, int], goal: Tuple[int, int],
                       passable) -> Optional[List[Tuple[int, int]]]:
    """Unweighted 4-directional BFS shortest path from start to goal,
    stepping only through cells where passable(r, c) is True (start and
    goal themselves are always allowed regardless of passable). Returns
    the full cell list including both endpoints, or None if unreachable."""
    h, w = grid_shape(g)
    if start == goal:
        return [start]
    visited = {start}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in DIRS_4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w) or nxt in visited:
                continue
            if nxt != goal and not passable(*nxt):
                continue
            visited.add(nxt)
            parent[nxt] = cur
            if nxt == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            q.append(nxt)
    return None


def reachable_from(g: Grid, sources: Set[Tuple[int, int]], passable) -> Set[Tuple[int, int]]:
    """All cells reachable from any of `sources` by 4-directional BFS
    through cells where passable(r, c) is True."""
    h, w = grid_shape(g)
    visited = set(sources)
    q = deque(sources)
    while q:
        cur = q.popleft()
        for dr, dc in DIRS_4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in visited and passable(*nxt):
                visited.add(nxt)
                q.append(nxt)
    return visited


def adjacency_graph(objects: List[Entity]) -> Dict[int, List[int]]:
    """Object adjacency graph via grid_world.touching() -- reused directly
    rather than reimplemented, finally giving that relation a real
    downstream consumer (see module docstring)."""
    adj: Dict[int, List[int]] = {o.id: [] for o in objects}
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if touching(a, b):
                adj[a.id].append(b.id)
                adj[b.id].append(a.id)
    return adj


# ------------------------------------------------------------ 1. shortest-path tracing between two markers

def fit_shortest_path_paint(pairs: List[Pair]) -> Optional[Law]:
    """Exactly two single-cell (or small) marker objects of distinct
    colours exist; the output additionally paints the shortest 4-
    directional path between them (through background cells) in a
    learned path colour, leaving the two markers themselves unchanged."""
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None

    def endpoints(g: Grid, bg: int):
        objs = extract_objects(g, bg, connectivity=8, multicolor=False)
        if len(objs) != 2:
            return None
        (a, b) = objs
        return a, b

    # learn the path colour from the first pair
    x0, y0 = pairs[0]
    bg0 = guess_backgrounds(x0, k=1)[0]
    ep0 = endpoints(x0, bg0)
    if ep0 is None:
        return None
    a0, b0 = ep0
    path0 = bfs_shortest_path(x0, next(iter(a0.cells)), next(iter(b0.cells)),
                               passable=lambda r, c: x0[r][c] == bg0)
    if not path0:
        return None
    interior0 = [p for p in path0 if x0[p[0]][p[1]] == bg0]
    if not interior0:
        return None
    path_colors = {y0[r][c] for r, c in interior0}
    if len(path_colors) != 1:
        return None
    path_color = next(iter(path_colors))

    def predict(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        ep = endpoints(g, bg)
        if ep is None:
            return g
        a, b = ep
        path = bfs_shortest_path(g, next(iter(a.cells)), next(iter(b.cells)),
                                  passable=lambda r, c: g[r][c] == bg)
        if not path:
            return g
        h, w = grid_shape(g)
        grid = [list(row) for row in g]
        for (r, c) in path:
            if g[r][c] == bg:
                grid[r][c] = path_color
        return tuple(tuple(row) for row in grid)

    if _fits_all(predict, pairs):
        return Law("shortest_path_paint", predict,
                    f"PATH(BFS shortest, between the 2 marker objects) -> PAINT(colour derived={path_color})",
                    constant_bits=1, op_count=2)
    return None


# ------------------------------------------------------------ 2. flood fill from a marker

def fit_flood_fill_from_marker(pairs: List[Pair]) -> Optional[Law]:
    """A single marker object exists inside a connected background region;
    the output flood-fills that entire reachable background region with a
    learned colour, starting from the marker's own position."""
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None

    x0, y0 = pairs[0]
    bg0 = guess_backgrounds(x0, k=1)[0]
    objs0 = extract_objects(x0, bg0, connectivity=8, multicolor=False)
    if len(objs0) != 1:
        return None
    marker0 = objs0[0]
    region0 = reachable_from(x0, set(marker0.cells), passable=lambda r, c: x0[r][c] == bg0)
    interior0 = region0 - marker0.cells
    if not interior0:
        return None
    fill_colors = {y0[r][c] for r, c in interior0}
    if len(fill_colors) != 1:
        return None
    fill_color = next(iter(fill_colors))

    def predict(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        objs = extract_objects(g, bg, connectivity=8, multicolor=False)
        if len(objs) != 1:
            return g
        marker = objs[0]
        region = reachable_from(g, set(marker.cells), passable=lambda r, c: g[r][c] == bg)
        h, w = grid_shape(g)
        grid = [list(row) for row in g]
        for (r, c) in region - marker.cells:
            grid[r][c] = fill_color
        return tuple(tuple(row) for row in grid)

    if _fits_all(predict, pairs):
        return Law("flood_fill_from_marker", predict,
                    f"FLOOD_FILL(reachable background from marker) -> PAINT(colour derived={fill_color})",
                    constant_bits=1, op_count=2)
    return None


# ------------------------------------------------------------ 3. reachable vs. unreachable region painting

def fit_reachable_unreachable_paint(pairs: List[Pair]) -> Optional[Law]:
    """Background cells reachable from the grid border are distinguished
    from background cells that are NOT reachable from the border (i.e.
    enclosed regions) -- a generalisation of fit_fill_holes that paints
    BOTH the reachable and unreachable background regions with their own
    learned colours, rather than only filling the enclosed ones."""
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None

    def regions(g: Grid, bg: int):
        h, w = grid_shape(g)
        border = {(r, c) for r in range(h) for c in range(w)
                  if (r in (0, h - 1) or c in (0, w - 1)) and g[r][c] == bg}
        reachable = reachable_from(g, border, passable=lambda r, c: g[r][c] == bg)
        all_bg = {(r, c) for r in range(h) for c in range(w) if g[r][c] == bg}
        unreachable = all_bg - reachable
        return reachable, unreachable

    x0, y0 = pairs[0]
    bg0 = guess_backgrounds(x0, k=1)[0]
    reach0, unreach0 = regions(x0, bg0)
    if not unreach0:
        return None  # nothing enclosed; fit_fill_holes already covers the simpler case
    reach_colors = {y0[r][c] for r, c in reach0} if reach0 else set()
    unreach_colors = {y0[r][c] for r, c in unreach0}
    if len(unreach_colors) != 1 or len(reach_colors) > 1:
        return None
    unreach_color = next(iter(unreach_colors))
    reach_color = next(iter(reach_colors)) if reach_colors else bg0

    def predict(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        reach, unreach = regions(g, bg)
        h, w = grid_shape(g)
        grid = [list(row) for row in g]
        for (r, c) in reach:
            grid[r][c] = reach_color
        for (r, c) in unreach:
            grid[r][c] = unreach_color
        return tuple(tuple(row) for row in grid)

    if _fits_all(predict, pairs):
        return Law("reachable_unreachable_paint", predict,
                    f"REGION(reachable-from-border vs enclosed) -> "
                    f"PAINT(reachable derived={reach_color}, enclosed derived={unreach_color})",
                    constant_bits=2, op_count=2)
    return None


# ------------------------------------------------------------ 4. adjacency-degree object recolouring

def fit_recolor_by_adjacency_degree(pairs: List[Pair]) -> Optional[Law]:
    """Each object's new colour is an association-table lookup keyed by
    its DEGREE in the object-adjacency graph (how many other objects it
    touches) -- the first fitter in this codebase to actually consume
    grid_world.touching()/relational_graph()'s relational structure."""
    from .budget import check as _budget_check
    _budget_check()
    if any(grid_shape(x) != grid_shape(y) for x, y in pairs):
        return None

    mapping: Dict[int, int] = {}
    for x, y in pairs:
        bg = guess_backgrounds(x, k=1)[0]
        objs = extract_objects(x, bg, connectivity=8, multicolor=False)
        if not objs:
            return None
        adj = adjacency_graph(objs)
        for o in objs:
            degree = len(adj[o.id])
            out_colors = {y[r][c] for r, c in o.cells}
            if len(out_colors) != 1:
                return None
            oc = next(iter(out_colors))
            if degree in mapping and mapping[degree] != oc:
                return None
            mapping[degree] = oc
    if len(mapping) < 2:
        return None  # a single-entry table is indistinguishable from a constant recolour

    def predict(g: Grid) -> Grid:
        bg = guess_backgrounds(g, k=1)[0]
        objs = extract_objects(g, bg, connectivity=8, multicolor=False)
        adj = adjacency_graph(objs)
        grid = [list(row) for row in g]
        for o in objs:
            degree = len(adj[o.id])
            if degree in mapping:
                for (r, c) in o.cells:
                    grid[r][c] = mapping[degree]
        return tuple(tuple(row) for row in grid)

    if _fits_all(predict, pairs):
        return Law("recolor_by_adjacency_degree", predict,
                    f"MAP(objects, Degree(adjacency_graph) -> LOOKUP(colour)) [{len(mapping)} entries]",
                    constant_bits=len(mapping), op_count=2)
    return None


# ------------------------------------------------------------ registry

CONNECTIVITY_FITTERS: List[Callable[[List[Pair]], Optional[Law]]] = [
    fit_shortest_path_paint,
    fit_flood_fill_from_marker,
    fit_reachable_unreachable_paint,
    fit_recolor_by_adjacency_degree,
]


def fit_all_connectivity(pairs: List[Pair]) -> List[Law]:
    from .budget import check as _budget_check
    _budget_check()
    return [law for law, _f in fit_all_connectivity_traced(pairs)]


def fit_all_connectivity_traced(pairs: List[Pair]):
    from .budget import check as _budget_check
    _budget_check()
    traced = []
    for fitter in CONNECTIVITY_FITTERS:
        try:
            result = fitter(pairs)
        except Exception:
            record_exception('connectivity_engine.py:346')
            result = None
        if result is not None:
            traced.append((result, fitter))
    return traced
