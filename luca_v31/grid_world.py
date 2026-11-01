"""grid_world.py -- RECONSTRUCTED.

The original LUCA grid_world.py was not included in the upload.  This module
re-implements the exact API surface consumed by primitives.py,
connectivity_engine.py, lazy_beam_search.py, luca_search.py and
luca_solver.py, inferred from their call sites:

    Grid = tuple[tuple[int,...],...]
    to_grid, grid_shape, color_counts, guess_backgrounds(g, k)
    extract_objects(g, bg, connectivity=8, multicolor=False) -> [Entity]
    extract_holes(g, bg) -> [Entity]        (enclosed background regions)
    Entity: id, cells(set), color, multicolor(dict), bbox, area, shape()
    touching(a, b), relational_graph(objects)

Behaviour is a faithful-as-possible reconstruction, NOT the original file;
LUCA results are therefore "LUCA fitters on reconstructed perception" and
are labelled as such in the evaluation report.
"""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

Grid = Tuple[Tuple[int, ...], ...]
Cell = Tuple[int, int]


def to_grid(rows) -> Grid:
    return tuple(tuple(int(c) for c in row) for row in rows)


def grid_shape(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def color_counts(g: Grid) -> Counter:
    return Counter(c for row in g for c in row)


def guess_backgrounds(g: Grid, k: int = 1) -> List[int]:
    """Most frequent colours, border-weighted (ties -> colour 0 first)."""
    h, w = grid_shape(g)
    cnt = color_counts(g)
    border = Counter()
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                border[g[r][c]] += 1
    n = max(1, h * w)
    nb = max(1, 2 * h + 2 * w - 4)
    ranked = sorted(cnt, key=lambda col: (-(0.55 * cnt[col] / n + 0.45 * border[col] / nb + (0.05 if col == 0 else 0)), col))
    return ranked[:k] if ranked else [0]


@dataclass
class Entity:
    id: int
    cells: Set[Cell]
    color: int
    multicolor: Dict[Cell, int] = field(default_factory=dict)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return (min(rs), min(cs), max(rs), max(cs))

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def size(self) -> int:
        return len(self.cells)

    def shape(self) -> FrozenSet[Cell]:
        r0, c0, _, _ = self.bbox
        return frozenset((r - r0, c - c0) for r, c in self.cells)


def _neigh(r, c, h, w, conn):
    d = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn == 8:
        d += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in d:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(cells: Set[Cell], h, w, conn) -> List[Set[Cell]]:
    seen: Set[Cell] = set()
    out = []
    for start in sorted(cells):
        if start in seen:
            continue
        comp = {start}
        seen.add(start)
        q = deque([start])
        while q:
            r, c = q.popleft()
            for n in _neigh(r, c, h, w, conn):
                if n in cells and n not in seen:
                    seen.add(n)
                    comp.add(n)
                    q.append(n)
        out.append(comp)
    return out


def extract_objects(g: Grid, bg: int, connectivity: int = 8, multicolor: bool = False) -> List[Entity]:
    from .budget import check as _budget_check
    _budget_check()
    h, w = grid_shape(g)
    ents: List[Entity] = []
    if multicolor:
        cells = {(r, c) for r in range(h) for c in range(w) if g[r][c] != bg}
        for i, comp in enumerate(_components(cells, h, w, connectivity)):
            mc = {(r, c): g[r][c] for r, c in comp}
            ents.append(Entity(i, comp, Counter(mc.values()).most_common(1)[0][0], mc))
        return ents
    i = 0
    for col in sorted(set(c for row in g for c in row)):
        if col == bg:
            continue
        cells = {(r, c) for r in range(h) for c in range(w) if g[r][c] == col}
        for comp in _components(cells, h, w, connectivity):
            ents.append(Entity(i, comp, col, {cell: col for cell in comp}))
            i += 1
    return ents


def extract_holes(g: Grid, bg: int) -> List[Entity]:
    """Background cells NOT 4-reachable from the grid border, grouped into
    4-connected regions."""
    from .budget import check as _budget_check
    _budget_check()
    h, w = grid_shape(g)
    border = {(r, c) for r in range(h) for c in range(w)
              if (r in (0, h - 1) or c in (0, w - 1)) and g[r][c] == bg}
    reach = set(border)
    q = deque(border)
    while q:
        r, c = q.popleft()
        for n in _neigh(r, c, h, w, 4):
            if g[n[0]][n[1]] == bg and n not in reach:
                reach.add(n)
                q.append(n)
    enclosed = {(r, c) for r in range(h) for c in range(w) if g[r][c] == bg} - reach
    return [Entity(i, comp, bg, {cell: bg for cell in comp})
            for i, comp in enumerate(_components(enclosed, h, w, 4))]


def touching(a: Entity, b: Entity) -> bool:
    for r, c in a.cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in b.cells:
                return True
    return False


def relational_graph(objects: List[Entity]) -> Dict[Tuple[int, int], Dict[str, bool]]:
    rel = {}
    for a in objects:
        for b in objects:
            if a.id == b.id:
                continue
            rel[(a.id, b.id)] = {"touching": touching(a, b), "same_color": a.color == b.color,
                                 "same_shape": a.shape() == b.shape(), "same_size": a.area == b.area}
    return rel
