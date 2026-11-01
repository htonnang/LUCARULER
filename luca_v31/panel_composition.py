"""Explicit panel selection, shared-coordinate placement and priority rendering.

Fit layout/background/priority from demonstrations, without task identifiers.
Each intermediate representation retains its source coordinates. No panel
is required to solve the whole task before it can be composed with another.
"""
from dataclasses import dataclass
from itertools import permutations, product
from .grid_world import guess_backgrounds
from .object_support import accepted
from .diagnostics import event


@dataclass(frozen=True)
class Layout:
    rows: int
    columns: int
    row_separator: int = 0
    column_separator: int = 0

    def shape(self, grid):
        h=len(grid)-(self.rows-1)*self.row_separator
        w=len(grid[0])-(self.columns-1)*self.column_separator
        if h<=0 or w<=0 or h%self.rows or w%self.columns:
            raise ValueError('unequal panel dimensions')
        return h//self.rows,w//self.columns


@dataclass(frozen=True)
class Panel:
    origin: tuple
    cells: tuple


LAYOUTS = tuple(Layout(rows,cols,rs,cs)
                for rows,cols in ((2,1),(1,2),(2,2))
                for rs in ((0,1) if rows>1 else (0,))
                for cs in ((0,1) if cols>1 else (0,)))


def select_panels(grid,layout):
    h,w=layout.shape(grid)
    if layout.row_separator and len(set(grid[h]))!=1:
        raise ValueError('nonuniform row separator')
    if layout.column_separator and len({row[w] for row in grid})!=1:
        raise ValueError('nonuniform column separator')
    panels=[]
    for row,col in product(range(layout.rows),range(layout.columns)):
        r=row*(h+layout.row_separator);c=col*(w+layout.column_separator)
        panels.append(Panel((r,c),tuple(tuple(grid[i][c:c+w]) for i in range(r,r+h))))
    return tuple(panels)


def background(grid,mode):
    if isinstance(mode,int):return mode
    values=guess_backgrounds(grid,k=2)
    rank=int(mode[-1])
    if rank>=len(values):raise ValueError('background rank absent')
    return values[rank]


def render_overlay(panels,bg,priority):
    """Later panel indices overwrite earlier non-background cells."""
    if sorted(priority)!=list(range(len(panels))):
        raise ValueError('priority must use each panel once')
    h,w=len(panels[0].cells),len(panels[0].cells[0])
    if any((len(p.cells),len(p.cells[0]))!=(h,w) for p in panels):
        raise ValueError('incompatible panel shapes')
    out=[[bg]*w for _ in range(h)]
    for index in priority:
        for r,row in enumerate(panels[index].cells):
            for c,colour in enumerate(row):
                if colour!=bg:out[r][c]=colour
    return tuple(map(tuple,out))


def compose(grid,layout,mode,priority):
    return render_overlay(select_panels(grid,layout),background(grid,mode),priority)


def fit(pairs):
    from .budget import check as _budget_check
    _budget_check()
    event('panel_fitter_called')
    if len(pairs)<2:return []
    layouts=[]
    for layout in LAYOUTS:
        try:
            if all(layout.shape(x)==(len(y),len(y[0])) for x,y in pairs):layouts.append(layout)
        except ValueError:continue
    if not layouts:
        event('panel_gate:no_layout');return []
    common=set.intersection(*({v for row in x for v in row} for x,y in pairs))
    found=[]
    for layout in layouts:
        try:extracted=[select_panels(x,layout) for x,y in pairs]
        except ValueError:
            event('panel_reject:separator');continue
        event('panel_layout_available',layout)
        for mode in ['rank0','rank1']+sorted(common):
            try:bgs=[background(x,mode) for x,y in pairs]
            except ValueError:continue
            # Necessary cell-wise condition before testing a global priority.
            if any(any(y[r][c] not in {p.cells[r][c] for p in panels}
                       for r in range(len(y)) for c in range(len(y[0])))
                   for panels,(x,y) in zip(extracted,pairs)):
                event('panel_reject:unavailable_colour');continue
            for priority in permutations(range(layout.rows*layout.columns)):
                event('panel_priority_tried')
                if any(render_overlay(panels,bg,priority)!=y
                       for panels,bg,(x,y) in zip(extracted,bgs,pairs)):
                    continue
                def predict(g,l=layout,m=mode,p=priority):return compose(g,l,m,p)
                name=f'panel_overlay_{layout.rows}x{layout.columns}_sep{layout.row_separator}{layout.column_separator}_{mode}_order'+''.join(map(str,priority))
                law=accepted(name,predict,pairs,'Select equal panels; align local coordinates; overlay inferred foreground priority',
                             bits=3+len(priority)+(isinstance(mode,int)),ops=3)
                if law:found.append(law)
    event('panel_exact_laws',len(found))
    return found
