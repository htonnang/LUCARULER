"""Explicit background policies independent of frequency rank."""
from collections import Counter
from .grid_world import guess_backgrounds


def resolve(grid,mode):
    if isinstance(mode,int):
        ranked=guess_backgrounds(grid,2)
        if not 0<=mode<len(ranked):raise ValueError('background rank unavailable')
        return ranked[mode]
    if mode.startswith('color:'):
        value=int(mode.split(':')[1])
        if not 0<=value<=9:raise ValueError('invalid literal background')
        return value
    h,w=len(grid),len(grid[0])
    if mode=='corners':
        values={grid[r][c] for r,c in ((0,0),(0,w-1),(h-1,0),(h-1,w-1))}
        if len(values)!=1:raise ValueError('corners disagree')
        return next(iter(values))
    if mode=='border':
        counts=Counter(grid[r][c] for r in range(h) for c in range(w) if r in (0,h-1) or c in (0,w-1))
        ranked=counts.most_common()
        if len(ranked)>1 and ranked[0][1]==ranked[1][1]:raise ValueError('border majority tied')
        return ranked[0][0]
    raise ValueError('unknown background policy')


def policies(pairs,novel_only=False):
    """Deduplicate by resolved demonstration backgrounds, never test answers."""
    common=set.intersection(*({v for row in x for v in row} for x,y in pairs))
    seen=set()
    if novel_only:
        for mode in (0,1):
            try:seen.add(tuple(resolve(x,mode) for x,y in pairs))
            except ValueError:pass
    candidates=['color:'+str(c) for c in sorted(common)]+['corners','border']
    if not novel_only:candidates += [0,1]
    for mode in candidates:
        try:key=tuple(resolve(x,mode) for x,y in pairs)
        except ValueError:continue
        if key in seen:continue
        seen.add(key)
        yield mode
