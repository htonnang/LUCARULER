"""Demonstration-only object correspondences and split/merge evidence.
Links are hypotheses, not a forced one-to-one assignment. No test labels used.
"""
from collections import Counter
from .grid_world import extract_objects,guess_backgrounds
from .primitives import GEOM_OPS


def normalized(cells):
    r=min(r for r,c in cells);c=min(c for r,c in cells)
    return frozenset((rr-r,cc-c) for rr,cc in cells)

def mask(cells):
    n=normalized(cells);h=max(r for r,c in n)+1;w=max(c for r,c in n)+1
    return tuple(tuple(int((r,c) in n) for c in range(w)) for r in range(h))

def descriptor(o):
    return dict(id=o.id,area=o.area,bbox=o.bbox,colors=dict(Counter(o.multicolor.values())))

def match(x,y,bg,connectivity=4,multicolor=True):
    a=extract_objects(x,bg,connectivity,multicolor);b=extract_objects(y,bg,connectivity,multicolor)
    result=dict(background=bg,connectivity=connectivity,multicolor=multicolor,
                inputs=[descriptor(o) for o in a],outputs=[descriptor(o) for o in b],links=[],merges=[],splits=[])
    if len(a)>40 or len(b)>40:
        result['truncated']='more than 40 objects';return result
    am=[mask(o.cells) for o in a];bm=[mask(o.cells) for o in b]
    for i,oa in enumerate(a):
        for j,ob in enumerate(b):
            geoms=[k for k,fn in GEOM_OPS.items() if fn(am[i])==bm[j]]
            if geoms:
                same_palette=Counter(oa.multicolor.values())==Counter(ob.multicolor.values())
                result['links'].append(dict(source=i,target=j,geometries=geoms,palette_preserved=same_palette,
                    offset=[ob.bbox[0]-oa.bbox[0],ob.bbox[1]-oa.bbox[1]],stationary_cells=oa.cells==ob.cells))
    # Split/merge evidence retains source coordinates. These are overlap links,
    # not proof that an arbitrary group forms the intended semantic object.
    if (len(x),len(x[0]))==(len(y),len(y[0])):
        for j,ob in enumerate(b):
            ids=[i for i,oa in enumerate(a) if oa.cells&ob.cells]
            if len(ids)>1:result['merges'].append(dict(sources=ids,target=j))
        for i,oa in enumerate(a):
            ids=[j for j,ob in enumerate(b) if oa.cells&ob.cells]
            if len(ids)>1:result['splits'].append(dict(source=i,targets=ids))
    return result


def analyze(pairs):
    reports=[]
    for x,y in pairs:
        policies=list(dict.fromkeys(guess_backgrounds(x,2)+guess_backgrounds(y,1)))
        reports.append([match(x,y,bg,conn,multi) for bg in policies for conn in (4,8) for multi in (True,False)])
    return reports


def movement_hints(pairs):
    """Exact-mask hypotheses shared across demonstrations; palette matches first.

    Colour-changing links are proposals only. Complete compositions still must
    reproduce every demonstration exactly before being accepted.
    """
    preserved_offsets=preserved_ops=all_offsets=all_ops=None
    for x,y in pairs:
        po=set();pg=set();ao=set();ag=set()
        for bg in guess_backgrounds(x,2):
            for multi in (True,False):
                for link in match(x,y,bg,4,multi)['links']:
                    ao.add(tuple(link['offset']));ag.update(link['geometries'])
                    if link['palette_preserved']:
                        po.add(tuple(link['offset']));pg.update(link['geometries'])
        preserved_offsets=po if preserved_offsets is None else preserved_offsets&po
        preserved_ops=pg if preserved_ops is None else preserved_ops&pg
        all_offsets=ao if all_offsets is None else all_offsets&ao
        all_ops=ag if all_ops is None else all_ops&ag
    po=preserved_offsets or set();pg=preserved_ops or set()
    return dict(offsets=sorted(po)+sorted((all_offsets or set())-po),
                geometries=sorted(pg)+sorted((all_ops or set())-pg))
