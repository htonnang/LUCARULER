"""Demonstration-fitted topology, connectivity and region-selection programs.

No task identifiers or test target grids are consulted. Ambiguous predictions
are rejected, and omitted-pair validation is required with >=3 demonstrations.
"""
from collections import Counter
from dataclasses import replace
from .primitives import Law
from .structural_rules import grid,components

def regions(mask,diagonal=False):
    h,w=len(mask),len(mask[0]);seen=set();out=[]
    directions=[(1,0),(-1,0),(0,1),(0,-1)]
    if diagonal:directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
    for r in range(h):
        for c in range(w):
            if not mask[r][c] or (r,c) in seen:continue
            stack=[(r,c)];seen.add((r,c));pts=[]
            while stack:
                a,b=stack.pop();pts.append((a,b))
                for dr,dc in directions:
                    x,y=a+dr,b+dc
                    if 0<=x<h and 0<=y<w and mask[x][y] and (x,y) not in seen:
                        seen.add((x,y));stack.append((x,y))
            out.append(pts)
    return out

def bounds(pts):
    return min(r for r,c in pts),min(c for r,c in pts),max(r for r,c in pts),max(c for r,c in pts)

def normalise(pts):
    a,b,x,y=bounds(pts);return tuple(sorted((r-a,c-b) for r,c in pts))

def hollow_colour(g,bg):
    h,w=len(g),len(g[0]);answer=set()
    for colour,pts in components(g,bg):
        occupied=set(pts)
        empty=regions([[(r,c) not in occupied for c in range(w)] for r in range(h)])
        if any(all(r not in (0,h-1) and c not in (0,w-1) for r,c in region) for region in empty):answer.add(colour)
    return next(iter(answer)) if len(answer)==1 else None

def marker_region_winner(g):
    h,w=len(g),len(g[0]);votes={}
    for r in range(1,h-1):
        for c in range(1,w-1):
            ns={g[r-1][c],g[r+1][c],g[r][c-1],g[r][c+1]}
            if len(ns)==1 and g[r][c] not in ns:
                marker=g[r][c];region=next(iter(ns));votes.setdefault(marker,Counter())[region]+=1
    if len(votes)!=1:return None
    counts=next(iter(votes.values()));winners=[c for c,n in counts.items() if n==max(counts.values())]
    return winners[0] if len(winners)==1 else None

def symmetry(g,axis):
    h,w=len(g),len(g[0])
    if axis=='horizontal':return all(g[r][c]==g[h-1-r][c] for r in range(h) for c in range(w))
    if axis=='vertical':return all(g[r][c]==g[r][w-1-c] for r in range(h) for c in range(w))
    if axis=='diagonal':return h==w and all(g[r][c]==g[c][r] for r in range(h) for c in range(w))
    if axis=='anti_diagonal':return h==w and all(g[r][c]==g[w-1-c][h-1-r] for r in range(h) for c in range(w))
    if axis=='any_reflection':return any(symmetry(g,a) for a in ('horizontal','vertical','diagonal','anti_diagonal'))
    return False

def marker_connected(g,bg,marker,diagonal):
    marker_groups=regions([[v==marker for v in row] for row in g])
    if len(marker_groups)<2:return None
    foreground=regions([[v!=bg for v in row] for row in g],diagonal)
    owners={pt:i for i,pts in enumerate(foreground) for pt in pts}
    return len({owners[pts[0]] for pts in marker_groups})==1

def predicate(g,model):
    name,*args=model
    if name=='symmetry':return symmetry(g,args[0])
    if name=='connectivity':return marker_connected(g,*args)
    return None

def predict(g,m):
    name,*args=m;h,w=len(g),len(g[0])
    if name=='predicate_label':
        pred,false,true=args;value=predicate(g,pred)
        return None if value is None else ((true if value else false,),)
    if name=='hollow_colour':
        colour=hollow_colour(g,args[0]);return None if colour is None else ((colour,),)
    if name=='marker_region_winner':
        colour=marker_region_winner(g);return None if colour is None else ((colour,),)
    if name=='panel_shape_inlay':
        bg,side=args
        if side in ('left','right'):
            if w%2:return None
            first=[row[:w//2] for row in g];second=[row[w//2:] for row in g]
        else:
            if h%2:return None
            first=g[:h//2];second=g[h//2:]
        target,donor=(first,second) if side in ('left','top') else (second,first)
        th,tw=len(target),len(target[0]);colours={v for row in target for v in row}-{bg}
        if len(colours)!=1:return None
        holes=regions([[v==bg for v in row] for row in target]);parts=components(donor,bg)
        if not holes or len(holes)!=len(parts):return None
        if any(any(r in (0,th-1) or c in (0,tw-1) for r,c in pts) for pts in holes):return None
        shapes={}
        for col,pts in parts:shapes.setdefault(normalise(pts),[]).append(col)
        out=[list(row) for row in target]
        for pts in holes:
            key=normalise(pts);choices=shapes.get(key,[])
            if not choices or len(set(choices))!=1:return None
            col=choices.pop()
            for r,c in pts:out[r][c]=col
        if any(shapes.values()):return None
        return grid(out)
    if name=='region_rectangularity':
        source,rectangle,other=args
        parts=regions([[v==source for v in row] for row in g]);out=[list(row) for row in g]
        if not parts:return None
        for pts in parts:
            a,b,x,y=bounds(pts);colour=rectangle if len(pts)==(x-a+1)*(y-b+1) else other
            for r,c in pts:out[r][c]=colour
        return grid(out)
    if name=='largest_clear_rectangle_inset':
        source,target,inset=args;best_area=0;best=set()
        for top in range(h):
            widths=[True]*w
            for bottom in range(top,h):
                widths=[v and g[bottom][c]==source for c,v in enumerate(widths)]
                c=0
                while c<w:
                    if not widths[c]:c+=1;continue
                    left=c
                    while c<w and widths[c]:c+=1
                    if bottom-top+1<=2*inset or c-left<=2*inset:continue
                    area=(bottom-top+1)*(c-left)
                    if area>best_area:best_area=area;best={(top,left,bottom,c-1)}
                    elif area==best_area:best.add((top,left,bottom,c-1))
        if len(best)!=1:return None
        a,b,x,y=next(iter(best));out=[list(row) for row in g]
        for r in range(a+inset,x-inset+1):
            for c in range(b+inset,y-inset+1):out[r][c]=target
        return grid(out)
    return None

def candidates(pairs):
    colours=set.intersection(*[{v for row in x for v in row} for x,y in pairs])
    scalar=all(len(y)==len(y[0])==1 for x,y in pairs)
    for bg in sorted(colours):
        for side in ('left','right','top','bottom'):yield ('panel_shape_inlay',bg,side)
    if scalar:
        yield ('marker_region_winner',)
        for bg in sorted(colours):yield ('hollow_colour',bg)
        predicates=[('symmetry',axis) for axis in ('horizontal','vertical','diagonal','anti_diagonal','any_reflection')]
        for bg in sorted(colours):
            for marker in sorted(colours-{bg}):
                for diagonal in (False,True):predicates.append(('connectivity',bg,marker,diagonal))
        for pred in predicates:
            mapping={};valid=True
            for x,y in pairs:
                value=predicate(x,pred)
                if value is None or value in mapping and mapping[value]!=y[0][0]:valid=False;break
                mapping[value]=y[0][0]
            if valid and set(mapping)=={False,True} and mapping[False]!=mapping[True]:yield ('predicate_label',pred,mapping[False],mapping[True])
    if all(len(x)==len(y) and len(x[0])==len(y[0]) for x,y in pairs):
        for source in sorted(colours):
            changed={y[r][c] for x,y in pairs for r in range(len(x)) for c in range(len(x[0])) if x[r][c]==source and x[r][c]!=y[r][c]}
            for rectangle in sorted(changed):
                for other in sorted(changed-{rectangle}):yield ('region_rectangularity',source,rectangle,other)
            for target in sorted(changed):
                for inset in (1,2):yield ('largest_clear_rectangle_inset',source,target,inset)

def fits(pairs):
    if len(pairs)<2 or all(grid(x)==grid(y) for x,y in pairs):return []
    return [m for m in candidates(pairs) if all(predict(x,m)==grid(y) for x,y in pairs)]

def induce(pairs):
    models=fits(pairs)
    if models and len(pairs)>=3:
        names={m[0] for m in models}
        for i,(x,y) in enumerate(pairs):
            other=fits(pairs[:i]+pairs[i+1:]);names={name for name in names if {predict(x,m) for m in other if m[0]==name}=={grid(y)}}
        models=[m for m in models if m[0] in names]
    return models

def upgrade_many(train_pairs,test_inputs,results):
    pairs=list(train_pairs);models=induce(pairs)
    if not models:return results
    out=[]
    for g,result in zip(test_inputs,results):
        predictions={predict(g,m) for m in models}
        if None in predictions or len(predictions)!=1:out.append(result);continue
        prediction=next(iter(predictions));model=models[0]
        if result.predictions and grid(result.predictions[0])==prediction:out.append(result);continue
        law=Law(name='region_reasoning:'+model[0],predict=lambda x,m=model:predict(x,m),description='Demonstration-fitted region reasoning with omitted-pair checks and agreement among surviving models.',constant_bits=16,op_count=6)
        law.program_spec=dict(op='region_reasoning',parameters=model);law.origin='region_reasoning'
        guesses=[prediction];laws=[law]
        if result.predictions:
            guesses.append(result.predictions[0])
            if result.laws:laws.append(result.laws[0])
        diagnostics=dict(result.diagnostics or {});diagnostics['region_reasoning']=dict(models=models,exact_demonstrations=len(pairs),omitted_pair_checks=len(pairs) if len(pairs)>=3 else 0)
        out.append(replace(result,predictions=guesses,laws=laws,failure_category=None,diagnostics=diagnostics,explanation=result.explanation+' | Region reasoning supplied attempt 1.'))
    return out
