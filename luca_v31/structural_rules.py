"""Small structural programs fitted only to demonstrations, without task IDs.

All fits are exact. Multiple compatible models must agree on a prediction.
Omitted-demonstration validation is mandatory when at least three pairs exist.
"""
from collections import Counter
from dataclasses import replace
from itertools import product, permutations
from .primitives import Law

def grid(g): return tuple(map(tuple,g))

def components(g,bg):
    h,w=len(g),len(g[0]); seen=set(); out=[]
    for r in range(h):
        for c in range(w):
            if g[r][c]==bg or (r,c) in seen: continue
            colour=g[r][c]; stack=[(r,c)]; seen.add((r,c)); pts=[]
            while stack:
                a,b=stack.pop();pts.append((a,b))
                for x,y in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                    if 0<=x<h and 0<=y<w and (x,y) not in seen and g[x][y]==colour:
                        seen.add((x,y));stack.append((x,y))
            out.append((colour,pts))
    return out

def predict(g,m):
    name,bg,*args=m; h,w=len(g),len(g[0]); colours=set(v for row in g for v in row)
    out=[[bg]*w for _ in range(h)]
    if name=='count_to_row':
        if len(colours-{bg})!=1:return None
        col=next(iter(colours-{bg}));n=sum(v==col for row in g for v in row)
        return ((col,)*n,) if 0<n<=30 else None
    if name=='prefix_staircase':
        if h!=1 or w%2 or len(colours-{bg})!=1:return None
        col=next(iter(colours-{bg}));n=sum(v==col for v in g[0])
        if list(g[0])!=[col]*n+[bg]*(w-n):return None
        return tuple(tuple(col if c<n+r else bg for c in range(w)) for r in range(w//2))
    if name=='translate_rebackground':
        newbg,dr,dc=args
        if len(colours-{bg})!=1 or newbg in colours-{bg}:return None
        out=[[newbg]*w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if g[r][c]!=bg and 0<=r+dr<h and 0<=c+dc<w:out[r+dr][c+dc]=g[r][c]
        return grid(out)
    if name=='colour_axis_lines':
        assignment,order=args
        if colours-{bg}!=set(c for c,axis in assignment):return None
        out=[list(row) for row in g];axes=dict(assignment)
        for col in order:
            pts=[(r,c) for r in range(h) for c in range(w) if g[r][c]==col]
            for r,c in pts:
                if axes[col]=='row':
                    for k in range(w):out[r][k]=col
                else:
                    for k in range(h):out[k][c]=col
        return grid(out)
    if name=='four_arm_gap_mark':
        source,target,distance=args
        if colours-{bg}!={source}:return None
        out=[list(row) for row in g]
        for r in range(h):
            for c in range(w):
                if g[r][c]!=bg:continue
                ok=True
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    for k in range(1,distance+2):
                        a,b=r+k*dr,c+k*dc
                        required=bg if k<distance else source
                        if not (0<=a<h and 0<=b<w and g[a][b]==required):ok=False;break
                    if not ok:break
                if ok:out[r][c]=target
        return grid(out)
    if name=='outline_corner_classification':
        convex,concave=args
        if len(colours-{bg})!=1:return None
        source=next(iter(colours-{bg}));outside=set();stack=[]
        for r in range(h):
            for c in range(w):
                if (r in (0,h-1) or c in (0,w-1)) and g[r][c]==bg:outside.add((r,c));stack.append((r,c))
        while stack:
            r,c=stack.pop()
            for a,b in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0<=a<h and 0<=b<w and g[a][b]==bg and (a,b) not in outside:outside.add((a,b));stack.append((a,b))
        out=[list(row) for row in g]
        for r in range(h):
            for c in range(w):
                if g[r][c]!=source:continue
                ns=[(dr,dc) for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)) if 0<=r+dr<h and 0<=c+dc<w and g[r+dr][c+dc]==source]
                if len(ns)!=2 or ns[0][0]==-ns[1][0] and ns[0][1]==-ns[1][1]:continue
                a=r+ns[0][0]+ns[1][0];b=c+ns[0][1]+ns[1][1]
                if not(0<=a<h and 0<=b<w) or g[a][b]!=bg:continue
                out[r][c]=concave if (a,b) in outside else convex
        return grid(out)
    if name in ('solid_2x2_opening','largest_solid_rectangle'):
        rectangles=[]
        for r in range(h-1):
            for c in range(w-1):
                col=g[r][c]
                if col==bg:continue
                if name=='solid_2x2_opening':
                    if g[r+1][c]==g[r][c+1]==g[r+1][c+1]==col:
                        for a,b in ((r,c),(r+1,c),(r,c+1),(r+1,c+1)):out[a][b]=col
                else:
                    right=w
                    for bottom in range(r,h):
                        k=c
                        while k<right and g[bottom][k]==col:k+=1
                        right=k
                        if right-c<2:break
                        if bottom>r:rectangles.append(((bottom-r+1)*(right-c),r,c,bottom,right,col))
        if name=='largest_solid_rectangle':
            if not rectangles:return None
            size=max(x[0] for x in rectangles);best={x for x in rectangles if x[0]==size}
            if len(best)!=1:return None
            _,r,c,bottom,right,col=next(iter(best))
            for a in range(r,bottom+1):
                for b in range(c,right):out[a][b]=col
        return grid(out)
    comps=components(g,bg)
    if not comps:return None
    if name=='nearest_marker_shape_recolour':
        sizes=sorted((len(p) for col,p in comps),reverse=True)
        if sizes[0]<2 or (len(sizes)>1 and sizes[1]!=1):return None
        shape=next(p for col,p in comps if len(p)==sizes[0]);markers=[(col,p[0]) for col,p in comps if len(p)==1]
        if not markers:return None
        distances=[(min((r-a)**2+(c-b)**2 if args[0]=='squared_euclidean' else abs(r-a)+abs(c-b) for r,c in shape),col) for col,(a,b) in markers]
        d=min(x[0] for x in distances);cs={col for dist,col in distances if dist==d}
        if len(cs)!=1:return None
        col=next(iter(cs))
        for r,c in shape:out[r][c]=col
        return grid(out)
    if name=='bottom_half_bar_recolour':
        source,target=args
        if colours-{bg}!={source}:return None
        out=[list(row) for row in g]
        for col,pts in comps:
            rows=[r for r,c in pts];cols={c for r,c in pts}
            if len(cols)!=1 or max(rows)!=h-1 or len(rows)!=h-min(rows):return None
            c=next(iter(cols))
            for r in range(h-len(rows)//2,h):out[r][c]=target
        return grid(out)
    if name=='alternating_rectangle_rings':
        source,odd,even=args
        if colours-{bg}!={source}:return None
        out=[list(row) for row in g]
        for col,pts in comps:
            top=min(r for r,c in pts);bottom=max(r for r,c in pts);left=min(c for r,c in pts);right=max(c for r,c in pts)
            if len(pts)!=(bottom-top+1)*(right-left+1):return None
            for r,c in pts:
                depth=min(r-top,bottom-r,c-left,right-c)
                if depth:out[r][c]=odd if depth%2 else even
        return grid(out)
    return None

def candidates(pairs):
    xs=[x for x,y in pairs];ys=[y for x,y in pairs]
    bgs=set.intersection(*[set(v for row in x for v in row) for x in xs])
    for bg in sorted(bgs):
        for name in ('count_to_row','prefix_staircase','solid_2x2_opening','largest_solid_rectangle'):
            yield (name,bg)
        for metric in ('squared_euclidean','manhattan'):yield ('nearest_marker_shape_recolour',bg,metric)
        cs=set(v for row in xs[0] for v in row)-{bg}
        if 2<=len(cs)<=3 and all(len(x)==len(y) and len(x[0])==len(y[0]) for x,y in pairs):
            # Every source pixel must be an isolated marker.
            if all(all(len(pts)==1 for col,pts in components(x,bg)) for x in xs):
                for axes in product(('row','column'),repeat=len(cs)):
                    for order in permutations(sorted(cs)):yield ('colour_axis_lines',bg,tuple(zip(sorted(cs),axes)),order)
        if len(cs)==1:
            source=next(iter(cs)); targets=set(v for y in ys for row in y for v in row)-{bg,source}
            for target in sorted(targets):yield ('bottom_half_bar_recolour',bg,source,target)
            for target in sorted(targets):
                for distance in (2,3,4):yield ('four_arm_gap_mark',bg,source,target,distance)
            for odd in sorted(targets):
                for even in sorted(targets-{odd}):yield ('alternating_rectangle_rings',bg,source,odd,even)
            newcolours=set(v for y in ys for row in y for v in row)-set(v for x in xs for row in x for v in row)
            for convex in sorted(newcolours):
                for concave in sorted(newcolours-{convex}):yield ('outline_corner_classification',bg,convex,concave)
        outcs=set.intersection(*[set(v for row in y for v in row) for y in ys])-{bg}
        for newbg in sorted(outcs):
            for dr,dc in ((0,-1),(0,1),(-1,0),(1,0)):
                yield ('translate_rebackground',bg,newbg,dr,dc)

def fits(pairs):
    if len(pairs)<2 or all(grid(x)==grid(y) for x,y in pairs):return []
    return [m for m in candidates(pairs) if all(predict(x,m)==grid(y) for x,y in pairs)]

def induce(pairs):
    models=fits(pairs)
    if not models:return []
    if len(pairs)>=3:
        # Validate each structural family with parameters relearned independently.
        names={m[0] for m in models}
        for i,(x,y) in enumerate(pairs):
            others=fits(pairs[:i]+pairs[i+1:]);valid=set()
            for name in names:
                ps={predict(x,m) for m in others if m[0]==name}
                if ps=={grid(y)}:valid.add(name)
            names &= valid
        models=[m for m in models if m[0] in names]
    return models

def upgrade_many(train_pairs,test_inputs,results):
    models=induce(list(train_pairs))
    if not models:return results
    upgraded=[]
    for g,result in zip(test_inputs,results):
        choices={predict(g,m) for m in models}
        if None in choices or len(choices)!=1:upgraded.append(result);continue
        prediction=next(iter(choices));model=models[0]
        if result.predictions and grid(result.predictions[0])==prediction:upgraded.append(result);continue
        law=Law(name='structural:'+model[0],predict=lambda x,m=model:predict(x,m),description='Exact demonstration-fitted structural rule with omitted-pair checks and prediction agreement.',constant_bits=16,op_count=5)
        law.program_spec=dict(op='structural_rules',parameters=model);law.origin='structural_rules'
        predictions=[prediction];laws=[law]
        if result.predictions:
            predictions.append(result.predictions[0])
            if result.laws:laws.append(result.laws[0])
        diagnostics=dict(result.diagnostics or {});diagnostics['structural_rules']=dict(models=models,exact_demonstrations=len(train_pairs),omitted_pair_checks=len(train_pairs) if len(train_pairs)>=3 else 0)
        upgraded.append(replace(result,predictions=predictions,laws=laws,failure_category=None,diagnostics=diagnostics,explanation=result.explanation+' | Demonstration-fitted structural rule supplied attempt 1.'))
    return upgraded
