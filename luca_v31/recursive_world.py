"""New replacement: recolour objects using their internal features; not recovered original code."""
from .object_support import CONFIGS, view, features, accepted
from .grid_world import grid_shape
MISSING = False

def fit_all_recursive_traced(pairs):
    from .budget import check as _budget_check
    _budget_check()
    if not pairs or any(grid_shape(x)!=grid_shape(y) for x,y in pairs): return []
    laws=[]
    for config in CONFIGS:
        observations=[];valid=True
        for x,y in pairs:
            bg,objs=view(x,config)
            if not objs or any(x[r][c]==bg and y[r][c]!=bg for r in range(len(x)) for c in range(len(x[0]))):
                valid=False;break
            for obj in objs:
                colors={y[r][c] for r,c in obj.cells}
                if len(colors)!=1:valid=False;break
                observations.append((features(obj,bg),next(iter(colors))))
            if not valid:break
        if not valid:continue
        for feature in ('holes','hole_area','colors','bbox_empty','area','height','width'):
            mapping={}
            for f,col in observations:
                key=f[feature]
                if key in mapping and mapping[key]!=col:break
                mapping[key]=col
            else:
                if len(mapping)<2:continue
                def predict(grid,config=config,feature=feature,mapping=dict(mapping)):
                    bg,objs=view(grid,config)
                    if not objs:raise ValueError('no objects')
                    out=[list(row) for row in grid]
                    for obj in objs:
                        key=features(obj,bg)[feature]
                        if key not in mapping:raise ValueError('unseen internal feature value')
                        for r,c in obj.cells:out[r][c]=mapping[key]
                    return tuple(map(tuple,out))
                law=accepted('internal_'+feature+'_'+str(config),predict,pairs,
                             f'VIEW(object) -> {feature} -> color lookup {mapping}; view={config}',bits=len(mapping),ops=2)
                if law:laws.append((law,fit_all_recursive))
    return laws

def fit_all_recursive(pairs):return [law for law,_ in fit_all_recursive_traced(pairs)]
