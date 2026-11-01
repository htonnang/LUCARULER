"""Persistent procedure scheduling. No grids, labels or task IDs in retrieval.

Memory is explicit: load/configure once, read-only during inference. Each recalled
program must fit all current demonstrations before it can make a prediction.
"""
from collections import Counter
from contextvars import ContextVar
import json
from pathlib import Path
from .relational_programs import fit_spec, as_law
from .grid_world import guess_backgrounds, extract_objects
from .diagnostics import event

_active=ContextVar('v10_procedure_memory',default=None)
SCHEMA='luca-v10-procedures-1'


def signature(pairs):
    # Only input features. Buckets keep scheduling independent of exact inputs.
    values=[]
    for x,_ in pairs:
        bg=guess_backgrounds(x)[0]
        values.append([min(len(x)//5,6),min(len(x[0])//5,6),len({v for row in x for v in row}),
                       min(len(extract_objects(x,bg,4,True)),10)])
    return [round(sum(v[i] for v in values)/len(values),2) for i in range(4)]


def blank():return {'schema':SCHEMA,'procedures':[],'training_tasks':0}


def validate(memory):
    if memory.get('schema')!=SCHEMA or not isinstance(memory.get('procedures'),list):
        raise ValueError('unsupported procedure memory')
    if len(memory['procedures'])>1000:raise ValueError('oversized memory')
    for entry in memory['procedures']:
        if entry['spec'].get('family') not in ('object_program','copy_correspondence','move_toward'):
            raise ValueError('unsupported procedure family')
    return memory


def configure(memory=None):
    return _active.set(validate(memory) if memory is not None else None)


def reset(token):_active.reset(token)


def load(path):return validate(json.loads(Path(path).read_text()))


def remember(memory,pairs,laws):
    """Train on demonstration-verified program specs only; no test argument."""
    context=signature(pairs)
    known={json.dumps(e['spec'],sort_keys=True):e for e in memory['procedures']}
    inserted=set()
    for law in laws:
        if not hasattr(law,'program_spec'): continue
        spec=fit_spec(law.program_spec,pairs)
        if spec is None:continue
        if spec.get('palette'):spec['palette']={'refit':True}
        # Canonical JSON removes tuple/list differences across save/load.
        spec=json.loads(json.dumps(spec))
        key=json.dumps(spec,sort_keys=True)
        if key in inserted:continue
        inserted.add(key)
        if key not in known:
            known[key]={'spec':spec,'support':0,'contexts':[]}
            memory['procedures'].append(known[key])
        entry=known[key];entry['support']+=1
        if context not in entry['contexts']:entry['contexts'].append(context)
    memory['training_tasks']+=1


def recall(pairs):
    memory=_active.get()
    if not memory:return []
    context=signature(pairs)
    def score(entry):
        distance=min(sum(abs(a-b) for a,b in zip(context,v)) for v in entry['contexts'])
        return distance,-entry['support'],json.dumps(entry['spec'],sort_keys=True)
    for entry in sorted(memory['procedures'],key=score)[:64]:
        event('v10_memory_attempt')
        fitted=fit_spec(entry['spec'],pairs)
        if fitted is not None:
            event('v10_memory_hit',fitted['family'])
            return [as_law(fitted,origin='memory')]
    return []
