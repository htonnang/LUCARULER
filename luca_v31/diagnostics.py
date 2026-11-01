"""Per-prediction bounded diagnostic counters."""
from collections import Counter
from contextvars import ContextVar
import sys
import time
_current = ContextVar('luca_diagnostics', default=None)

def begin():
    return _current.set({'counts': Counter(), 'examples': [], 'started': time.monotonic()})

def event(name, detail=None):
    state = _current.get()
    if state is None: return
    state['counts'][name] += 1
    if detail is not None and len(state['examples']) < 24:
        state['examples'].append({'event': name, 'detail': str(detail)[:240]})

def record_exception(stage):
    exc = sys.exc_info()[1]
    if isinstance(exc, ValueError) and str(exc) == 'complement placement needs exactly one foreground colour':
        event('rejected_precondition:complement_foreground_count')
        return
    event('caught_exception:' + stage + ':' + type(exc).__name__, str(exc))

def snapshot():
    state = _current.get()
    if state is None: return {}
    return {'counts': dict(state['counts']), 'examples': list(state['examples']),
            'elapsed_seconds': round(time.monotonic() - state['started'], 4)}

def end(token):
    _current.reset(token)
