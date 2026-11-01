"""Cooperative budget, never caught by legacy `except Exception` clauses.
Checks occur at fitter/search boundaries. This is not a preemptive OS deadline.
"""
from contextvars import ContextVar
from time import monotonic
_state=ContextVar('luca_v27_budget',default=None)
class Exhausted(BaseException):pass

def begin(seconds):
 if seconds<0:raise ValueError('negative budget')
 now=monotonic();return _state.set(dict(start=now,deadline=now+seconds,seconds=seconds,checks=0,laws=[]))
def end(token):_state.reset(token)
def check():
 s=_state.get()
 if s is not None:
  s['checks']+=1
  if monotonic()>=s['deadline']:raise Exhausted()
def remember(laws):
 s=_state.get()
 if s is not None:s['laws']=sorted(laws,key=lambda law:10*law.op_count+2*law.constant_bits)
def candidates():
 s=_state.get();return list(s['laws']) if s else []
def stats():
 s=_state.get()
 return {} if s is None else dict(allowance_seconds=s['seconds'],elapsed_seconds=round(monotonic()-s['start'],6),checks=s['checks'],retained_candidates=len(s['laws']),exhausted=monotonic()>=s['deadline'])

def stop_checks():_state.set(None)

def retain_result(result):
 s=_state.get()
 if s is not None:s['result']=result

def last_result():
 s=_state.get();return s.get('result') if s is not None else None
