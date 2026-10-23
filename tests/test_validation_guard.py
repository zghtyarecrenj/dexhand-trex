from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.validation_guard import ValidationGuard,KEYS

def metric(o=1.,d=1.,t=1.):
 return {KEYS[0]:o,KEYS[1]:o,KEYS[2]:d,KEYS[3]:t}

def test_plateau_never_stops_before_full_coverage():
 g=ValidationGuard(metric())
 for step in [500,1000,1500,2000,2500,3000,3500]:
  assert not g.update(step,metric(d=.8,t=.8))['stop']
 x=g.update(3721,metric(d=.8,t=.8))
 assert x['stop'] and x['best_guarded_step']==500

def test_material_improvement_resets_patience():
 g=ValidationGuard(metric())
 for step,d in [(500,.9),(1000,.85),(1500,.8),(2000,.75),(2500,.7),(3000,.65),(3500,.6),(3721,.55)]:
  x=g.update(step,metric(d=d,t=.8));assert not x['stop'] and x['stale_validations']==0

def test_retention_guard_requires_consecutive_failures_and_recovers():
 g=ValidationGuard(metric())
 assert not g.update(500,metric(o=1.3,d=.5))['stop']
 assert not g.update(1000,metric(o=1.,d=.5))['stop']
 assert not g.update(1500,metric(o=1.3,d=.5))['stop']
 x=g.update(2000,metric(o=1.3,d=.4))
 assert x['stop'] and x['reason']=='original_retention_loss_over_20_percent_twice'

def test_lower_aggregate_cannot_hide_original_or_tactile_regression():
 g=ValidationGuard(metric())
 assert g.update(500,metric(d=.8,t=.8))['significant_improvement']
 assert not g.update(1000,metric(o=1.1,d=.1,t=.1))['significant_improvement']
 assert not g.update(1500,metric(d=.1,t=1.2))['significant_improvement']
