"""Explicit validation-only stopping heuristics; never alter the training loss."""
KEYS=[f'val/{d}/{h}_loss' for d in ('original','dagger') for h in ('action','tactile')]

class ValidationGuard:
    def __init__(self, baseline, min_steps=3721, patience=4):
        self.baseline={k:baseline[k] for k in KEYS}
        self.best=None
        self.best_step=0
        self.stale=0
        self.retention_bad=0
        self.min_steps=min_steps
        self.patience=patience

    def update(self, step, metrics):
        # A modest 5% guard band is a screening heuristic, not demonstrated
        # robot retention. Both original heads must remain within the band.
        eligible=all(metrics[k]<=1.05*self.baseline[k] for k in KEYS[:2])
        # Require meaningful action improvement without sacrificing tactile
        # correction loss by more than the guard band; no aggregate loss.
        improved=eligible and (self.best is None or (
            metrics[KEYS[2]]<.99*self.best[KEYS[2]] and
            metrics[KEYS[3]]<=1.05*self.best[KEYS[3]]))
        if improved:
            self.best={k:metrics[k] for k in KEYS};self.best_step=step;self.stale=0
        else:self.stale+=1
        severely_worse=any(metrics[k]>1.20*self.baseline[k] for k in KEYS[:2])
        self.retention_bad=self.retention_bad+1 if severely_worse else 0
        reason=None
        if step>=1000 and self.retention_bad>=2:
            reason='original_retention_loss_over_20_percent_twice'
        elif step>=self.min_steps and self.stale>=self.patience:
            reason='no_material_guarded_dagger_improvement_for_four_validations_after_full_coverage'
        return dict(stop=reason is not None,reason=reason,
                    eligible_original_guard=eligible,significant_improvement=improved,
                    best_guarded_step=self.best_step,stale_validations=self.stale,
                    consecutive_severe_retention_failures=self.retention_bad,
                    minimum_coverage_steps=self.min_steps)
