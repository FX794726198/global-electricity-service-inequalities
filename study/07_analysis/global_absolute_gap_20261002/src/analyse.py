"""Absolute annual-burden accounting on the unchanged benchmark population."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
ROOT = P.parents[1]
for folder in ['data', 'metadata', 'reports', 'figures', 'source_data']:
    (P / folder).mkdir(exist_ok=True)
src = ROOT / '10_manuscript/20261002_global_blackout_inequalities_v15/source_data/Fig07_country_inequality.csv'
a = pd.read_csv(src)
assert len(a) == 152 and a.ISO3.is_unique
assert a.groupby('gdp_quartile').size().eq(38).all()
assert np.allclose(a.saifi * a.caidi_h, a.saidi_h)
assert np.array_equal(pd.qcut(a.gdp_ppp_pc_at_year, 4, labels=['Q1','Q2','Q3','Q4']).astype(str), a.gdp_quartile)
cols = ['saifi', 'caidi_h', 'saidi_h']
assert a[cols].gt(0).all().all()
a.to_csv(P / 'source_data/country_benchmarks.csv', index=False)
rng = np.random.default_rng(20261002)
B = 4000

def summarize(x):
    f, d, h = x.mean(axis=-2).T
    cov = (x[..., 0] * x[..., 1]).mean(axis=-1) - f * d
    return np.array([f, h, h / f, d, cov]).T

names = ['frequency', 'annual_hours', 'interruption_weighted_duration_h',
         'equal_benchmark_duration_h', 'frequency_duration_covariance']
point, boot, rows = {}, {}, []
for q, z in a.groupby('gdp_quartile', sort=True):
    x = z[cols].to_numpy()
    point[q] = summarize(x)
    boot[q] = summarize(x[rng.integers(0, len(x), (B, len(x)))])
    for k, label in enumerate(names):
        lo, hi = np.quantile(boot[q][:, k], [.025, .975])
        rows.append(dict(quartile=q, metric=label, estimate=point[q][k],
                         ci_low=lo, ci_high=hi, n=len(z)))
pd.DataFrame(rows).to_csv(P / 'data/arithmetic_group_summary.csv', index=False)

def decompose(l, h):
    cf = (l[...,0] - h[...,0]) * (l[...,2] + h[...,2]) / 2
    cd = (l[...,2] - h[...,2]) * (l[...,0] + h[...,0]) / 2
    delta = l[...,1] - h[...,1]
    return np.array([cf, cd, delta, cf/delta, cd/delta]).T

values = decompose(point['Q1'], point['Q4'])
draws = decompose(boot['Q1'], boot['Q4'])
labels = ['frequency_component_h', 'duration_component_h', 'annual_gap_h',
          'frequency_fraction', 'duration_fraction']
rows = []
for j, label in enumerate(labels):
    lo, hi = np.quantile(draws[:, j], [.025, .975])
    rows.append(dict(metric=label, estimate=values[j], ci_low=lo, ci_high=hi))
pd.DataFrame(rows).to_csv(P / 'data/absolute_decomposition.csv', index=False)
pd.DataFrame(draws, columns=labels).to_csv(P / 'data/paired_bootstrap.csv', index=False)
checks = {
    'fixed_152_unique_economies': True,
    'fixed_38_per_group': True,
    'native_row_identity': bool(np.allclose(a.saifi*a.caidi_h, a.saidi_h)),
    'point_factor_identity': all(np.isclose(x[0]*x[2], x[1]) for x in point.values()),
    'bootstrap_factor_identity': all(np.allclose(x[:,0]*x[:,2],x[:,1]) for x in boot.values()),
    'point_covariance_identity': all(np.isclose(x[0]*x[3]+x[4],x[1]) for x in point.values()),
    'bootstrap_covariance_identity': all(np.allclose(x[:,0]*x[:,3]+x[:,4],x[:,1]) for x in boot.values()),
    'absolute_decomposition_identity': bool(np.isclose(values[0]+values[1],values[2])),
    'bootstrap_decomposition_identity': bool(np.allclose(draws[:,0]+draws[:,1],draws[:,2])),
    'finite_draws': bool(np.isfinite(draws).all()),
}
assert all(checks.values())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
meta = dict(checks=checks, seed=20261002, replicates=B,
            nonpositive_gap_draws=int((draws[:,2]<=0).sum()),
            plan_sha256=sha(P/'ANALYSIS_PLAN.md'), script_sha256=sha(Path(__file__)),
            input_path=str(src), input_sha256=sha(src),
            units='Equally weighted source-defined service benchmarks; no national customer weights',
            status='PASS')
(P/'metadata/validation.json').write_text(json.dumps(meta, indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
print(pd.read_csv(P/'data/arithmetic_group_summary.csv').to_string(index=False))
