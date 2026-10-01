"""Q1 conflict resolution: local, text-type conditional arbitration of conflicting evaluators.

A conflict on pair (i,j) means z_i >= 0.8 and z_j <= 0.2 on the fixed A1 scale.
Non-conflict samples keep the equal-weight Q. For a conflict sample of text type t
the pair keeps its total weight 2/22 and splits it with share s_i (s_j = 1 - s_i):
    Q_res = Q_equal + (2/22) * [ s_i z_i + s_j z_j - (z_i + z_j)/2 ].
This keeps the common cross-type scale, so domain Q and Q0 stay comparable.

Main rule (information weighting, revised after the manual text check):
    I_i(t) = -ln P_t(z_i >= 0.8),  I_j(t) = -ln P_t(z_j <= 0.2),  s_i = I_i / (I_i + I_j),
estimated per text type on A1. A verdict that an evaluator gives to a large share of a
text type carries little information about any single document of that type; e.g. a
prose-trained cleanliness model puts 30% of all code at the floor, so its low score on
a code file is weak evidence. The rule uses no manual labels.
Pairs that measure different dimensions are not arbitrated: educational value vs
advertising (knowledge content vs commercial intent) keeps equal weights.

Comparisons kept for the record:
  v1 reliability arbitration, s_i = rho_i/(rho_i+rho_j), rho = max(corr(z_i, consensus_{-i}), 0);
     the manual text check contradicted its direction on both code and advertising pairs.
  global type-specific reweighting of all samples; breaks cross-type comparability.
Parameters are estimated on A1 and applied unchanged to A2/A3; A2/A3 re-estimates
are a stability check (A1 arxiv/github rows are contained in A2/A3).
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / '03_src'))
sys.path.insert(0, str(PROJECT / '03_src/q1'))
from fmodel.quality import score_quality, semantic_conflict
import quality_baseline as qb

TYPES = {'github': 'code', 'arxiv': 'science', 'stackexchange': 'qa',
         'book': 'prose', 'c4': 'prose', 'commoncrawl': 'prose', 'wikipedia': 'prose'}
PAIRS = {'education_vs_no_ad': ('fineweb_edu', 'ad_en'),
         'qurater_vs_cleanliness': ('qurater', 'modernbert_cleanliness'),
         'reasoning_vs_readability': ('modernbert_reasoning', 'modernbert_readability'),
         'fluency_vs_cleanliness': ('fluency_en', 'modernbert_cleanliness')}
DIFFERENT_DIMENSION = {'education_vs_no_ad'}
HIGH, LOW, FLOOR_P = 0.8, 0.2, 1e-4
KEY = ['fluency_en', 'modernbert_cleanliness', 'modernbert_reasoning', 'modernbert_readability',
       'fineweb_edu', 'ad_en', 'qurater']


def group_of(name):
    return 'dsir' if name in qb.DSIR else 'model' if name in qb.MODEL else 'natural'


def normalized(sources):
    """Rebuild the fixed-scale normalized indicators exactly as quality_baseline does."""
    base = sources['A1'][0]
    word_col = qb.FIELDS.index('rps_doc_mean_word_length')
    anchor = float(np.nanmedian(base[:, word_col]))
    for x, _, _ in sources.values():
        x[:, word_col] = -np.abs(x[:, word_col] - anchor)
    lo, hi = np.nanquantile(sources['A1'][0], [.01, .99], axis=0)
    hi = np.where(hi <= lo, lo + 1, hi)
    settings = [qb.Indicator(n, -1 if n in qb.REVERSE else 1, float(lo[j]), float(hi[j]), 1)
                for j, n in enumerate(qb.FIELDS)]
    out = {}
    for code, (x, ids, domains) in sources.items():
        r = score_quality(x, settings, min_coverage=.8, clip=True)
        out[code] = (r['normalized'], r['scores'], ids, domains)
    return out


def reliability(z):
    """Correlation of each indicator with the group-balanced consensus of the others."""
    groups = np.array([group_of(n) for n in qb.FIELDS])
    r = np.zeros(z.shape[1])
    for i in range(z.shape[1]):
        parts = []
        for g in ['natural', 'dsir', 'model']:
            cols = [j for j in np.flatnonzero(groups == g) if j != i]
            parts.append(np.nanmean(z[:, cols], axis=1))
        consensus = np.nanmean(np.column_stack(parts), axis=1)
        ok = np.isfinite(z[:, i]) & np.isfinite(consensus)
        r[i] = np.corrcoef(z[ok, i], consensus[ok])[0, 1] if z[ok, i].std() > 0 else 0.0
    w = np.maximum(r, 0)
    return r, w / w.sum()


def information(z):
    """Per indicator: -ln P(z >= HIGH) and -ln P(z <= LOW) within one text type."""
    ok = np.isfinite(z)
    p_high = np.array([np.mean(z[ok[:, i], i] >= HIGH) for i in range(z.shape[1])])
    p_low = np.array([np.mean(z[ok[:, i], i] <= LOW) for i in range(z.shape[1])])
    return -np.log(np.maximum(p_high, FLOOR_P)), -np.log(np.maximum(p_low, FLOOR_P)), p_high, p_low


def weighted_q(z, w):
    ok = np.isfinite(z)
    return np.nansum(z * w, axis=1) / (ok * w).sum(axis=1)


def main():
    sources = {code: qb.read_source(code, path) for code, path in qb.FILES.items()}
    data = normalized(sources)
    z1, _, _, d1 = data['A1']
    t1 = np.array([TYPES[d] for d in d1])
    weights, corr, info = {}, {}, {}
    for t in sorted(set(t1)):
        corr[t], weights[t] = reliability(z1[t1 == t])
        info[t] = information(z1[t1 == t])
    rho = {t: np.maximum(corr[t], 0) for t in corr}
    n_ind = len(qb.FIELDS)

    def share(rule, name, t, ih, il):
        if rule == 'information':
            if name in DIFFERENT_DIMENSION:
                return .5
            a, b = info[t][0][ih], info[t][1][il]
        else:
            a, b = rho[t][ih], rho[t][il]
        return .5 if a + b == 0 else a / (a + b)

    def arbitrate(code, rule):
        z, q_old, _, domains = data[code]
        types = np.array([TYPES[d] for d in domains])
        w = np.full(z.shape, 1 / n_ind)
        any_conflict = np.zeros(len(q_old), bool)
        for name, (hi_name, lo_name) in PAIRS.items():
            ih, il = qb.FIELDS.index(hi_name), qb.FIELDS.index(lo_name)
            flag = semantic_conflict(z, ih, il)['conflict']
            any_conflict |= flag
            for t in set(types[flag]):
                m = flag & (types == t)
                s = share(rule, name, t, ih, il)
                total = w[m, ih] + w[m, il]
                w[m, ih], w[m, il] = total * s, total * (1 - s)
        return weighted_q(z, w), any_conflict

    def global_reweight(code):
        z, q_old, _, domains = data[code]
        types = np.array([TYPES[d] for d in domains])
        q = np.full(len(q_old), np.nan)
        for t in set(types):
            m = types == t
            q[m] = weighted_q(z[m], weights[t])
        return q

    rows, domain_table, conflict_table = {}, [], []
    for code in ['A1', 'A2', 'A3']:
        z, q_old, ids, domains = data[code]
        q_info, anyc = arbitrate(code, 'information')
        q_rel, _ = arbitrate(code, 'reliability')
        q_glob = global_reweight(code)
        rows[code] = pd.DataFrame({'source': code, 'id': ids, 'domain': domains, 'Q_equal': q_old,
                                   'Q_resolved': q_info, 'Q_resolved_v1_reliability': q_rel,
                                   'Q_global_reweight': q_glob, 'any_conflict': anyc})
        for d in sorted(set(domains)):
            m = domains == d
            domain_table.append({'source': code, 'domain': d, 'type': TYPES[d], 'n': int(m.sum()),
                                 'n_conflict_any': int(anyc[m].sum()),
                                 'Q_equal': float(np.nanmean(q_old[m])), 'Q_resolved': float(np.nanmean(q_info[m])),
                                 'Q_resolved_v1_reliability': float(np.nanmean(q_rel[m])),
                                 'Q_global_reweight': float(np.nanmean(q_glob[m])),
                                 'spearman_equal_vs_global': float(spearmanr(q_old[m], q_glob[m]).correlation)})
        for name, (hi_name, lo_name) in PAIRS.items():
            ih, il = qb.FIELDS.index(hi_name), qb.FIELDS.index(lo_name)
            flag = semantic_conflict(z, ih, il)['conflict']
            for d in sorted(set(domains[flag])):
                m, c = domains == d, flag & (domains == d)
                if c.sum() < 5:
                    continue
                t = TYPES[d]
                s_info, s_rel = share('information', name, t, ih, il), share('reliability', name, t, ih, il)
                pct = lambda q: pd.Series(q[m]).rank(pct=True).to_numpy()
                conflict_table.append({
                    'source': code, 'conflict': name, 'domain': d, 'type': t, 'n_conflict': int(c.sum()),
                    'rate': float(c.sum() / m.sum()),
                    'arbitrated': name not in DIFFERENT_DIMENSION,
                    'p_high_verdict': float(info[t][2][ih]), 'p_low_verdict': float(info[t][3][il]),
                    'info_high': float(info[t][0][ih]), 'info_low': float(info[t][1][il]),
                    'share_high': float(s_info),
                    'reliability_high': float(corr[t][ih]), 'reliability_low': float(corr[t][il]),
                    'share_high_v1_reliability': float(s_rel),
                    'Q_conflict_equal': float(np.nanmean(q_old[c])),
                    'Q_conflict_resolved': float(np.nanmean(q_info[c])),
                    'Q_conflict_v1_reliability': float(np.nanmean(q_rel[c])),
                    'Q_nonconflict_domain': float(np.nanmean(q_old[m & ~flag])),
                    'rank_shift_median_pct': float(np.nanmedian(np.abs(pct(q_info) - pct(q_old))[c[m]]))})

    # stability: re-estimate on A3 (code) and A2 (science) and compare the arbitration shares
    stability = {}
    for code, t in [('A3', 'code'), ('A2', 'science')]:
        z_ext = data[code][0]
        r_ext, _ = reliability(z_ext)
        info_ext = information(z_ext)
        pairs = {}
        for name, (hi_name, lo_name) in PAIRS.items():
            ih, il = qb.FIELDS.index(hi_name), qb.FIELDS.index(lo_name)
            a, b = info_ext[0][ih], info_ext[1][il]
            pairs[name] = {'share_high_A1': float(share('information', name, t, ih, il)),
                           'share_high_ext': .5 if name in DIFFERENT_DIMENSION else float(a / (a + b)),
                           'p_low_verdict_A1': float(info[t][3][il]), 'p_low_verdict_ext': float(info_ext[3][il])}
        stability[code] = {'type': t, 'pairs': pairs,
                           'reliability_vector_correlation': float(np.corrcoef(corr[t], r_ext)[0, 1])}

    report = {'method': ('local conflict arbitration by within-type information of each verdict (A1-estimated, '
                         'fixed for A2/A3); educational value vs advertising not arbitrated'),
              'revision': ('v1 used consensus reliability; the manual text check of 27 samples contradicted its '
                           'direction on code and advertising conflicts, so the rule was replaced by information '
                           'weighting, which uses no manual labels'),
              'types': TYPES, 'thresholds': {'high': HIGH, 'low': LOW},
              'different_dimension_pairs': sorted(DIFFERENT_DIMENSION),
              'verdict_probabilities': {t: {'p_high': dict(zip(qb.FIELDS, map(float, info[t][2]))),
                                            'p_low': dict(zip(qb.FIELDS, map(float, info[t][3])))} for t in info},
              'consensus_correlation': {t: dict(zip(qb.FIELDS, map(float, r))) for t, r in corr.items()},
              'domain_scores': domain_table, 'conflicts': conflict_table, 'stability': stability}
    out = PROJECT / '05_results/tables/q1_conflict_resolution.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.concat(rows.values(), ignore_index=True).to_parquet(
        PROJECT / '02_data/processed/q1_quality_scores_reliability.parquet', index=False)

    for r in domain_table:
        print('  %s %-13s n=%6d conflicts=%5d Q %.4f -> %.4f (v1 %.4f) | global %.3f' % (
            r['source'], r['domain'], r['n'], r['n_conflict_any'], r['Q_equal'], r['Q_resolved'],
            r['Q_resolved_v1_reliability'], r['Q_global_reweight']))
    for r in conflict_table:
        print('  %s %-25s %-12s n=%5d P(hi)=%.3f P(lo)=%.3f share_hi %.2f (v1 %.2f) Q %.3f -> %.3f (v1 %.3f)' % (
            r['source'], r['conflict'], r['domain'], r['n_conflict'], r['p_high_verdict'], r['p_low_verdict'],
            r['share_high'], r['share_high_v1_reliability'], r['Q_conflict_equal'], r['Q_conflict_resolved'],
            r['Q_conflict_v1_reliability']))
    for k, v in stability.items():
        print(' stability', k, {n: (round(x['share_high_A1'], 2), round(x['share_high_ext'], 2)) for n, x in v['pairs'].items()})


if __name__ == '__main__':
    main()
