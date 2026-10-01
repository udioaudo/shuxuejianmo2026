"""Exploratory 22-signal quality baseline on all supplied A1/A2/A3 records.

Methods are explicit choices, not official optimal weights or clinical-style
ground truth. Preserve every record and its scoring status.
"""

from pathlib import Path
import json
import lzma
import math
import sys
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "03_src"))
from fmodel.quality import Indicator, score_quality, semantic_conflict, summarize_domains

ROOT = PROJECT / "02_data/raw/real_attachments/A_data_value"
FILES = {
    "A1": ROOT / "slimpajama_quality_signal_sample.jsonl.xz",
    "A2": next((ROOT / "slimpajama_quality_extended").glob("arxiv_*.jsonl.xz")),
    "A3": next((ROOT / "slimpajama_quality_extended").glob("github_*.jsonl.xz")),
}
META = {"id", "content", "sub_path", "_source_domain", "_source_path"}
with lzma.open(FILES["A1"], "rt", encoding="utf-8") as handle:
    FIRST = json.loads(next(handle))
FIELDS = sorted(set(FIRST) - META)
if len(FIELDS) != 22:
    raise ValueError(f"Expected exactly 22 quality signal fields, saw {len(FIELDS)}")

DSIR = {"dsir_books", "dsir_wiki", "dsir_math"}
MODEL = {"fineweb_edu", "ad_en", "fluency_en", "qurater",
         "modernbert_professionalism", "modernbert_readability",
         "modernbert_reasoning", "modernbert_cleanliness"}
NATURAL = set(FIELDS) - DSIR - MODEL
if len(DSIR) != 3 or len(MODEL) != 8 or len(NATURAL) != 11:
    raise ValueError("Unexpected quality signal groups")
REVERSE = {"rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
           "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
           "rps_lines_numerical_chars_fraction"}
PROVISIONAL = {"rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
               "rps_doc_word_count", "rps_doc_num_sentences", "qurater"}


def sigmoid(delta):
    if delta >= 0:
        return 1/(1+math.exp(-delta))
    exp = math.exp(delta)
    return exp/(1+exp)


def scalar(name, value):
    if value is None:
        return math.nan
    if name.startswith("modernbert_"):
        if not isinstance(value, list) or len(value) != 6:
            return math.nan
        v = np.asarray(value, dtype=float)
        if not np.isfinite(v).all():
            return math.nan
        probs = np.exp(v-v.max())
        probs /= probs.sum()
        return float(np.dot(np.arange(6), probs))
    if name in ("ad_en", "fluency_en"):
        if not isinstance(value, list) or len(value) != 2:
            return math.nan
        a, b = map(float, value)
        return sigmoid(b-a) if math.isfinite(a) and math.isfinite(b) else math.nan
    if name == "fineweb_edu":
        if not isinstance(value, list) or len(value) != 1:
            return math.nan
        value = value[0]
    if name == "qurater":
        if not isinstance(value, list) or len(value) != 4:
            return math.nan
        v = np.asarray(value, dtype=float)
        return float(v.mean()) if np.isfinite(v).all() else math.nan
    if isinstance(value, list):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(number):
        return math.nan
    if name in ("rps_doc_word_count", "rps_doc_num_sentences"):
        if number < 0:
            return math.nan
        return math.log1p(number)
    return number


def read_source(code, path):
    features, ids, domains = [], [], []
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if set(row) - META != set(FIELDS):
                raise ValueError(f"{code} row {line_no} has unexpected schema")
            ids.append(str(row["id"]))
            domains.append(str(row.get("_source_domain") or
                               {"A2":"arxiv", "A3":"github"}.get(code)))
            features.append([scalar(name, row[name]) for name in FIELDS])
    if not features:
        raise ValueError(f"Empty input {code}")
    return np.asarray(features, dtype=float), np.asarray(ids), np.asarray(domains)


def main():
    sources = {}
    for code, path in FILES.items():
        sources[code] = read_source(code, path)
    base = sources["A1"][0]
    word_col = FIELDS.index("rps_doc_mean_word_length")
    word_anchor = float(np.nanmedian(base[:, word_col]))
    for code, (x, ids, domains) in sources.items():
        x[:, word_col] = -np.abs(x[:, word_col] - word_anchor)
    base = sources["A1"][0]
    lo, hi = np.nanquantile(base, [.01, .99], axis=0)
    constants = []
    for j in range(len(FIELDS)):
        if not np.isfinite([lo[j], hi[j]]).all():
            raise ValueError(f"No numeric values for {FIELDS[j]}")
        if hi[j] <= lo[j]:
            constants.append(FIELDS[j])
            hi[j] = lo[j] + 1
    settings = []
    for j, name in enumerate(FIELDS):
        direction = -1 if name in REVERSE else 1
        settings.append(Indicator(name, direction, float(lo[j]), float(hi[j]), 1))
    group_settings = [Indicator(i.name, i.direction, i.lower, i.upper,
                                1/(3*len(NATURAL if i.name in NATURAL else
                                         DSIR if i.name in DSIR else MODEL)))
                      for i in settings]
    report = {"method": "exploratory_equal_and_equal_group_weight_22_signal_scores",
              "normalization_reference": "A1 pooled 1st and 99th percentiles; A2/A3 reuse bounds",
              "source_semantics": "Source dataset card consulted for logits ordering and QuRating components",
              "indicator_order": FIELDS, "uncertain_direction_or_shape_fields": sorted(PROVISIONAL),
              "constant_indicators": constants, "word_length_anchor": word_anchor,
              "indicator_config": [{"name":i.name,"direction":i.direction,
                                    "lower":i.lower,"upper":i.upper} for i in settings],
              "list_compression": {"modernbert":"softmax expected class 0..5",
                                   "ad_en":"probability of no-ad class (index 1)",
                                   "fluency_en":"probability of fluent class (index 1)",
                                   "qurater":"mean of 4 source ratings",
                                   "fineweb_edu":"single list element"},
              "coverage_policy":"weighted available fraction >= 0.8; otherwise retain row with Q null",
              "conflict_policy":"four predeclared high>=.8/low<=.2 semantic tensions on fixed A1 scale",
              "scores": {}}
    tables = []
    sets = {code:set(ids) for code, (_, ids, _) in sources.items()}
    report["id_overlap"] = {"A1_A2":len(sets["A1"] & sets["A2"]),
                            "A1_A3":len(sets["A1"] & sets["A3"]),
                            "A2_A3":len(sets["A2"] & sets["A3"])}
    for code, (x, ids, domains) in sources.items():
        equal = score_quality(x, settings, min_coverage=.8, clip=True)
        grouped = score_quality(x, group_settings, min_coverage=.8, clip=True)
        pairs = {"education_vs_no_ad":("fineweb_edu","ad_en"),
                 "qurater_vs_cleanliness":("qurater","modernbert_cleanliness"),
                 "reasoning_vs_readability":("modernbert_reasoning","modernbert_readability"),
                 "fluency_vs_cleanliness":("fluency_en","modernbert_cleanliness")}
        conflicts = {name:semantic_conflict(equal["normalized"],FIELDS.index(high),FIELDS.index(low))
                     for name,(high,low) in pairs.items()}
        conflict = conflicts["education_vs_no_ad"]
        frame = pd.DataFrame({"source":code,"id":ids,"domain":domains,
                              "Q_equal":equal["scores"],"Q_grouped":grouped["scores"],
                              "coverage_equal":equal["coverage"],
                              "coverage_grouped":grouped["coverage"],
                              "conflict":conflict["conflict"],
                              **{name:result["conflict"] for name,result in conflicts.items()}})
        tables.append(frame)
        report["scores"][code] = {"rows":len(frame),"scored_equal":int(frame.Q_equal.notna().sum()),
            "scored_grouped":int(frame.Q_grouped.notna().sum()),
            "clipped_values_equal":equal["clipped_values"],
            "conflict_evaluable":int(conflict["evaluable"].sum()),
            "conflict_count":int(conflict["conflict"].sum()),
            "semantic_conflict_counts":{name:int(result["conflict"].sum())
                                         for name,result in conflicts.items()},
            "domains_equal":summarize_domains(frame.Q_equal.to_numpy(), domains),
            "domains_grouped":summarize_domains(frame.Q_grouped.to_numpy(), domains)}
    all_rows = pd.concat(tables, ignore_index=True)
    output = PROJECT/"02_data/processed"/"q1_quality_scores_exploratory.parquet"
    output.parent.mkdir(parents=True,exist_ok=True)
    all_rows.to_parquet(output,index=False)
    target=PROJECT/"05_results/tables"/"q1_quality_baseline.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"rows":len(all_rows),"overlap":report["id_overlap"],
                      "scores":{k:{"rows":v["rows"],"scored_equal":v["scored_equal"],
                                      "conflict_count":v["conflict_count"]}
                                for k,v in report["scores"].items()},
                      "status":"exploratory_not_validated_Q_for_downstream_optimization"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
