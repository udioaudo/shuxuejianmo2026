"""A1/A18 lexical domain mapping with A16 known-pair checks.

Similarity is a proxy, not proof of equal quality or identical data domains.
Raw text remains in memory; only aggregate similarity is persisted.
"""

from collections import defaultdict
from pathlib import Path
import json
import lzma
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'02_data/raw/real_attachments/A_data_value'
SEED=20260923
MAX_PER_DOMAIN=200
CHARS=4000


def reservoir(path, text_field, domain_field):
    rng=np.random.default_rng(SEED)
    counts=defaultdict(int)
    samples=defaultdict(list)
    with lzma.open(path,'rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            domain=str(row[domain_field])
            content=str(row.get(text_field,'') or '')[:CHARS]
            counts[domain]+=1
            if len(samples[domain])<MAX_PER_DOMAIN:
                samples[domain].append(content)
            else:
                draw=int(rng.integers(0,counts[domain]))
                if draw<MAX_PER_DOMAIN:
                    samples[domain][draw]=content
    return samples,dict(counts)


def centroids(samples,vectorizer):
    vectors={}
    for domain,texts in samples.items():
        matrix=vectorizer.transform(texts)
        average=np.asarray(matrix.mean(axis=0)).ravel()
        norm=np.linalg.norm(average)
        vectors[domain]=average/norm if norm else average
    return vectors


def main():
    sample,count_a1=reservoir(A/'slimpajama_quality_signal_sample.jsonl.xz',
                              'content','_source_domain')
    regmix,count_a18=reservoir(A/'regmix_domain_sample.jsonl.xz',
                               'text','_source_domain')
    vectorizer=HashingVectorizer(analyzer='char',ngram_range=(3,4),
                                n_features=2**15,alternate_sign=False,norm='l2',dtype=np.float32)
    source=centroids(sample,vectorizer)
    destination=centroids(regmix,vectorizer)
    guide=pd.read_csv(A/'domain_mapping_guide.csv')
    q=json.loads((ROOT/'05_results/tables/q1_quality_baseline.json').read_text(encoding='utf-8'))
    quality={r['domain']:r['quality_mean'] for r in q['scores']['A1']['domains_equal']}
    output=[]
    for row in guide.itertuples(index=False):
        name=row.mixture_domain
        if name not in destination:
            raise ValueError(f'Missing A18 domain: {name}')
        sims={domain:float(np.dot(destination[name],v)) for domain,v in source.items()}
        ranked=sorted(sims,key=sims.get,reverse=True)
        known=row.quality_domain if row.quality_domain!='(none)' else None
        output.append({'mixture_domain':name,'guide_quality_domain':known,
                       'guide_mapping_type':row.mapping_type,
                       'text_count_A18':count_a18[name],
                       'ranked_quality_domains':[{'domain':k,'cosine':sims[k],
                                                  'A1_equal_Q':quality[k]} for k in ranked],
                       'top1_margin':sims[ranked[0]]-sims[ranked[1]],
                       'known_mapping_rank':ranked.index(known)+1 if known else None,
                       'conditional_top1_quality_proxy':quality[ranked[0]]})
    known=[r for r in output if r['guide_quality_domain'] is not None]
    results={'A1_texts_processed':sum(count_a1.values()),
             'A18_texts_processed':sum(count_a18.values()),
             'sample_cap_per_domain':MAX_PER_DOMAIN,
             'characters_per_document':CHARS,
             'A1_domain_counts':count_a1,'A18_domain_counts':count_a18,
             'known_guide_pairs':len(known),
             'known_top1_correct':sum(r['known_mapping_rank']==1 for r in known),
             'known_top3_correct':sum(r['known_mapping_rank']<=3 for r in known),
             'unknown_domains':sum(r['guide_quality_domain'] is None for r in output),
             'method':'fixed-seed reservoir + hashed character 3/4-gram mean document vectors + cosine',
             'status':'lexical similarity only; semantic quality transfer not established',
             'mappings':output}
    target=ROOT/'05_results/tables/q1_lexical_domain_mapping.json'
    target.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'A1_processed':results['A1_texts_processed'],
                      'A18_processed':results['A18_texts_processed'],
                      'known_top1':results['known_top1_correct'],
                      'known_pairs':results['known_guide_pairs'],
                      'known_top3':results['known_top3_correct'],
                      'unknown':results['unknown_domains'],
                      'sample_unknown_top1':{r['mixture_domain']:r['ranked_quality_domains'][0]['domain']
                                             for r in output if r['guide_quality_domain'] is None}},
                     ensure_ascii=False))


if __name__=='__main__':main()
