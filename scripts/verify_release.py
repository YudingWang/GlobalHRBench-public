"""Recompute the public snapshot offline. Never call a model or alter labels."""
from pathlib import Path
import sys,json,hashlib,math
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.human_metrics import indexed,rule_metrics,aggregate_fixed,paired_cluster_bootstrap,FAILURES
from evaluation.broad_metrics import mean,paired_summary,equal_groups
from evaluation.agreement import agreement
from evaluation.evidence.scorer import score_qa as evidence_score
from evaluation.routing.scorer import score_qa as routing_score
ROOT=Path(__file__).resolve().parents[1]
def read(p):return [json.loads(s) for s in (ROOT/p).read_text().splitlines() if s]
def load(p):return json.loads((ROOT/p).read_text())
def eq(a,b):
 if isinstance(a,(int,float)) and isinstance(b,(int,float)):assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)
 else:assert a==b,(a,b)
def run():
 manifest=load('MANIFEST.json')
 for path,h in manifest['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
 qs=read('data/questions.jsonl');core=read('data/reviewed_core.jsonl')
 assert len(indexed(qs,'canonical_question_id'))==501 and len(indexed(core,'canonical_question_id'))==138
 ann=indexed(read('data/required_rules.jsonl'),'canonical_question_id');gold=indexed(read('data/evidence_sets.jsonl'),'canonical_question_id')
 scores=read('results/human/answer_items.jsonl');sm=indexed(scores,'request_id');labels=indexed(read('results/human/final_labels.jsonl'),'request_id');answers=indexed(read('results/human/responses.jsonl'),'request_id')
 assert len(sm)==1104 and set(sm)==set(labels)==set(answers)
 metrics=['correctness','answer_value_coverage','qualification_coverage','required_atom_coverage','full_required_rule_completeness','source_fidelity']
 for s in scores:
  rid=s['request_id'];r=labels[rid];review=r['released_review'];a=ann[s['canonical_question_id']];failed=s['output_status'] in FAILURES
  rm=rule_metrics(a,{x['atom_id']:x['label'] for x in review['atoms']})
  for m in metrics:
   if m=='correctness':v=float(review['correctness']=='correct')
   elif m=='source_fidelity':v=float(review['whole_answer_fidelity']=='supported')
   else:v=rm[m]
   if failed and v is not None:v=0.0
   eq(s[m],v)
  response=answers[rid];assert hashlib.sha256(response['answer_text'].encode()).hexdigest()==response['response_sha256']
  g=gold[s['canonical_question_id']];required=set(g['primary_retrieval_gold_ids'])|set(g['required_dependency_source_ids'])
  eq(s['all_required_evidence_available'],required<=set(response['evidence_source_ids']))
 # Every reported question/group/language/evidence-available aggregate.
 tables=read('results/human/answer_tables.jsonl')
 for t in tables:
  rows=[s for s in scores if (s['model_id'],s['condition'])==(t['model_id'],t['condition'])]
  if t['slice']=='human_audit_all_required_evidence_available':rows=[s for s in rows if s['all_required_evidence_available']]
  elif t['slice'].startswith('language='):rows=[s for s in rows if s['language']==t['slice'].split('=',1)[1]]
  else:assert t['slice']=='human_audit'
  role={'qualification_coverage':'is_qualification','answer_value_coverage':'is_answer_value'}.get(t['metric'])
  if role:rows=[s for s in rows if any(a[role] and a['required_for_omission_scoring'] for a in ann[s['canonical_question_id']]['atoms'])]
  result=aggregate_fixed(rows,rows,t['metric'],None if t['weighting']=='question' else t['weighting'])
  for k,v in result.items():eq(t[k],v)
 for t in read('results/human/human_unclear_sensitivity_tables.jsonl'):
  rows=[s for s in scores if (s['model_id'],s['condition'])==(t['model_id'],t['condition'])]
  for k,v in aggregate_fixed(rows,rows,t['metric']+'_unclear_missing',None if t['weighting']=='question' else t['weighting']).items():eq(t[k],v)
 for t in load('results/human/answer_sensitivity.json'):
  c={s['canonical_question_id']:s for s in scores if s['model_id']==t['model_id'] and s['condition']=='public_cue_bm25_top5'}
  d={s['canonical_question_id']:s for s in scores if s['model_id']==t['model_id'] and s['condition']=='gold_designated_evidence'}
  rows=[dict(canonical_question_id=q,c=r[t['metric']],d=d[q][t['metric']],fact_group_id=r['fact_group_id'],dependency_group_id=r['dependency_group_id']) for q,r in c.items()]
  for k,v in paired_cluster_bootstrap(rows,'c','d',t['cluster']).items():eq(t[k],v)
 for cell in load('results/human/joint_counts.json'):
  counts=Counter()
  for s in scores:
   if (s['model_id'],s['condition'])!=(cell['model_id'],cell['condition']):continue
   r=labels[s['request_id']]['released_review'];full=rule_metrics(ann[s['canonical_question_id']],{a['atom_id']:a['label'] for a in r['atoms']})['full_required_rule_completeness']==1
   fidelity=r['whole_answer_fidelity'];counts[('full' if full else 'incomplete')+'_'+(fidelity if fidelity in ['supported','unsupported'] else 'other')]+=1
  for k,v in cell['counts'].items():eq(counts[k],v)
 # Reconstruct pre-adjudication agreement from anonymous initial forms.
 forms=read('results/human/initial_reviews.jsonl');assert len(forms)==2208
 fm={(r['request_id'],r['review_slot']):r for r in forms};assert len(fm)==2208
 pairs={m:[] for m in ['correctness','whole_answer_fidelity','atoms']};all_exact=0;needs_adjudication=0
 for rid in sm:
  a,b=[fm[rid,slot] for slot in ['reviewer_1','reviewer_2']]
  for k in ['correctness','whole_answer_fidelity']:pairs[k].append((a[k],b[k]))
  aa={x['atom_id']:x['label'] for x in a['atoms']};bb={x['atom_id']:x['label'] for x in b['atoms']};assert aa.keys()==bb.keys()
  pairs['atoms'].extend((aa[k],bb[k]) for k in aa)
  same=aa==bb and all(a[k]==b[k] for k in ['correctness','whole_answer_fidelity']);all_exact+=same
  unclear=any(v=='unclear' for v in list(aa.values())+list(bb.values())+[a['correctness'],b['correctness'],a['whole_answer_fidelity'],b['whole_answer_fidelity']]);needs_adjudication+=not same or unclear
 expected=load('results/human/agreement.json')
 for k in pairs:eq(agreement(pairs[k]),expected[k])
 eq(all_exact/1104,expected['all_labels_exact_output_agreement']['raw_exact_agreement']);assert needs_adjudication==113
 # Broad evaluation retains its own automated scorers and question versions.
 broad=read('results/broad/evidence_items.jsonl');routes=read('results/broad/retrieval_items.jsonl');summary=load('results/broad/summary.json');queries=indexed(read('data/broad_questions.jsonl'),'public_id')
 assert len(broad)==2520 and len(routes)==840
 matrix=read('results/broad/matrix_items.jsonl');assert len(matrix)==3360
 cells={(r['model'],r['canonical_question_id'],r['condition']):r for r in matrix};assert len(cells)==3360
 assert len({r['model'] for r in matrix})==14
 for r in routes:
  for condition,field in [('rag','bm25'),('country_filtered_rag','country_filtered')]:
   eq(cells[r['model'],r['canonical_question_id'],condition]['metrics']['correct'],r[field+'_correct'])
 old=read('results/broad/historical_seven_model/evidence_items.jsonl')
 for r in old:
  if r['condition'] in ['closed','oracle']:eq(cells[r['model'],r['canonical_question_id'],r['condition']]['metrics'],r['metrics'])
 bm=indexed(broad,'request_id');rescored=0
 for r in read('results/broad/available_predictions.jsonl'):
  if 'response_sha256' in r:
   assert hashlib.sha256(r['raw_text'].encode()).hexdigest()==r['response_sha256']
   payload=json.loads(r['request']['messages'][1]['content'])
   assert payload['input']['question']==queries[r['query_id']]['question']
   assert [e['id'] for e in payload['evidence']]==r['evidence_source_ids']
   assert r['request']['model']==r['model_id']
  cond=r['condition'] if r['condition'] in ('closed','oracle') else 'rag'
  scorer=evidence_score if r['study']=='evidence_access' else routing_score
  result=scorer(r['prediction'],queries[r['query_id']],setting=cond,visible_passage_ids=set(r['evidence_source_ids']) if cond!='closed' else None)
  expected_metrics=bm[r['request_id']]['metrics'] if r['study']=='evidence_access' else r['metrics']
  for k,v in expected_metrics.items():eq(result[k],v)
  rescored+=1
 assert rescored==1920
 for k,v in paired_summary(routes,'bm25_correct','country_filtered_correct').items():eq(summary['retrieval_pooled'][k],v)
 for cell in summary['retrieval_ordered']:
  rows=[r for r in routes if r['model']==cell['model_id']]
  for k,v in paired_summary(rows,'bm25_correct','country_filtered_correct').items():eq(cell[k],v)
  eq(cell['bm25_equal_groups'],equal_groups(rows,'bm25_correct')['estimate']);eq(cell['routed_equal_groups'],equal_groups(rows,'country_filtered_correct')['estimate'])
 for cell in summary['evidence_cells']:
  rows=[r for r in broad if (r['model'],r['condition'])==(cell['model_id'],cell['condition'])]
  assert len(rows)==60
  for m,v in cell['metrics'].items():
   eq(v['question'],mean([r['metrics'][m] for r in rows]));eq(v['equal_groups'],equal_groups([dict(r,v=r['metrics'][m]) for r in rows],'v')['estimate'])
 for t in summary['evidence_pooled']:
  rr=[r for r in broad if r['condition']==t['condition']]
  eq(t['target_answer_score'],mean([r['metrics']['correct'] for r in rr]));eq(t['n_outputs'],len(rr))
 by={(r['model'],r['canonical_question_id'],r['condition']):r for r in broad}
 for t in summary['evidence_paired']:
  rows=[dict(group_id=by[m,q,t['left_condition']]['group_id'],left=by[m,q,t['left_condition']]['metrics']['correct'],right=by[m,q,t['right_condition']]['metrics']['correct']) for m in sorted({r['model'] for r in broad}) for q in sorted({r['canonical_question_id'] for r in broad})]
  for k,v in paired_summary(rows,'left','right').items():eq(t[k],v)
 result={'passed':True,'questions':501,'reviewed_questions':138,'human_outputs':1104,'initial_reviews':2208,'adjudications':113,'human_aggregate_rows_checked':len(tables),'broad_predictions_rescored':rescored,'retrieval_pairs':840,'evidence_access_outputs':2520,'full_matrix_outputs':3360,'common_bm25_verified':True,'model_calls':0,'scores_modified':False}
 print(json.dumps(result,indent=2))
if __name__=='__main__':run()
