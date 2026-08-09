#!/usr/bin/env python3
"""R4.2 post-freeze failure localization for minimum-confident bootstrap."""
from __future__ import annotations
import argparse, ast, json
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer


def fit_similarity(v,m):
    v,m=np.asarray(v,float),np.asarray(m,float); z=v[:,0]+1j*v[:,1]; w=m[:,0]+1j*m[:,1]; z0=z-z.mean(); w0=w-w.mean()
    a=np.sum(w0*np.conj(z0))/np.sum(np.abs(z0)**2); b=w.mean()-a*z.mean()
    return dict(a_real=float(a.real),a_imag=float(a.imag),b_real=float(b.real),b_imag=float(b.imag),scale=float(abs(a)),rotation_deg=float(np.degrees(np.angle(a))))


def apply_similarity(xy,m):
    p=np.asarray(xy,float); z=p[:,0]+1j*p[:,1]; w=complex(m['a_real'],m['a_imag'])*z+complex(m['b_real'],m['b_imag']); return np.column_stack([w.real,w.imag])


def parse_list(v):
    try: return list(ast.literal_eval(str(v)))
    except Exception: return [x for x in str(v).split(',') if x]


def norm_tiles(v): return [str(x).strip().strip("'\"") for x in parse_list(v)]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-root',type=Path,required=True); ap.add_argument('--research-root',type=Path,required=True)
    a=ap.parse_args(); run=a.run_root.resolve(); r=a.research_root.resolve(); out=r/'postfreeze_eval'; out.mkdir(parents=True,exist_ok=True)
    report=json.loads((r/'provisional_bootstrap_report.json').read_text()); lock=report['map_lock']; lq=int(lock['lock_query_id']); eq=[int(x) for x in lock['evidence_query_ids']]; sel=[str(x) for x in lock['tile_ids']]
    ref=pd.read_csv(run/'evaluation/reference_attachment.csv'); tr=Transformer.from_crs('EPSG:4326','EPSG:3346',always_xy=True); e,n=tr.transform(ref.eval_ref_lon.to_numpy(float),ref.eval_ref_lat.to_numpy(float)); ref['gt_easting']=e; ref['gt_northing']=n; ref.query_id=pd.to_numeric(ref.query_id).astype(int)
    cand=pd.read_csv(run/'reports/s8_12e1_top20_verifier_reranker/512_s256_orb_hybrid_top20_img518/s8_12e1_all_candidate_verifier_scores.csv'); cand.query_id=pd.to_numeric(cand.query_id).astype(int)
    avail=[]
    for q,tile in zip(eq,sel):
        g=cand[cand.query_id==q].copy(); gt=ref[ref.query_id==q].iloc[0]; g['truth_center_error_m']=np.hypot(g.center_easting-gt.gt_easting,g.center_northing-gt.gt_northing); g=g.sort_values('hybrid_rank'); b4=g[g.hybrid_rank<=4].sort_values('truth_center_error_m').iloc[0]; b20=g.sort_values('truth_center_error_m').iloc[0]; s=g[g.tile_id.astype(str)==tile].iloc[0]
        avail.append(dict(query_id=q,selected_tile=tile,selected_error_m=float(s.truth_center_error_m),selected_hybrid_rank=float(s.hybrid_rank),best_top4_tile=str(b4.tile_id),best_top4_error_m=float(b4.truth_center_error_m),best_top20_tile=str(b20.tile_id),best_top20_error_m=float(b20.truth_center_error_m),best_top20_hybrid_rank=float(b20.hybrid_rank),best_top20_dino_rank=float(b20['rank'])))
    availability=pd.DataFrame(avail)
    hyp=pd.read_csv(r/'hypothesis_updates.csv'); qh=hyp[hyp.update_query_id==lq].copy(); qh['pareto']=qh.pareto.astype(str).str.lower().isin(['true','1'])
    rel=pd.read_csv(run/'metadata/s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_blind_raw.csv'); rel['query_id']=pd.to_numeric(rel.token0_id).astype(int); pre=rel[rel.query_id<=lq].merge(ref[['query_id','gt_easting','gt_northing']],on='query_id',validate='one_to_one').sort_values('query_id')
    gt_model=fit_similarity(pre[['visual_x_px','visual_y_px']],pre[['gt_easting','gt_northing']]); sentinel=pre[['visual_x_px','visual_y_px']].to_numpy(float)[[0,-1]]; gt_pred=apply_similarity(sentinel,gt_model)
    def disagree(row):
        m={k:float(row[k]) for k in ['a_real','a_imag','b_real','b_imag']}; d=np.linalg.norm(apply_similarity(sentinel,m)-gt_pred,axis=1); return pd.Series(dict(gt_disagreement_start_m=float(d[0]),gt_disagreement_lock_m=float(d[1]),gt_disagreement_max_m=float(d.max())))
    diag=pd.concat([qh.reset_index(drop=True),qh.reset_index(drop=True).apply(disagree,axis=1)],axis=1); diag.to_csv(out/'r4_2_lock_hypothesis_truth_diagnostic.csv',index=False); availability.to_csv(out/'r4_2_candidate_availability.csv',index=False)
    frozen=diag[diag.tile_ids.apply(lambda x:norm_tiles(x)==sel)]; ranks=diag.gt_disagreement_max_m.rank(method='min'); fr=int(ranks.loc[frozen.index[0]]) if len(frozen) else None
    clusters=pd.read_csv(r/'transform_clusters.csv'); lc=clusters[clusters.update_query_id==lq]
    result=dict(stage='R4.2_POSTFREEZE_FAILURE_LOCALIZATION',lock_query_id=lq,evidence_query_ids=eq,selected_tiles=sel,candidate_availability=availability.to_dict('records'),admissible_hypotheses_at_lock=int(len(qh)),pareto_hypotheses_at_lock=int(qh.pareto.sum()),frozen_hypothesis_gt_agreement_rank=fr,best_hypotheses_by_gt_transform_agreement=diag.sort_values(['gt_disagreement_max_m','median_center_residual_m']).head(15).to_dict('records'),lock_clusters=lc.to_dict('records'),evaluation_contract=dict(reference_used=True,reference_usage='postfreeze_diagnosis_only',algorithm_modified=False))
    rp=out/'r4_2_failure_localization.json'; rp.write_text(json.dumps(result,indent=2)); print(availability.to_string(index=False)); print(); print(diag.sort_values(['gt_disagreement_max_m','median_center_residual_m']).head(15)[['pareto','tile_ids','candidate_choice_ranks','median_center_residual_m','sum_hybrid_rank','sum_dino_rank','scale_m_per_visual_px','rotation_deg','gt_disagreement_start_m','gt_disagreement_lock_m','gt_disagreement_max_m']].to_string(index=False)); print(); print(lc.to_string(index=False)); print('report:',rp); print('STATUS: PASS_R4_2_FAILURE_LOCALIZATION_EXECUTION')

if __name__=='__main__': main()
