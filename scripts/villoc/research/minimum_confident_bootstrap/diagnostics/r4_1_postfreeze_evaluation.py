#!/usr/bin/env python3
"""R4.1 post-freeze evaluation for minimum-confident bootstrap.

Reference/GT is allowed only after an R4.0 blind freeze. This script does not
modify bootstrap inputs, parameters, or frozen blind outputs.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def fit_similarity(v, m):
    v, m = np.asarray(v,float), np.asarray(m,float)
    z, w = v[:,0]+1j*v[:,1], m[:,0]+1j*m[:,1]
    z0, w0 = z-z.mean(), w-w.mean()
    den = float(np.sum(np.abs(z0)**2))
    if den <= 1e-12: raise RuntimeError('degenerate visual geometry')
    a = np.sum(w0*np.conj(z0))/den; b = w.mean()-a*z.mean()
    return dict(a_real=float(a.real), a_imag=float(a.imag), b_real=float(b.real), b_imag=float(b.imag),
                scale=float(abs(a)), rotation_deg=float(np.degrees(np.angle(a))))


def apply_similarity(xy, m):
    p=np.asarray(xy,float); z=p[:,0]+1j*p[:,1]
    w=complex(m['a_real'],m['a_imag'])*z+complex(m['b_real'],m['b_imag'])
    return np.column_stack([w.real,w.imag])


def metrics(est, gt):
    e=np.linalg.norm(np.asarray(est)-np.asarray(gt),axis=1)
    return dict(rmse_m=float(np.sqrt(np.mean(e*e))), mean_m=float(e.mean()), median_m=float(np.median(e)),
                p95_m=float(np.percentile(e,95)), max_m=float(e.max()), final_m=float(e[-1]))


def path_len(x):
    x=np.asarray(x,float); return 0.0 if len(x)<2 else float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum())


def angle_delta(a,b): return float((a-b+180)%360-180)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-root',type=Path,required=True); ap.add_argument('--research-root',type=Path,required=True)
    a=ap.parse_args(); run=a.run_root.resolve(); r=a.research_root.resolve(); out=r/'postfreeze_eval'; out.mkdir(parents=True,exist_ok=True)
    mf=r/'freeze/r4_0_blind_freeze_manifest.json'; hf=r/'freeze/r4_0_blind_freeze_manifest.sha256'
    expected=hf.read_text().split()[0]; actual=sha256(mf)
    if expected!=actual: raise RuntimeError('freeze hash mismatch')
    frozen=json.loads(mf.read_text())['frozen_algorithm_result']; lock=frozen['map_lock']
    if frozen['localization_state']!='PROVISIONAL_ABSOLUTE_LOCK': raise RuntimeError('unexpected frozen state')

    ref=pd.read_csv(run/'evaluation/reference_attachment.csv'); rel=pd.read_csv(run/'metadata/s8_xfeat_relative_frontend/s8r4_xfeat_relative_trajectory_blind_raw.csv')
    cand=pd.read_csv(r/'candidate_evidence.csv'); strict=json.loads((run/'reports/blind_map_bootstrap/blind_map_bootstrap_report.json').read_text())['map_lock']
    tr=Transformer.from_crs('EPSG:4326','EPSG:3346',always_xy=True); e,n=tr.transform(ref.eval_ref_lon.to_numpy(float),ref.eval_ref_lat.to_numpy(float))
    ref=ref[['query_id']].copy(); ref['gt_easting']=e; ref['gt_northing']=n; ref.query_id=pd.to_numeric(ref.query_id).astype(int)
    rel['query_id']=pd.to_numeric(rel.token0_id).astype(int); merged=rel.merge(ref,on='query_id',validate='one_to_one').sort_values('query_id')
    V=merged[['visual_x_px','visual_y_px']].to_numpy(float); GT=merged[['gt_easting','gt_northing']].to_numpy(float)
    eq=[int(x) for x in lock['evidence_query_ids']]; lq=int(lock['lock_query_id']); ev=merged[merged.query_id.isin(eq)]; pre=merged[merged.query_id<=lq]
    gt_ev=fit_similarity(ev[['visual_x_px','visual_y_px']],ev[['gt_easting','gt_northing']]); gt_pre=fit_similarity(pre[['visual_x_px','visual_y_px']],pre[['gt_easting','gt_northing']])
    r3=dict(a_real=float(lock['a_real']),a_imag=float(lock['a_imag']),b_real=float(lock['b_real']),b_imag=float(lock['b_imag']),scale=float(lock['scale_m_per_visual_px']),rotation_deg=float(lock['rotation_deg']))
    old=dict(a_real=float(strict['a_real']),a_imag=float(strict['a_imag']),b_real=float(strict['b_real']),b_imag=float(strict['b_imag']),scale=float(strict['scale_m_per_visual_px']),rotation_deg=float(strict['rotation_deg']))
    old_est,r3_est=apply_similarity(V,old),apply_similarity(V,r3); mask=(merged.query_id>=lq).to_numpy()
    ev_err=np.linalg.norm(apply_similarity(ev[['visual_x_px','visual_y_px']],r3)-ev[['gt_easting','gt_northing']].to_numpy(float),axis=1)
    intrinsic=np.linalg.norm(apply_similarity(ev[['visual_x_px','visual_y_px']],gt_ev)-ev[['gt_easting','gt_northing']].to_numpy(float),axis=1)
    result=dict(stage='R4.1_POSTFREEZE_TRAJ01_BOOTSTRAP_EVALUATION',freeze_manifest_sha256=actual,
        gt_similarity_exact_r3_evidence=gt_ev,gt_similarity_prefix_to_lock=gt_pre,
        old_strict_q19=dict(scale=old['scale'],rotation_deg=old['rotation_deg'],scale_ratio_vs_gt=old['scale']/gt_pre['scale'],rotation_error_deg_vs_gt=angle_delta(old['rotation_deg'],gt_pre['rotation_deg'])),
        new_r3=dict(scale=r3['scale'],rotation_deg=r3['rotation_deg'],scale_ratio_vs_gt=r3['scale']/gt_pre['scale'],rotation_error_deg_vs_gt=angle_delta(r3['rotation_deg'],gt_pre['rotation_deg'])),
        evidence_transform_truth_error=dict(median_m=float(np.median(ev_err)),max_m=float(ev_err.max()),intrinsic_gt_fit_median_m=float(np.median(intrinsic)),intrinsic_gt_fit_max_m=float(intrinsic.max())),
        trajectory_metrics=dict(strict_full=metrics(old_est,GT),r3_full=metrics(r3_est,GT),strict_common_post_lock=metrics(old_est[mask],GT[mask]),r3_common_post_lock=metrics(r3_est[mask],GT[mask])),
        path_ratios=dict(strict_full=path_len(old_est)/path_len(GT),r3_full=path_len(r3_est)/path_len(GT),strict_common_post_lock=path_len(old_est[mask])/path_len(GT[mask]),r3_common_post_lock=path_len(r3_est[mask])/path_len(GT[mask])),
        evaluation_contract=dict(reference_used=True,reference_usage='postfreeze_evaluation_only',algorithm_outputs_modified=False,algorithm_parameters_modified=False))
    rp=out/'r4_1_postfreeze_evaluation.json'; rp.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); print('report:',rp); print('STATUS: PASS_R4_1_POSTFREEZE_AUDIT_EXECUTION')

if __name__=='__main__': main()
