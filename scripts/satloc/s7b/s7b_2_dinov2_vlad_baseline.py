#!/usr/bin/env python3
"""
S7B.2 — DINOv2-VLAD / AnyLoc-style Retrieval Baseline

Self-contained local-token aggregation baseline:
  image -> DINOv2 patch tokens -> MiniBatchKMeans codebook -> VLAD -> cosine retrieval

GT/reference rule: lon/lat are used only after ranking for offline Recall@K/error metrics.
"""
from __future__ import annotations

import argparse, hashlib, json, math, platform, sys, time, re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print = partial(print, flush=True)

Q_MANIFEST = Path("outputs/satloc/metadata/s7_retrieval_upgrade/s7_0_query_manifest.csv")
SAT_INDEX = Path("outputs/satloc/metadata/satellite_tiles_index_enriched.csv")
SCENE_LABELS = Path("outputs/satloc/metadata/s7_retrieval_upgrade/s7_scene_labels_canonical_traj01.csv")
META_OUT = Path("outputs/satloc/metadata/s7b_dinov2_vlad")
REPORT_OUT = Path("outputs/satloc/reports/s7b_dinov2_vlad")
FIG_OUT = Path("outputs/satloc/figures/s7b_dinov2_vlad")


@dataclass
class Cache:
    descriptors: np.ndarray
    ids: list[str]
    paths: list[str]
    meta: dict[str, Any]


def ap() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S7B.2 DINOv2-VLAD retrieval baseline")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--query-manifest", type=Path, default=Q_MANIFEST)
    p.add_argument("--satellite-index", type=Path, default=SAT_INDEX)
    p.add_argument("--scene-labels", type=Path, default=SCENE_LABELS)
    p.add_argument("--metadata-out", type=Path, default=META_OUT)
    p.add_argument("--report-out", type=Path, default=REPORT_OUT)
    p.add_argument("--figure-out", type=Path, default=FIG_OUT)

    # Frozen schema from S7B.1 preflight
    p.add_argument("--query-token-col", default="token0_id")
    p.add_argument("--query-path-col", default="image_path")
    p.add_argument("--query-lon-col", default="lon")
    p.add_argument("--query-lat-col", default="lat")
    p.add_argument("--sat-id-col", default="tile_index")
    p.add_argument("--sat-path-col", default="tile_path")
    p.add_argument("--sat-lon-col", default="lon_center")
    p.add_argument("--sat-lat-col", default="lat_center")
    p.add_argument("--scene-token-col", default="token")
    p.add_argument("--scene-col", default="primary_scene")

    p.add_argument("--model-name", default="dinov2_vits14")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--crop-mode", choices=["center_square", "resize_square"], default="center_square")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--batch-size", type=int, default=1)

    p.add_argument("--vlad-clusters", type=int, default=32)
    p.add_argument("--codebook-sat-tiles", type=int, default=1500, help="0=all selected satellite tiles")
    p.add_argument("--codebook-queries", type=int, default=0, help="0 keeps codebook satellite-only")
    p.add_argument("--patches-per-image", type=int, default=32)
    p.add_argument("--max-codebook-patches", type=int, default=120000)
    p.add_argument("--kmeans-batch-size", type=int, default=4096)
    p.add_argument("--kmeans-max-iter", type=int, default=100)
    p.add_argument("--random-seed", type=int, default=7)
    p.add_argument("--no-powerlaw", action="store_true")
    p.add_argument("--no-intra-normalize", action="store_true")

    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--threshold-m", type=float, default=40.0)
    p.add_argument("--recall-ks", default="1,5,10,20,50,100")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--max-sat-tiles", type=int, default=0)
    p.add_argument("--sat-subset-mode", choices=["first", "uniform"], default="uniform")
    p.add_argument("--cache-tag", default="")
    p.add_argument("--rebuild-codebook", action="store_true")
    p.add_argument("--rebuild-sat-cache", action="store_true")
    p.add_argument("--rebuild-query-cache", action="store_true")
    p.add_argument("--allow-missing-images", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def rpath(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def read_csv(path: Path, name: str, required=True) -> pd.DataFrame:
    if not path.exists():
        if required: raise FileNotFoundError(f"Missing {name}: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def req(df: pd.DataFrame, cols: list[str], name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise ValueError(f"{name} missing {miss}; available={list(df.columns)}")


def js(x: Any) -> Any:
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, float) and not math.isfinite(x): return None
    try:
        if pd.isna(x): return None
    except Exception: pass
    return x


def sname(x: Any) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")
    return s or "run"


def resolve_img(root: Path, v: Any) -> str:
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except Exception: pass
    p = Path(str(v).strip())
    if not str(p): return ""
    cands = [p] if p.is_absolute() else [root / p]
    for c in cands:
        if c.exists(): return str(c)
    return ""


def rec_ks(s: str, top_k: int) -> list[int]:
    ks = sorted({int(x) for x in s.split(",") if x.strip() and 0 < int(x) <= top_k})
    if top_k not in ks: ks.append(top_k)
    return ks


def l2(x: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def hdist(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1 = float(lon1), float(lat1)
    lon2, lat2 = lon2.astype(float), lat2.astype(float)
    R = 6371008.8
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = np.deg2rad(lat2 - lat1), np.deg2rad(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def preprocess(img: Image.Image, size: int, crop: str) -> np.ndarray:
    img = img.convert("RGB")
    if crop == "center_square":
        w, h = img.size; side = min(w, h); left = (w-side)//2; top = (h-side)//2
        img = img.crop((left, top, left+side, top+side))
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    img = img.resize((size, size), resample=resample)
    a = np.asarray(img).astype(np.float32)/255.0
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)
    std = np.asarray([0.229, 0.224, 0.225], np.float32)
    return np.transpose((a-mean)/std, (2,0,1))


def choose_device(reqdev: str):
    import torch
    info = {"requested_device": reqdev, "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())}
    if reqdev == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        d = "cuda"
    elif reqdev == "mps":
        if not info["mps_available"]: raise RuntimeError("MPS requested but unavailable")
        d = "mps"
    elif reqdev == "cpu": d = "cpu"
    elif torch.cuda.is_available(): d = "cuda"
    elif info["mps_available"]: d = "mps"
    else: d = "cpu"
    info["selected_device"] = d
    if d == "cuda": info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return d, info


class DinoPatches:
    def __init__(self, model_name: str, size: int, crop: str, device: str):
        import torch
        self.torch = torch; self.model_name = model_name; self.size = size; self.crop = crop
        self.device, self.device_info = choose_device(device)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        self.model.eval().to(self.device)

    def meta(self):
        return {"backend":"torch_hub", "model_name":self.model_name, "image_size":self.size,
                "crop_mode":self.crop, **self.device_info}

    def tokens(self, paths: list[str]) -> list[np.ndarray]:
        torch = self.torch
        arr = [preprocess(Image.open(p), self.size, self.crop) for p in paths]
        x = torch.from_numpy(np.stack(arr, axis=0)).to(self.device)
        with torch.inference_mode():
            out = self.model.forward_features(x)
            if not isinstance(out, dict) or "x_norm_patchtokens" not in out:
                raise RuntimeError("DINOv2 forward_features missing x_norm_patchtokens")
            tok = out["x_norm_patchtokens"].detach().float().cpu().numpy().astype(np.float32)
        return [l2(tok[i], axis=1) for i in range(tok.shape[0])]


def dfhash(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha1()
    for c in cols:
        if c not in df.columns: continue
        h.update(c.encode())
        for v in df[c].astype(str).fillna("").tolist():
            h.update(v.encode()); h.update(b"\0")
    return h.hexdigest()[:12]


def load_query(root: Path, a) -> pd.DataFrame:
    q = read_csv(rpath(root, a.query_manifest), "S7 query manifest")
    req(q, [a.query_token_col,a.query_path_col,a.query_lon_col,a.query_lat_col], "S7 query manifest")
    out = q.copy()
    out["token"] = pd.to_numeric(out[a.query_token_col], errors="raise").astype(int)
    out["query_image_path"] = out[a.query_path_col].apply(lambda x: resolve_img(root, x))
    out["query_lon_eval_only"] = pd.to_numeric(out[a.query_lon_col], errors="coerce")
    out["query_lat_eval_only"] = pd.to_numeric(out[a.query_lat_col], errors="coerce")
    sp = rpath(root, a.scene_labels)
    if sp.exists():
        sc = read_csv(sp, "scene labels", required=False)
        if not sc.empty and a.scene_token_col in sc.columns and a.scene_col in sc.columns:
            ss = sc[[a.scene_token_col,a.scene_col]].copy()
            ss["token"] = pd.to_numeric(ss[a.scene_token_col], errors="coerce").astype("Int64")
            ss = ss.dropna(subset=["token"]); ss["token"] = ss["token"].astype(int)
            ss = ss[["token",a.scene_col]].rename(columns={a.scene_col:"primary_scene"}).drop_duplicates("token")
            out = out.merge(ss, on="token", how="left", validate="many_to_one")
    if "primary_scene" not in out: out["primary_scene"] = "unlabeled"
    out["primary_scene"] = out["primary_scene"].fillna("unlabeled").astype(str).str.strip().replace({"":"unlabeled"})
    out = out.drop_duplicates("token").sort_values("token").reset_index(drop=True)
    if len(out) != 263: raise ValueError(f"Expected 263 official query tokens, got {len(out)}")
    return out


def load_sat(root: Path, a) -> pd.DataFrame:
    s = read_csv(rpath(root, a.satellite_index), "satellite index")
    req(s, [a.sat_id_col,a.sat_path_col,a.sat_lon_col,a.sat_lat_col], "satellite index")
    out = s.copy()
    out["tile_id"] = out[a.sat_id_col].astype(str)
    out["sat_image_path"] = out[a.sat_path_col].apply(lambda x: resolve_img(root, x))
    out["sat_lon_eval_only"] = pd.to_numeric(out[a.sat_lon_col], errors="coerce")
    out["sat_lat_eval_only"] = pd.to_numeric(out[a.sat_lat_col], errors="coerce")
    return out.drop_duplicates("tile_id").reset_index(drop=True)


def missing(df: pd.DataFrame, col: str, name: str, allow: bool):
    m = df[col].astype(str).str.len().eq(0)
    if m.any():
        msg = f"{name}: {int(m.sum())}/{len(df)} missing paths\n{df.loc[m].head(10)}"
        if not allow: raise FileNotFoundError(msg)
        print("[WARN]", msg); df = df.loc[~m].copy()
    return df.reset_index(drop=True)


def subset_sat(s: pd.DataFrame, n: int, mode: str):
    if n <= 0 or n >= len(s): return s.reset_index(drop=True)
    if mode == "first": return s.head(n).reset_index(drop=True)
    idx = np.linspace(0, len(s)-1, n).round().astype(int)
    return s.iloc[sorted(set(idx.tolist()))].reset_index(drop=True)


def codebook_images(sat: pd.DataFrame, q: pd.DataFrame, a) -> pd.DataFrame:
    if a.codebook_sat_tiles <= 0 or a.codebook_sat_tiles >= len(sat): ss = sat.copy()
    else:
        idx = np.linspace(0, len(sat)-1, a.codebook_sat_tiles).round().astype(int)
        ss = sat.iloc[sorted(set(idx.tolist()))].copy()
    ss = ss.rename(columns={"tile_id":"image_id", "sat_image_path":"image_path"})
    ss["source"] = "satellite"; ss = ss[["source","image_id","image_path"]]
    if a.codebook_queries > 0:
        qs = q.head(min(a.codebook_queries, len(q))).rename(columns={"token":"image_id", "query_image_path":"image_path"})
        qs["source"] = "query"; qs = qs[["source","image_id","image_path"]]
        ss = pd.concat([ss, qs], ignore_index=True)
    return ss.reset_index(drop=True)


def collect_samples(ext: DinoPatches, imgs: pd.DataFrame, a) -> np.ndarray:
    rng = np.random.default_rng(a.random_seed)
    paths = imgs["image_path"].astype(str).tolist()
    samples, total = [], 0; t0 = time.time()
    print(f"[S7B.2] Collecting codebook patch samples from {len(paths)} images...")
    for st in range(0, len(paths), a.batch_size):
        en = min(st+a.batch_size, len(paths))
        for patches in ext.tokens(paths[st:en]):
            take = min(a.patches_per_image, patches.shape[0], a.max_codebook_patches-total)
            if take <= 0: break
            idx = rng.choice(patches.shape[0], size=take, replace=False)
            samples.append(patches[idx]); total += take
        if en == len(paths) or en % max(20*a.batch_size,1)==0 or total >= a.max_codebook_patches:
            print(f"  codebook sampling: images={en}/{len(paths)} patches={total}/{a.max_codebook_patches} rate={en/max(time.time()-t0,1e-9):.2f} img/s")
        if total >= a.max_codebook_patches: break
    if not samples: raise RuntimeError("No codebook patches sampled")
    X = l2(np.vstack(samples).astype(np.float32), axis=1)
    print(f"[S7B.2] Codebook sample matrix: {X.shape}")
    return X


def codebook(path: Path, ext: DinoPatches, imgs: pd.DataFrame, a):
    if path.exists() and not a.rebuild_codebook:
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta_json"].item())) if "meta_json" in d else {}
        print(f"[S7B.2] Using cached VLAD codebook: {path}")
        return l2(d["centers"].astype(np.float32), axis=1), meta
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise RuntimeError("S7B.2 requires scikit-learn. Install: pip install scikit-learn") from e
    X = collect_samples(ext, imgs, a)
    print(f"[S7B.2] Fitting MiniBatchKMeans: clusters={a.vlad_clusters}")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=a.vlad_clusters, random_state=a.random_seed,
                         batch_size=a.kmeans_batch_size, max_iter=a.kmeans_max_iter,
                         n_init=3, verbose=0)
    km.fit(X)
    centers = l2(km.cluster_centers_.astype(np.float32), axis=1)
    meta = {"clusters":a.vlad_clusters, "patch_samples":int(X.shape[0]), "patch_dim":int(X.shape[1]),
            "codebook_images":int(len(imgs)), "runtime_s":float(time.time()-t0), "inertia":float(km.inertia_),
            "image_size":a.image_size, "patches_per_image":a.patches_per_image, "max_codebook_patches":a.max_codebook_patches,
            **ext.meta()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, centers=centers.astype(np.float32), meta_json=np.asarray(json.dumps(meta, default=js)))
    print(f"[S7B.2] Saved VLAD codebook: {path}")
    return centers, meta


def vlad(patches: np.ndarray, centers: np.ndarray, intra=True, power=True) -> np.ndarray:
    patches = l2(patches, axis=1); centers = l2(centers, axis=1)
    assign = np.argmax(patches @ centers.T, axis=1)
    K, D = centers.shape; V = np.zeros((K,D), np.float32)
    for k in range(K):
        m = assign == k
        if np.any(m): V[k] = (patches[m] - centers[k]).sum(axis=0)
    if intra: V = l2(V, axis=1)
    flat = V.reshape(-1)
    if power: flat = np.sign(flat) * np.sqrt(np.abs(flat) + 1e-12)
    n = np.linalg.norm(flat)
    if n > 1e-12: flat = flat / n
    return flat.astype(np.float32)


def load_desc(path: Path, ids: list[str]) -> Cache | None:
    if not path.exists(): return None
    d = np.load(path, allow_pickle=False)
    cids = [str(x) for x in d["ids"].tolist()]
    if cids != [str(x) for x in ids]: return None
    meta = json.loads(str(d["meta_json"].item())) if "meta_json" in d else {}
    paths = [str(x) for x in d["paths"].tolist()]
    return Cache(l2(d["descriptors"].astype(np.float32), axis=1), cids, paths, meta)


def save_desc(path: Path, desc: np.ndarray, ids: list[str], paths: list[str], meta: dict[str,Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, descriptors=desc.astype(np.float32), ids=np.asarray(ids, dtype=str),
                        paths=np.asarray(paths, dtype=str), meta_json=np.asarray(json.dumps(meta, default=js)))


def extract_vlad(name: str, df: pd.DataFrame, id_col: str, path_col: str, cpath: Path, rebuild: bool, ext: DinoPatches, centers: np.ndarray, a) -> Cache:
    ids = df[id_col].astype(str).tolist(); paths = df[path_col].astype(str).tolist()
    if cpath.exists() and not rebuild:
        c = load_desc(cpath, ids)
        if c is not None:
            print(f"[S7B.2] Using cached {name} VLAD descriptors: {cpath}")
            return c
    print(f"[S7B.2] Extracting {name} VLAD descriptors: rows={len(df)} batch_size={a.batch_size}")
    t0 = time.time(); desc = []
    for st in range(0, len(paths), a.batch_size):
        en = min(st+a.batch_size, len(paths))
        for p in ext.tokens(paths[st:en]):
            desc.append(vlad(p, centers, intra=not a.no_intra_normalize, power=not a.no_powerlaw))
        if en == len(paths) or en % max(20*a.batch_size,1)==0:
            print(f"  {name}: {en}/{len(paths)} images, {en/max(time.time()-t0,1e-9):.2f} img/s")
    D = l2(np.vstack(desc).astype(np.float32), axis=1)
    meta = {"name":name, "rows":len(df), "descriptor_dim":int(D.shape[1]), "runtime_s":float(time.time()-t0),
            "vlad_clusters":int(centers.shape[0]), "patch_dim":int(centers.shape[1]),
            "intra_normalize":not a.no_intra_normalize, "powerlaw":not a.no_powerlaw, **ext.meta()}
    save_desc(cpath, D, ids, paths, meta)
    print(f"[S7B.2] Saved {name} VLAD descriptors: {cpath}")
    return Cache(D, ids, paths, meta)


def topk(q: np.ndarray, s: np.ndarray, k: int):
    q = l2(q, axis=1); s = l2(s, axis=1); k = min(k, s.shape[0])
    sim = q @ s.T
    if k == s.shape[0]: idx = np.argsort(-sim, axis=1)[:, :k]
    else:
        ip = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, ip, axis=1), axis=1)
        idx = np.take_along_axis(ip, order, axis=1)
    return idx, np.take_along_axis(sim, idx, axis=1)


def outputs(qdf, sdf, idx, sim, threshold, ks, skip_eval):
    sat_ids = sdf["tile_id"].astype(str).to_numpy(); sat_lon = sdf["sat_lon_eval_only"].to_numpy(float); sat_lat = sdf["sat_lat_eval_only"].to_numpy(float)
    crows, qrows = [], []
    for qi, qr in qdf.reset_index(drop=True).iterrows():
        inds, sims = idx[qi], sim[qi]
        errs = np.full(len(inds), np.nan, float)
        if not skip_eval and pd.notna(qr["query_lon_eval_only"]) and pd.notna(qr["query_lat_eval_only"]):
            errs = hdist(qr["query_lon_eval_only"], qr["query_lat_eval_only"], sat_lon[inds], sat_lat[inds])
        token = int(qr["token"])
        for rank,(si,sc,er) in enumerate(zip(inds,sims,errs), start=1):
            crows.append({"token":token,"rank":rank,"tile_id":str(sat_ids[si]),"dinov2_vlad_similarity":float(sc),
                          "eval_error_m":float(er) if math.isfinite(float(er)) else np.nan,
                          "hit_le_threshold_eval_only":bool(math.isfinite(float(er)) and float(er)<=threshold),
                          "query_image_path":qr["query_image_path"],"sat_image_path":sdf.iloc[int(si)]["sat_image_path"],"primary_scene":qr["primary_scene"]})
        if np.isfinite(errs).any():
            oi = int(np.nanargmin(errs)); oracle_err = float(errs[oi]); oracle_rank = oi+1; oracle_tile = str(sat_ids[inds[oi]]); top1_err = float(errs[0])
        else:
            oracle_err = np.nan; oracle_rank = np.nan; oracle_tile = ""; top1_err = np.nan
        row = {"token":token,"primary_scene":qr["primary_scene"],"query_image_path":qr["query_image_path"],
               "top1_tile_id":str(sat_ids[inds[0]]),"top1_similarity":float(sims[0]),"top1_error_m_eval_only":top1_err,
               "top1_hit_le_threshold_eval_only":bool(math.isfinite(top1_err) and top1_err<=threshold),
               "oracle_tile_id_eval_only":oracle_tile,"oracle_error_m_eval_only":oracle_err,"oracle_rank_eval_only":oracle_rank}
        for k in ks:
            kk = min(k, len(errs)); row[f"hit_at_{k}_eval_only"] = bool(kk>0 and np.isfinite(errs[:kk]).any() and np.nanmin(errs[:kk]) <= threshold)
        qrows.append(row)
    cand = pd.DataFrame(crows); query = pd.DataFrame(qrows)
    recall = pd.DataFrame([{"k":k,"hits":int(query[f"hit_at_{k}_eval_only"].sum()),"queries":len(query),"recall":float(query[f"hit_at_{k}_eval_only"].mean())} for k in ks])
    scene_rows = []
    for sc,g in query.groupby("primary_scene", dropna=False):
        r = {"primary_scene":sc,"queries":len(g),"top1_hits":int(g["top1_hit_le_threshold_eval_only"].sum()),"top1_hit_rate":float(g["top1_hit_le_threshold_eval_only"].mean()),
             "median_top1_error_m":float(pd.to_numeric(g["top1_error_m_eval_only"], errors="coerce").median()),
             "median_oracle_error_m":float(pd.to_numeric(g["oracle_error_m_eval_only"], errors="coerce").median()),
             "median_oracle_rank":float(pd.to_numeric(g["oracle_rank_eval_only"], errors="coerce").median())}
        for k in ks: r[f"recall_at_{k}"] = float(g[f"hit_at_{k}_eval_only"].mean())
        scene_rows.append(r)
    scene = pd.DataFrame(scene_rows).sort_values(["queries","primary_scene"], ascending=[False,True])
    return cand, query, recall, scene


def plot_recall(recall, out: Path, threshold):
    if recall.empty: return
    fig, ax = plt.subplots(figsize=(8,4.8)); ax.bar(recall["k"].astype(str), recall["recall"]); ax.set_ylim(0,1)
    ax.set_xlabel("K"); ax.set_ylabel(f"Recall@K <= {threshold:.0f} m"); ax.set_title("S7B.2 DINOv2-VLAD retrieval"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=180); plt.close(fig)


def plot_hist(vals, out: Path, title: str, xlabel: str):
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if v.empty: return
    v = v.clip(upper=float(v.quantile(0.98)))
    fig, ax = plt.subplots(figsize=(8,4.8)); ax.hist(v, bins=min(50, max(10, int(math.sqrt(len(v)))))); ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Count"); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=180); plt.close(fig)


def main():
    a = ap(); root = a.repo_root.resolve()
    mo, ro, fo = rpath(root,a.metadata_out), rpath(root,a.report_out), rpath(root,a.figure_out)
    for d in [mo,ro,fo]: d.mkdir(parents=True, exist_ok=True)
    ks = rec_ks(a.recall_ks, a.top_k)
    run = sname(f"torch_hub_{a.model_name}_vlad_k{a.vlad_clusters}_img{a.image_size}_top{a.top_k}_q{a.max_queries if a.max_queries>0 else 263}_sat{a.max_sat_tiles if a.max_sat_tiles>0 else 8625}_cb{a.codebook_sat_tiles if a.codebook_sat_tiles>0 else 'all'}_{a.cache_tag}")

    print("\nS7B.2 — DINOv2-VLAD / AnyLoc-style Retrieval")
    print("---------------------------------------------")
    print(f"Repository root:        {root}")
    print(f"Run name:               {run}")
    print(f"Model name:             {a.model_name}")
    print(f"Image size:             {a.image_size}")
    print(f"VLAD clusters:          {a.vlad_clusters}")
    print(f"Codebook sat tiles:     {a.codebook_sat_tiles}")
    print(f"Patches per image:      {a.patches_per_image}")
    print(f"Max codebook patches:   {a.max_codebook_patches}")
    print(f"Top-K:                  {a.top_k}")
    print(f"Threshold m:            {a.threshold_m}\n")

    q = load_query(root, a); s = load_sat(root, a)
    if a.max_queries > 0: q = q.head(a.max_queries).reset_index(drop=True)
    s = subset_sat(s, a.max_sat_tiles, a.sat_subset_mode)
    q = missing(q, "query_image_path", "queries", a.allow_missing_images)
    s = missing(s, "sat_image_path", "satellite tiles", a.allow_missing_images)
    print(f"Queries:                {len(q)}")
    print(f"Satellite tiles:        {len(s)}")
    print("Scene distribution:"); print(q["primary_scene"].value_counts(dropna=False).to_string()); print()

    print("[S7B.2] Loading DINOv2 patch-token model/backend...")
    ext = DinoPatches(a.model_name, a.image_size, a.crop_mode, a.device)

    cbimgs = codebook_images(s, q, a)
    cb_hash = dfhash(cbimgs, ["source","image_id","image_path"]); s_hash = dfhash(s, ["tile_id","sat_image_path"]); q_hash = dfhash(q, ["token","query_image_path"])
    base = sname(f"torch_hub_{a.model_name}_vlad_k{a.vlad_clusters}_img{a.image_size}_{a.crop_mode}_ppi{a.patches_per_image}_maxp{a.max_codebook_patches}_cb{cb_hash}")
    cb_path = mo / f"s7b2_dinov2_vlad_codebook_{base}.npz"
    s_path = mo / f"s7b2_dinov2_vlad_satellite_descriptors_{base}_{s_hash}.npz"
    q_path = mo / f"s7b2_dinov2_vlad_query_descriptors_{base}_{q_hash}.npz"
    env_path = ro / f"s7b2_dinov2_vlad_environment_{run}.json"
    print(f"[S7B.2] Codebook cache:       {cb_path}")
    print(f"[S7B.2] Satellite VLAD cache: {s_path}")
    print(f"[S7B.2] Query VLAD cache:     {q_path}")

    centers, cbmeta = codebook(cb_path, ext, cbimgs, a)
    env = {"python":sys.version,"platform":platform.platform(),"args":{k:js(v) for k,v in vars(a).items()},"extractor":ext.meta(),"run_name":run,"queries":len(q),"satellite_tiles":len(s),"codebook_meta":cbmeta}
    env_path.write_text(json.dumps(env, indent=2, default=js), encoding="utf-8")

    s_cache = extract_vlad("satellite", s, "tile_id", "sat_image_path", s_path, a.rebuild_sat_cache, ext, centers, a)
    q_cache = extract_vlad("query", q, "token", "query_image_path", q_path, a.rebuild_query_cache, ext, centers, a)

    t0 = time.time(); idx, sim = topk(q_cache.descriptors, s_cache.descriptors, a.top_k); rtime = time.time()-t0
    cand, query, recall, scene = outputs(q, s, idx, sim, a.threshold_m, ks, a.skip_eval)

    cand_csv = mo / f"s7b2_dinov2_vlad_candidate_scores_{run}.csv"; query_csv = mo / f"s7b2_dinov2_vlad_query_summary_{run}.csv"
    recall_csv = mo / f"s7b2_dinov2_vlad_recall_summary_{run}.csv"; scene_csv = mo / f"s7b2_dinov2_vlad_scene_summary_{run}.csv"
    summary_json = ro / f"s7b2_dinov2_vlad_summary_{run}.json"
    recall_fig = fo / f"s7b2_dinov2_vlad_recall_at_k_{run}.png"; top1_fig = fo / f"s7b2_dinov2_vlad_top1_error_hist_{run}.png"; oracle_fig = fo / f"s7b2_dinov2_vlad_oracle_error_hist_{run}.png"
    cand.to_csv(cand_csv, index=False); query.to_csv(query_csv, index=False); recall.to_csv(recall_csv, index=False); scene.to_csv(scene_csv, index=False)
    plot_recall(recall, recall_fig, a.threshold_m); plot_hist(query["top1_error_m_eval_only"], top1_fig, "S7B.2 DINOv2-VLAD top-1 error", "Top-1 error [m], clipped p98"); plot_hist(query["oracle_error_m_eval_only"], oracle_fig, "S7B.2 DINOv2-VLAD oracle error", "Oracle error [m], clipped p98")

    top1_hits = int(query["top1_hit_le_threshold_eval_only"].sum()); topk_col = f"hit_at_{max(ks)}_eval_only"; topk_hits = int(query[topk_col].sum())
    summary = {"stage":"S7B.2_dinov2_vlad_anyloc_style_retrieval","status":"COMPLETE","run_name":run,"repo_root":str(root),"queries":len(q),"satellite_tiles":len(s),"top_k":a.top_k,"threshold_m":a.threshold_m,"recall_ks":ks,"model_name":a.model_name,"image_size":a.image_size,"vlad_clusters":a.vlad_clusters,"vlad_descriptor_dim":int(q_cache.descriptors.shape[1]),"codebook_meta":cbmeta,"satellite_descriptor_runtime_s":s_cache.meta.get("runtime_s"),"query_descriptor_runtime_s":q_cache.meta.get("runtime_s"),"retrieval_runtime_s":rtime,"top1_hits_le_threshold":top1_hits,"top1_hit_rate":float(top1_hits/len(query)),f"top{max(ks)}_hits_le_threshold":topk_hits,f"top{max(ks)}_recall":float(topk_hits/len(query)),"median_top1_error_m_eval_only":float(pd.to_numeric(query["top1_error_m_eval_only"], errors="coerce").median()),"median_oracle_error_m_eval_only":float(pd.to_numeric(query["oracle_error_m_eval_only"], errors="coerce").median()),"median_oracle_rank_eval_only":float(pd.to_numeric(query["oracle_rank_eval_only"], errors="coerce").median()),"outputs":{"candidate_scores_csv":str(cand_csv),"query_summary_csv":str(query_csv),"recall_summary_csv":str(recall_csv),"scene_summary_csv":str(scene_csv),"summary_json":str(summary_json),"environment_json":str(env_path),"codebook_cache":str(cb_path),"query_descriptor_cache":str(q_path),"satellite_descriptor_cache":str(s_path),"recall_figure":str(recall_fig),"top1_error_hist":str(top1_fig),"oracle_error_hist":str(oracle_fig)},"locked_rule":"DINOv2 patch extraction, VLAD codebook fitting, VLAD descriptors, and ranking do not use GT/reference/GNSS; lon/lat/error metrics are computed only after ranking for offline evaluation."}
    summary_json.write_text(json.dumps(summary, indent=2, default=js), encoding="utf-8")

    print("\nS7B.2 — DINOv2-VLAD / AnyLoc-style Retrieval")
    print("---------------------------------------------")
    print("Status:                         COMPLETE")
    print(f"Run name:                       {run}")
    print(f"Device selected:                {ext.device}")
    print(f"Queries:                        {len(q)}")
    print(f"Satellite tiles:                {len(s)}")
    print(f"VLAD clusters:                  {a.vlad_clusters}")
    print(f"VLAD descriptor dim:            {q_cache.descriptors.shape[1]}")
    print(f"Top-K evaluated:                {a.top_k}")
    print(f"Threshold m:                    {a.threshold_m:.1f}")
    print(f"Top-1 hits:                     {top1_hits}/{len(query)}")
    print(f"Top-{max(ks)} hits:                  {topk_hits}/{len(query)}")
    print(f"Median top-1 error m:           {summary['median_top1_error_m_eval_only']:.3f}")
    print(f"Median oracle error m:          {summary['median_oracle_error_m_eval_only']:.3f}")
    print(f"Median oracle rank:             {summary['median_oracle_rank_eval_only']:.3f}")
    print(f"Retrieval runtime s:            {rtime:.3f}\n")
    print("Recall summary:"); print(recall.to_string(index=False)); print("\nScene summary:"); print(scene.to_string(index=False)); print()
    print(f"Candidate scores CSV:           {cand_csv}")
    print(f"Query summary CSV:              {query_csv}")
    print(f"Recall summary CSV:             {recall_csv}")
    print(f"Scene summary CSV:              {scene_csv}")
    print(f"Summary JSON:                   {summary_json}")
    print(f"Environment JSON:               {env_path}")
    print(f"Codebook cache:                 {cb_path}")
    print(f"Satellite VLAD cache:           {s_path}")
    print(f"Query VLAD cache:               {q_path}")
    print(f"Recall figure:                  {recall_fig}")
    print(f"Top1 error hist:                {top1_fig}")
    print(f"Oracle error hist:              {oracle_fig}")
    print("\nLocked rule: GT/reference coordinates used only after ranking for offline evaluation.")

if __name__ == "__main__":
    raise SystemExit(main())
