#!/usr/bin/env python3
"""
S7B.4 / S7D-prep — Failure-token diagnostic and budget-aware rescue analysis

Reads one or more ranked candidate CSV streams, diagnoses anchor-stream failures,
checks stream rescues at a chosen budget, plots budget/scene/failure summaries,
and optionally writes visual panels showing UAV preprocessing, ViT patch-grid,
DINO token maps, VLAD assignment maps, and top/best satellite candidates.

Locked rule: ranking/rescue uses already-ranked retrieval outputs only.
eval_error_m is used only after ranking for offline diagnostics/evaluation.

Command UseD:

PYTHONUNBUFFERED=1 python -u scripts/satloc/s7b/s7b_4_failure_rescue_visual_diagnostics.py \
  --repo-root "$PWD" \
  --stream center=outputs/satloc/metadata/s7b_dinov2_vlad/s7b2_dinov2_vlad_candidate_scores_torch_hub_dinov2_vits14_vlad_k32_img224_top100_q263_sat8625_cb500_full263_k32_cb500_img224.csv \
  --stream resize=outputs/satloc/metadata/s7b_dinov2_vlad/s7b2_dinov2_vlad_candidate_scores_torch_hub_dinov2_vits14_vlad_k32_img224_top100_q263_sat8625_cb500_full263_k32_cb500_img224_resize_square.csv \
  --anchor-stream center \
  --image-stream center \
  --failure-budget 100 \
  --threshold-m 40 \
  --tag center_resize_k32_img224_top100 \
  --make-visual-diagnostics \
  --dino-diagnostics \
  --crop-mode center_square \
  --image-size 224 \
  --device cpu \
  --codebook-npz outputs/satloc/metadata/s7b_dinov2_vlad/s7b2_dinov2_vlad_codebook_torch_hub_dinov2_vits14_vlad_k32_img224_center_square_ppi32_maxp60000_cbb2ec53569bff.npz \
  --diagnostic-scenes agricultural_open_field \
  --max-diagnostic-tokens 24 \
  2>&1 | tee outputs/satloc/reports/s7b_failure_rescue/s7b4_failure_rescue_center_resize_k32_img224_top100.log

"""
from __future__ import annotations

import argparse, json, math, re, sys, warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--stream", action="append", required=True, help="name=path candidate CSV. Repeat.")
    p.add_argument("--anchor-stream", default="center")
    p.add_argument("--image-stream", default="center")
    p.add_argument("--threshold-m", type=float, default=40.0)
    p.add_argument("--failure-budget", type=int, default=100)
    p.add_argument("--budgets", default="1,5,10,20,50,100,150,200")
    p.add_argument("--tag", default="")
    p.add_argument("--metadata-out", type=Path, default=Path("outputs/satloc/metadata/s7b_failure_rescue"))
    p.add_argument("--report-out", type=Path, default=Path("outputs/satloc/reports/s7b_failure_rescue"))
    p.add_argument("--figure-out", type=Path, default=Path("outputs/satloc/figures/s7b_failure_rescue"))

    # Visual diagnostics.
    p.add_argument("--make-visual-diagnostics", action="store_true")
    p.add_argument("--diagnostic-scenes", default="agricultural_open_field")
    p.add_argument("--diagnostic-tokens", default="")
    p.add_argument("--max-diagnostic-tokens", type=int, default=24)

    # DINO / ViT explanatory diagnostics.
    p.add_argument("--dino-diagnostics", action="store_true")
    p.add_argument("--model-name", default="dinov2_vits14")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--crop-mode", choices=["center_square", "resize_square"], default="center_square")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--codebook-npz", type=Path, default=None)
    p.add_argument("--max-orb-keypoints", type=int, default=300)
    return p.parse_args()


def safe_name(x: Any) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower())
    return s.strip("_") or "run"


def resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def js(x: Any) -> Any:
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, float) and not math.isfinite(x): return None
    try:
        if pd.isna(x): return None
    except Exception:
        pass
    return x


def ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def strs(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def parse_streams(items: list[str], root: Path) -> list[tuple[str, Path]]:
    out, seen = [], set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"--stream must be name=path, got {item}")
        n, p = item.split("=", 1)
        n = safe_name(n)
        if n in seen: raise ValueError(f"duplicate stream {n}")
        seen.add(n)
        path = resolve(root, Path(p))
        if not path.exists(): raise FileNotFoundError(path)
        out.append((n, path))
    return out


def score_col(df: pd.DataFrame) -> str | None:
    for c in ["dinov2_vlad_similarity", "dinov2_similarity", "similarity", "rrf_score", "score"]:
        if c in df.columns: return c
    for c in df.columns:
        l = c.lower()
        if "similarity" in l or l.endswith("_score"):
            return c
    return None


def load_stream(name: str, path: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "rank" not in df.columns and "union_rank" in df.columns:
        df = df.rename(columns={"union_rank": "rank"})
    req = ["token", "rank", "tile_id", "eval_error_m"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing {miss}; columns={list(df.columns)}")
    sc = score_col(df)
    out = pd.DataFrame({
        "stream": name,
        "token": pd.to_numeric(df["token"], errors="raise").astype(int),
        "rank": pd.to_numeric(df["rank"], errors="raise").astype(int),
        "tile_id": df["tile_id"].astype(str),
        "score": pd.to_numeric(df[sc], errors="coerce") if sc else np.nan,
        "eval_error_m": pd.to_numeric(df["eval_error_m"], errors="coerce"),
        "primary_scene": df["primary_scene"].fillna("unlabeled").astype(str) if "primary_scene" in df.columns else "unlabeled",
    })
    for c in ["query_image_path", "sat_image_path", "present_streams"]:
        if c in df.columns: out[c] = df[c]
    out["hit_eval_only"] = out["eval_error_m"].le(threshold)
    return out.drop_duplicates(["stream", "token", "tile_id"], keep="first").sort_values(["stream", "token", "rank"])


def per_stream_summary(all_df: pd.DataFrame, budgets: list[int], threshold: float) -> pd.DataFrame:
    rows = []
    for (stream, token), g in all_df.groupby(["stream", "token"], sort=True):
        g = g.sort_values("rank")
        e = pd.to_numeric(g["eval_error_m"], errors="coerce").to_numpy(float)
        r = pd.to_numeric(g["rank"], errors="coerce").to_numpy(float)
        row = {
            "stream": stream, "token": int(token),
            "primary_scene": g["primary_scene"].dropna().astype(str).iloc[0],
            "max_available_rank": int(np.nanmax(r)) if len(r) else 0,
            "top1_error_m": float(e[0]) if len(e) and np.isfinite(e[0]) else np.nan,
            "top1_hit": bool(len(e) and np.isfinite(e[0]) and e[0] <= threshold),
        }
        if len(e) and np.isfinite(e).any():
            p = int(np.nanargmin(e))
            row["best_error_m_available"] = float(e[p])
            row["best_rank_available"] = int(r[p])
            row["best_tile_id_available"] = str(g.iloc[p]["tile_id"])
        else:
            row["best_error_m_available"] = np.nan
            row["best_rank_available"] = np.nan
            row["best_tile_id_available"] = ""
        for b in budgets:
            gb = g[g["rank"] <= b]
            eb = pd.to_numeric(gb["eval_error_m"], errors="coerce").to_numpy(float)
            row[f"hit_at_{b}"] = bool(len(eb) and np.isfinite(eb).any() and np.nanmin(eb) <= threshold)
            row[f"min_error_at_{b}_m"] = float(np.nanmin(eb)) if len(eb) and np.isfinite(eb).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def budget_recall(per: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows = []
    for stream, g in per.groupby("stream", sort=True):
        q = int(g["token"].nunique())
        for b in budgets:
            col = f"hit_at_{b}"
            hits = int(g[col].sum()) if col in g.columns else 0
            rows.append({"stream": stream, "budget_k": b, "hits": hits, "queries": q, "recall": hits/q if q else 0.0})
    return pd.DataFrame(rows)


def scene_recall(per: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows = []
    for (stream, scene), g in per.groupby(["stream", "primary_scene"], sort=True):
        row = {"stream": stream, "primary_scene": scene, "queries": int(len(g))}
        row["top1_hits"] = int(g["top1_hit"].sum())
        row["median_top1_error_m"] = float(pd.to_numeric(g["top1_error_m"], errors="coerce").median())
        row["median_best_error_m_available"] = float(pd.to_numeric(g["best_error_m_available"], errors="coerce").median())
        row["median_best_rank_available"] = float(pd.to_numeric(g["best_rank_available"], errors="coerce").median())
        for b in budgets:
            col = f"hit_at_{b}"
            row[f"hits_at_{b}"] = int(g[col].sum())
            row[f"recall_at_{b}"] = float(g[col].mean()) if len(g) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def failure_table(per: pd.DataFrame, anchor: str, budget: int, threshold: float) -> pd.DataFrame:
    streams = sorted(per["stream"].unique())
    adf = per[per["stream"] == anchor]
    if adf.empty: raise ValueError(f"anchor stream {anchor} not found; streams={streams}")
    rows = []
    for _, a in adf.iterrows():
        token = int(a["token"])
        anchor_hit = bool(a[f"hit_at_{budget}"])
        row = {
            "token": token,
            "primary_scene": a["primary_scene"],
            "anchor_stream": anchor,
            f"anchor_hit_at_{budget}": anchor_hit,
            "anchor_top1_error_m": a["top1_error_m"],
            "anchor_best_error_m_available": a["best_error_m_available"],
            "anchor_best_rank_available": a["best_rank_available"],
        }
        any_hit, rescue = False, []
        best_err, best_rank, best_stream, best_tile = np.inf, np.nan, "", ""
        for s in streams:
            sg = per[(per["stream"] == s) & (per["token"] == token)]
            if sg.empty:
                row[f"{s}_present"] = False
                row[f"{s}_hit_at_{budget}"] = False
                continue
            sr = sg.iloc[0]
            hit = bool(sr[f"hit_at_{budget}"])
            row[f"{s}_present"] = True
            row[f"{s}_hit_at_{budget}"] = hit
            row[f"{s}_top1_error_m"] = sr["top1_error_m"]
            row[f"{s}_best_error_m_available"] = sr["best_error_m_available"]
            row[f"{s}_best_rank_available"] = sr["best_rank_available"]
            if hit:
                any_hit = True
                if s != anchor and not anchor_hit: rescue.append(s)
            e = sr["best_error_m_available"]
            if pd.notna(e) and float(e) < best_err:
                best_err, best_rank, best_stream, best_tile = float(e), sr["best_rank_available"], s, sr["best_tile_id_available"]
        row[f"any_stream_hit_at_{budget}"] = any_hit
        row[f"rescued_by_non_anchor_at_{budget}"] = bool((not anchor_hit) and rescue)
        row[f"rescue_streams_at_{budget}"] = ",".join(rescue)
        row["best_stream_available"] = best_stream
        row["best_error_any_stream_available_m"] = best_err if np.isfinite(best_err) else np.nan
        row["best_rank_any_stream_available"] = best_rank
        row["best_tile_any_stream_available"] = best_tile
        if anchor_hit:
            group = "anchor_success"
        elif rescue:
            group = "non_anchor_rescue"
        elif np.isfinite(best_err) and best_err <= threshold:
            group = "budget_rank_miss"
        elif np.isfinite(best_err) and best_err <= 100:
            group = "near_pool_miss_40_100m"
        else:
            group = "pool_failure_or_far_ambiguity"
        row["failure_group"] = group
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["failure_group", "primary_scene", "token"])


def plot_budget(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9,5))
    for stream, g in df.groupby("stream", sort=True):
        g = g.sort_values("budget_k")
        ax.plot(g["budget_k"], g["recall"], marker="o", label=stream)
    ax.set_xlabel("Candidate budget K"); ax.set_ylabel("Recall@K")
    ax.set_ylim(0,1); ax.set_title("S7B.4 budget-aware recall")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=180); plt.close(fig)


def plot_counts(series: pd.Series, out: Path, title: str, rotate=25):
    c = series.value_counts().sort_values(ascending=False)
    if c.empty: return
    fig, ax = plt.subplots(figsize=(9.5,5))
    ax.bar(c.index.astype(str), c.values); ax.set_title(title); ax.set_ylabel("Tokens")
    ax.tick_params(axis="x", rotation=rotate); ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=180); plt.close(fig)


def plot_hist(series: pd.Series, out: Path, title: str, xlabel: str):
    v = pd.to_numeric(series, errors="coerce").dropna()
    if v.empty: return
    fig, ax = plt.subplots(figsize=(8.5,5))
    ax.hist(v, bins=min(40, max(10, int(math.sqrt(len(v))))))
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Tokens")
    ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=180); plt.close(fig)


def preprocess(img: Image.Image, image_size: int, crop_mode: str) -> Image.Image:
    img = img.convert("RGB")
    if crop_mode == "center_square":
        w,h = img.size; side = min(w,h); l=(w-side)//2; t=(h-side)//2
        img = img.crop((l,t,l+side,t+side))
    res = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    return img.resize((image_size,image_size), res)


def read_panel(path: str, size=(320,320)) -> Image.Image:
    if not path or not Path(path).exists():
        img = Image.new("RGB", size, (245,245,245)); ImageDraw.Draw(img).text((15,15), "missing", fill=(0,0,0)); return img
    img = Image.open(path).convert("RGB"); img.thumbnail(size)
    can = Image.new("RGB", size, (245,245,245)); can.paste(img, ((size[0]-img.width)//2, (size[1]-img.height)//2)); return can


def title(img: Image.Image, text: str, sub: str="") -> Image.Image:
    bh = 46 if sub else 30; out = Image.new("RGB", (img.width, img.height+bh), "white"); out.paste(img, (0,bh))
    d = ImageDraw.Draw(out); d.text((8,5), text[:60], fill=(0,0,0))
    if sub: d.text((8,24), sub[:60], fill=(0,0,0))
    return out


def grid_overlay(img: Image.Image, grid: int) -> Image.Image:
    out = img.resize((320,320)); d = ImageDraw.Draw(out); w,h=out.size
    for i in range(1, grid):
        x=int(i*w/grid); y=int(i*h/grid)
        d.line((x,0,x,h), fill=(255,255,255)); d.line((0,y,w,y), fill=(255,255,255))
    return out


def orb_overlay(img: Image.Image, max_kp: int) -> Image.Image:
    try:
        import cv2
        arr = np.asarray(img.convert("RGB")); gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=max_kp); kps = orb.detect(gray, None)
        drawn = cv2.drawKeypoints(arr, kps, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        return Image.fromarray(drawn).resize((320,320))
    except Exception:
        out = img.resize((320,320)); ImageDraw.Draw(out).text((8,8), "ORB unavailable", fill=(0,0,0)); return out


def arr_to_pil(a: np.ndarray, cmap=None) -> Image.Image:
    fig, ax = plt.subplots(figsize=(3.2,3.2), dpi=100); ax.imshow(a, cmap=cmap); ax.axis("off"); fig.tight_layout(pad=0); fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:,:,:3]; plt.close(fig); return Image.fromarray(arr)


def sheet(panels: list[Image.Image], cols=3) -> Image.Image:
    if not panels: return Image.new("RGB", (1,1), "white")
    cw = max(p.width for p in panels); ch = max(p.height for p in panels); rows = int(math.ceil(len(panels)/cols))
    out = Image.new("RGB", (cw*cols, ch*rows), "white")
    for i,p in enumerate(panels):
        r,c = divmod(i, cols); out.paste(p, (c*cw+(cw-p.width)//2, r*ch+(ch-p.height)//2))
    return out


class DinoVis:
    def __init__(self, args):
        import torch
        self.torch = torch; self.args = args; self.device = self._device(args.device)
        print("[S7B.4] Loading DINOv2 for visual diagnostics...")
        self.model = torch.hub.load("facebookresearch/dinov2", args.model_name, pretrained=True).eval().to(self.device)
        self.centers = None
        if args.codebook_npz:
            p = resolve(args.repo_root, args.codebook_npz)
            if p.exists():
                z = np.load(p, allow_pickle=False)
                if "centers" in z:
                    self.centers = self._l2(z["centers"].astype(np.float32)); print(f"[S7B.4] Loaded codebook: {p}")
    def _device(self, req):
        t = self.torch
        if req == "cuda": return "cuda"
        if req == "mps": return "mps"
        if req == "cpu": return "cpu"
        if t.cuda.is_available(): return "cuda"
        if hasattr(t.backends, "mps") and t.backends.mps.is_available(): return "mps"
        return "cpu"
    @staticmethod
    def _l2(x, axis=1, eps=1e-12): return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)
    def tokens(self, path: str):
        proc = preprocess(Image.open(path).convert("RGB"), self.args.image_size, self.args.crop_mode)
        arr = np.asarray(proc).astype(np.float32)/255.0
        mean=np.array([0.485,0.456,0.406], np.float32); std=np.array([0.229,0.224,0.225], np.float32)
        arr=(arr-mean)/std; arr=np.transpose(arr,(2,0,1))
        b=self.torch.from_numpy(arr[None]).to(self.device)
        with self.torch.inference_mode():
            out=self.model.forward_features(b); tok=out["x_norm_patchtokens"][0].detach().float().cpu().numpy().astype(np.float32)
        g=int(round(math.sqrt(tok.shape[0]))); return proc, tok, g
    def pca_rgb(self, tok, grid):
        x=tok-tok.mean(0, keepdims=True)
        try:
            _,_,vh=np.linalg.svd(x, full_matrices=False); z=x@vh[:3].T
        except Exception: z=x[:,:3]
        z=z.reshape(grid,grid,3); z-=z.min((0,1), keepdims=True); z/=np.maximum(z.max((0,1), keepdims=True),1e-9); return z
    def deviation(self, tok, grid):
        z=np.linalg.norm(tok-tok.mean(0,keepdims=True), axis=1).reshape(grid,grid); z=(z-z.min())/max(z.max()-z.min(),1e-9); return z
    def assign(self, tok, grid):
        if self.centers is None: return None
        sim=self._l2(tok.astype(np.float32))@self.centers.T; return np.argmax(sim, axis=1).reshape(grid,grid)


def pick_tokens(fail: pd.DataFrame, args) -> list[int]:
    picked=[]
    def add(ts):
        for t in ts:
            ti=int(t)
            if ti not in picked: picked.append(ti)
    if args.diagnostic_tokens: add(ints(args.diagnostic_tokens))
    scenes=set(strs(args.diagnostic_scenes))
    if scenes:
        add(fail[fail["primary_scene"].isin(scenes)].sort_values(["failure_group","token"])["token"].tolist())
    miss = fail[~fail[f"anchor_hit_at_{args.failure_budget}"]].copy()
    add(miss.sort_values("best_error_any_stream_available_m", ascending=False)["token"].tolist())
    add(miss.sort_values("best_error_any_stream_available_m", ascending=True)["token"].tolist())
    return picked[:args.max_diagnostic_tokens]


def row_for(all_df, token, stream=None, rank=1):
    d=all_df[all_df["token"]==token]
    if stream: d=d[d["stream"]==stream]
    d=d.sort_values("rank")
    if d.empty: return None
    e=d[d["rank"]==rank]
    return (e.iloc[0] if not e.empty else d.iloc[0])


def best_row(all_df, token):
    d=all_df[all_df["token"]==token].copy()
    if d.empty: return None
    d["err_num"]=pd.to_numeric(d["eval_error_m"], errors="coerce")
    return d.sort_values(["err_num","rank"]).iloc[0]


def make_visuals(all_df, fail, args, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = pick_tokens(fail, args); print(f"[S7B.4] Visual tokens: {tokens}")
    dvis = DinoVis(args) if args.dino_diagnostics else None
    outputs=[]
    for token in tokens:
        top = row_for(all_df, token, args.image_stream, 1)
        if top is None:
            top = row_for(all_df, token, None, 1)
        best=best_row(all_df, token)
        if top is None: continue
        qpath=str(top.get("query_image_path", "")); scene=str(top.get("primary_scene","unlabeled"))
        panels=[title(read_panel(qpath), f"token {token} UAV original", scene)]
        if qpath and Path(qpath).exists():
            proc=preprocess(Image.open(qpath), args.image_size, args.crop_mode)
            grid=max(1,args.image_size//14)
            panels.append(title(proc.resize((320,320)), f"DINO input {args.crop_mode}", f"{args.image_size}x{args.image_size}"))
            panels.append(title(grid_overlay(proc, grid), "ViT patch grid", f"{grid}x{grid} patch tokens"))
            panels.append(title(orb_overlay(proc, args.max_orb_keypoints), "ORB keypoints contrast", "classical sparse features"))
        if dvis is not None and qpath and Path(qpath).exists():
            try:
                proc,tok,g=dvis.tokens(qpath)
                panels.append(title(arr_to_pil(dvis.pca_rgb(tok,g)), "DINO token PCA", "patch embedding map"))
                panels.append(title(arr_to_pil(dvis.deviation(tok,g), cmap="viridis"), "DINO token deviation", "distinctiveness proxy"))
                a=dvis.assign(tok,g)
                if a is not None: panels.append(title(arr_to_pil(a, cmap="tab20"), "VLAD cluster assignment", "nearest codeword/patch"))
            except Exception as exc:
                warnings.warn(f"DINO visual failed token {token}: {exc}")
        top_err = top.get("eval_error_m", np.nan)
        if pd.notna(top_err):
            top_subtitle = f"err={float(top_err):.1f}m rank={top.get('rank','')}"
        else:
            top_subtitle = "err=nan"
        panels.append(title(read_panel(str(top.get("sat_image_path",""))), f"{args.image_stream} top-1 sat", top_subtitle))
        if best is not None:
            be=best.get("eval_error_m", np.nan)
            panels.append(title(read_panel(str(best.get("sat_image_path",""))), "best available candidate", f"{best.get('stream','')} rank={best.get('rank','')} err={float(be):.1f}m" if pd.notna(be) else "err=nan"))
        out=out_dir / f"s7b4_visual_token_{int(token):04d}_{safe_name(scene)}.png"
        sheet(panels, cols=3).save(out); outputs.append(str(out)); print(f"[S7B.4] Wrote {out}")
    return outputs


def main():
    args=parse_args(); root=args.repo_root.resolve(); args.repo_root=root
    md=resolve(root,args.metadata_out); rp=resolve(root,args.report_out); fg=resolve(root,args.figure_out)
    for d in [md,rp,fg]: d.mkdir(parents=True, exist_ok=True)
    streams=parse_streams(args.stream, root)
    tag=safe_name(args.tag or "_".join(n for n,_ in streams))
    requested=ints(args.budgets)
    print("\nS7B.4 / S7D-prep — Failure-token diagnostic and budget-aware rescue analysis")
    print("---------------------------------------------------------------------------")
    print(f"Repository root:        {root}")
    print(f"Tag:                    {tag}")
    print(f"Anchor stream:          {args.anchor_stream}")
    print(f"Threshold m:            {args.threshold_m}")
    print(f"Failure budget:         {args.failure_budget}")
    dfs=[]; max_rank=0
    for n,p in streams:
        df=load_stream(n,p,args.threshold_m); dfs.append(df); max_rank=max(max_rank,int(df["rank"].max()))
        print(f"[S7B.4] Loaded {n}: rows={len(df)} tokens={df['token'].nunique()} max_rank={df['rank'].max()}")
    all_df=pd.concat(dfs, ignore_index=True)
    budgets=sorted(set([b for b in requested if b<=max_rank] + ([args.failure_budget] if args.failure_budget<=max_rank else [])))
    if max(requested)>max_rank: print(f"[S7B.4][WARN] available ranks only up to {max_rank}; higher budgets clipped.")
    if args.failure_budget>max_rank: raise ValueError(f"failure-budget {args.failure_budget} > max available rank {max_rank}")
    per=per_stream_summary(all_df,budgets,args.threshold_m)
    brec=budget_recall(per,budgets)
    srec=scene_recall(per,budgets)
    fail=failure_table(per,args.anchor_stream,args.failure_budget,args.threshold_m)
    misses=fail[~fail[f"anchor_hit_at_{args.failure_budget}"]].copy()

    per_csv=md/f"s7b4_per_stream_token_summary_{tag}.csv"; brec_csv=md/f"s7b4_budget_recall_curves_{tag}.csv"
    srec_csv=md/f"s7b4_scene_budget_recall_{tag}.csv"; fail_csv=md/f"s7b4_failure_token_diagnostics_{tag}.csv"
    miss_csv=md/f"s7b4_anchor_miss_shortlist_{tag}.csv"; summary_json=rp/f"s7b4_failure_rescue_summary_{tag}.json"
    per.to_csv(per_csv,index=False); brec.to_csv(brec_csv,index=False); srec.to_csv(srec_csv,index=False); fail.to_csv(fail_csv,index=False); misses.to_csv(miss_csv,index=False)
    budget_fig=fg/f"s7b4_budget_recall_curves_{tag}.png"; group_fig=fg/f"s7b4_failure_group_counts_{tag}.png"
    scene_fig=fg/f"s7b4_anchor_miss_scene_counts_{tag}.png"; rank_fig=fg/f"s7b4_anchor_miss_best_rank_hist_{tag}.png"; err_fig=fg/f"s7b4_anchor_miss_best_error_hist_{tag}.png"
    plot_budget(brec,budget_fig); plot_counts(fail["failure_group"],group_fig,"S7B.4 diagnostic groups")
    plot_counts(misses["primary_scene"],scene_fig,f"S7B.4 {args.anchor_stream} misses by scene")
    plot_hist(misses["best_rank_any_stream_available"],rank_fig,"Anchor misses: best available rank","rank")
    plot_hist(misses["best_error_any_stream_available_m"],err_fig,"Anchor misses: best available error","error [m]")
    visuals=[]
    if args.make_visual_diagnostics:
        visuals=make_visuals(all_df, fail, args, fg/"visual_diagnostics"/tag)
    group_summary=fail["failure_group"].value_counts().rename_axis("failure_group").reset_index(name="count")
    scene_miss=misses["primary_scene"].value_counts().rename_axis("primary_scene").reset_index(name="anchor_miss_count")
    total=fail["token"].nunique(); anchor_hits=int(fail[f"anchor_hit_at_{args.failure_budget}"].sum())
    any_hits=int(fail[f"any_stream_hit_at_{args.failure_budget}"].sum()); rescues=int(fail[f"rescued_by_non_anchor_at_{args.failure_budget}"].sum())
    summary={
        "stage":"S7B.4_S7D_prep_failure_rescue_visual_diagnostics", "status":"COMPLETE", "tag":tag,
        "streams":[{"name":n,"path":str(p)} for n,p in streams], "anchor_stream":args.anchor_stream,
        "threshold_m":args.threshold_m, "failure_budget":args.failure_budget, "available_max_rank":max_rank, "budgets_used":budgets,
        "tokens":total, f"{args.anchor_stream}_hits_at_{args.failure_budget}":anchor_hits,
        f"any_stream_hits_at_{args.failure_budget}":any_hits, f"non_anchor_rescues_at_{args.failure_budget}":rescues,
        "anchor_miss_count":int(len(misses)), "group_summary":group_summary.to_dict(orient="records"),
        "scene_anchor_miss_summary":scene_miss.to_dict(orient="records"),
        "outputs":{"per_stream_token_summary_csv":str(per_csv),"budget_recall_curves_csv":str(brec_csv),"scene_budget_recall_csv":str(srec_csv),"failure_token_diagnostics_csv":str(fail_csv),"anchor_miss_shortlist_csv":str(miss_csv),"summary_json":str(summary_json),"budget_recall_figure":str(budget_fig),"failure_group_figure":str(group_fig),"anchor_miss_scene_figure":str(scene_fig),"anchor_miss_best_rank_hist":str(rank_fig),"anchor_miss_best_error_hist":str(err_fig),"visual_diagnostics":visuals},
        "visual_note":"DINOv2 is a ViT patch-token model, not a sparse keypoint detector. Panels show ViT input, patch grid, token PCA/deviation maps, optional VLAD assignments, and ORB keypoints only as classical contrast.",
        "locked_rule":"GT/error used only after ranking for offline diagnostics/evaluation. Visualizations are explanatory only."
    }
    summary_json.write_text(json.dumps(summary, indent=2, default=js), encoding="utf-8")
    print("\nS7B.4 / S7D-prep — Failure-token diagnostic and budget-aware rescue analysis")
    print("---------------------------------------------------------------------------")
    print("Status:                         COMPLETE")
    print(f"Tag:                            {tag}")
    print(f"Streams:                        {', '.join(n for n,_ in streams)}")
    print(f"Tokens:                         {total}")
    print(f"Available max rank:             {max_rank}")
    print(f"Budgets used:                   {budgets}")
    print(f"Anchor hits@{args.failure_budget}:              {anchor_hits}/{total}")
    print(f"Any-stream hits@{args.failure_budget}:          {any_hits}/{total}")
    print(f"Non-anchor rescues@{args.failure_budget}:       {rescues}")
    print(f"Anchor misses@{args.failure_budget}:            {len(misses)}")
    print("\nFailure group summary:"); print(group_summary.to_string(index=False))
    print("\nAnchor miss scene summary:"); print(scene_miss.to_string(index=False))
    print("\nBudget recall curves:"); print(brec.to_string(index=False))
    print(f"\nPer-stream token summary CSV:    {per_csv}")
    print(f"Budget recall curves CSV:        {brec_csv}")
    print(f"Scene budget recall CSV:         {srec_csv}")
    print(f"Failure token diagnostics CSV:   {fail_csv}")
    print(f"Anchor miss shortlist CSV:       {miss_csv}")
    print(f"Summary JSON:                    {summary_json}")
    print(f"Budget recall figure:            {budget_fig}")
    print(f"Failure group figure:            {group_fig}")
    print(f"Anchor miss scene figure:        {scene_fig}")
    print(f"Best rank hist figure:           {rank_fig}")
    print(f"Best error hist figure:          {err_fig}")
    if visuals:
        print(f"Visual diagnostic figures:       {len(visuals)}")
        for p in visuals[:20]: print(f"  {p}")
    print("\nLocked rule: GT/error used only after ranking for offline diagnostics.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
