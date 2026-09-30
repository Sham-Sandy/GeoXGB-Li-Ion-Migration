
from __future__ import annotations
import logging, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ase.io import read as ase_read

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nebDFT2k"
INDEX = DATA / "nebDFT2k_index.csv"
OUT = ROOT / "processed"
OUT.mkdir(parents=True, exist_ok=True)

DESCRIPTORS = OUT / "descriptors_all.csv"
OFFICIAL = OUT / "official_1681_dataset.csv"

SEED = 42
np.random.seed(SEED)


def setup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

def load_index():
    if not INDEX.exists():
        raise FileNotFoundError(f"Missing: {INDEX}")
    df = pd.read_csv(INDEX)
    req = {"material_id","edge_id","em_dft","_split"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"Index missing: {sorted(miss)}")
    df["material_id"] = df["material_id"].astype(str)
    df["edge_id"] = df["edge_id"].astype(str)
    df["event_key"] = df["material_id"].str.strip() + "::" + df["edge_id"].str.strip()

    def norm(x):
        s=str(x).strip().lower()
        if s in {"train","training"}: return "train"
        if s in {"val","valid","validation"}: return "val"
        if s in {"test","testing"}: return "test"
        return np.nan

    df["split"] = df["_split"].map(norm)
    if df["split"].isna().any():
        raise RuntimeError("Unknown split labels.")
    if len(df) != 1681:
        raise RuntimeError(f"Expected 1681, got {len(df)}")
    if tuple((df["split"]==s).sum() for s in ("train","val","test")) != (1220,241,220):
        raise RuntimeError("Official split mismatch.")
    return df

def parse_xyz_frames(path: Path):
    text = path.read_text(errors="ignore").splitlines()
    frames = []
    i = 0
    while i < len(text):
        line = text[i].strip()
        if not line:
            i += 1
            continue
        try:
            n = int(line)
        except Exception:
            i += 1
            continue
        if n <= 0 or i + 1 + n >= len(text):
            break
        start = i + 2
        end = start + n
        symbols, coords = [], []
        for row in text[start:end]:
            parts = row.split()
            if len(parts) < 4:
                continue
            sym = parts[0]
            try:
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            except Exception:
                continue
            symbols.append(sym)
            coords.append(xyz)
        if len(coords) == n:
            frames.append((symbols, np.asarray(coords, dtype=float)))
        i = end
    return frames

def parse_event_key_from_filename(path: Path, index: pd.DataFrame):
    stem = path.name
    if stem.endswith("_init.xyz"):
        stem = stem[:-9]
    for _, row in index.iterrows():
        mid = str(row["material_id"])
        eid = str(row["edge_id"])
        if mid in stem and eid in stem:
            return f"{mid}::{eid}"
    matches = index[index["material_id"].astype(str).map(lambda x: x in stem)]
    if len(matches) == 1:
        r = matches.iloc[0]
        return f"{r['material_id']}::{r['edge_id']}"
    return None

def pair_distances(coords: np.ndarray, center_idx: int) -> np.ndarray:
    if len(coords) <= 1:
        return np.empty(0, dtype=float)
    c = coords[center_idx]
    d = np.linalg.norm(coords - c[None,:], axis=1)
    return np.delete(d, center_idx)

def choose_mobile_indices(symbols: List[str]) -> List[int]:
    for el in ("Li","Na","K"):
        idx = [i for i,s in enumerate(symbols) if s == el]
        if idx:
            return idx
    return []

def local_descriptor(symbols, coords, center_idx) -> Dict[str,float]:
    d = pair_distances(coords, center_idx)
    if len(d) == 0:
        return {
            "mobile_coordination_2p8":0.0,
            "mobile_coordination_3p2":0.0,
            "mobile_coordination_4p0":0.0,
            "mobile_min_host_distance":np.nan,
            "mobile_mean_host_distance":np.nan,
            "mobile_std_host_distance":np.nan,
            "mobile_p10_distance":np.nan,
            "mobile_p25_distance":np.nan,
            "mobile_p50_distance":np.nan,
            "mobile_p75_distance":np.nan,
            "mobile_p90_distance":np.nan,
            "mobile_local_density":0.0,
        }

    mobile_set = set(choose_mobile_indices(symbols))
    host_idx = [
        i for i in range(len(symbols))
        if i != center_idx and i not in mobile_set
    ]
    if host_idx:
        hd = np.linalg.norm(
            coords[host_idx] - coords[center_idx][None,:],
            axis=1
        )
    else:
        hd = d

    hd = np.asarray(hd,dtype=float)
    if len(hd)==0:
        hd=d

    return {
        "mobile_coordination_2p8": float(np.sum(hd <= 2.8)),
        "mobile_coordination_3p2": float(np.sum(hd <= 3.2)),
        "mobile_coordination_4p0": float(np.sum(hd <= 4.0)),
        "mobile_min_host_distance": float(np.min(hd)),
        "mobile_mean_host_distance": float(np.mean(hd)),
        "mobile_std_host_distance": float(np.std(hd)),
        "mobile_p10_distance": float(np.percentile(hd,10)),
        "mobile_p25_distance": float(np.percentile(hd,25)),
        "mobile_p50_distance": float(np.percentile(hd,50)),
        "mobile_p75_distance": float(np.percentile(hd,75)),
        "mobile_p90_distance": float(np.percentile(hd,90)),
        "mobile_local_density": float(np.sum(hd <= 4.0)/(4.0**3)),
    }

def frame_geometry_descriptors(symbols, coords):
    mobile = choose_mobile_indices(symbols)
    out = {
        "n_atoms": float(len(symbols)),
        "n_mobile": float(len(mobile)),
        "structure_span_x": float(np.ptp(coords[:,0])) if len(coords) else np.nan,
        "structure_span_y": float(np.ptp(coords[:,1])) if len(coords) else np.nan,
        "structure_span_z": float(np.ptp(coords[:,2])) if len(coords) else np.nan,
    }

    if not mobile:
        out.update({
            "mobile_pair_distance_min":np.nan,
            "mobile_pair_distance_max":np.nan,
            "mobile_pair_distance_mean":np.nan,
            "mobile_centroid_spread":np.nan,
        })
        return out

    local_all = [
        local_descriptor(symbols,coords,idx)
        for idx in mobile
    ]
    keys = list(local_all[0].keys())
    for k in keys:
        vals = [float(x[k]) for x in local_all if np.isfinite(x[k])]
        out[k] = float(np.mean(vals)) if vals else np.nan

    if len(mobile) >= 2:
        mp = coords[mobile]
        pdist=[]
        for i in range(len(mp)):
            for j in range(i+1,len(mp)):
                pdist.append(float(np.linalg.norm(mp[i]-mp[j])))
        out["mobile_pair_distance_min"] = min(pdist)
        out["mobile_pair_distance_max"] = max(pdist)
        out["mobile_pair_distance_mean"] = float(np.mean(pdist))
        out["mobile_centroid_spread"] = float(
            np.mean(np.linalg.norm(mp - np.mean(mp,axis=0),axis=1))
        )
    else:
        out["mobile_pair_distance_min"]=np.nan
        out["mobile_pair_distance_max"]=np.nan
        out["mobile_pair_distance_mean"]=np.nan
        out["mobile_centroid_spread"]=0.0

    return out

def safe_float(x, default=np.nan):
    try:
        v=float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def trajectory_descriptors(frames):
    out = {
        "path_n_frames":float(len(frames)),
        "path_hop_count":np.nan,
        "path_total_length":np.nan,
        "path_direct_distance":np.nan,
        "path_tortuosity":np.nan,
        "path_mean_hop_distance":np.nan,
        "path_max_hop_distance":np.nan,
        "path_hop_distance_std":np.nan,
        "path_bottleneck_min":np.nan,
        "path_bottleneck_mean":np.nan,
        "path_bottleneck_std":np.nan,
        "path_bottleneck_max":np.nan,
        "path_coordination_mean":np.nan,
        "path_coordination_std":np.nan,
        "path_coordination_change":np.nan,
    }
    if len(frames) < 2:
        return out

    mobile_positions=[]
    prev=None
    for symbols,coords in frames:
        candidates=choose_mobile_indices(symbols)
        if not candidates:
            return out
        if prev is None:
            idx=candidates[0]
        else:
            pts=coords[candidates]
            idx=candidates[int(np.argmin(
                np.linalg.norm(pts-prev[None,:],axis=1)
            ))]
        pos=coords[idx].astype(float)
        mobile_positions.append(pos)
        prev=pos

    P=np.asarray(mobile_positions)
    hops=np.linalg.norm(np.diff(P,axis=0),axis=1)
    direct=float(np.linalg.norm(P[-1]-P[0]))
    total=float(np.sum(hops))

    out["path_hop_count"]=float(len(hops))
    out["path_total_length"]=total
    out["path_direct_distance"]=direct
    out["path_tortuosity"]=float(total/max(direct,1e-8))
    out["path_mean_hop_distance"]=float(np.mean(hops))
    out["path_max_hop_distance"]=float(np.max(hops))
    out["path_hop_distance_std"]=float(np.std(hops))

    clearances=[]
    coordinations=[]
    for frame_idx,(symbols,coords) in enumerate(frames):
        candidates=choose_mobile_indices(symbols)
        if not candidates:
            continue
        mobile_pos=P[frame_idx]
        all_mobile=np.asarray([coords[i] for i in candidates])
        tracked=int(np.argmin(
            np.linalg.norm(all_mobile-mobile_pos[None,:],axis=1)
        ))
        mobile_idx=candidates[tracked]
        mobile_set=set(candidates)
        host=[i for i in range(len(symbols)) if i not in mobile_set]
        if not host:
            continue
        hd=np.linalg.norm(
            coords[host]-coords[mobile_idx][None,:],axis=1
        )
        if len(hd):
            clearances.append(float(np.min(hd)))
            coordinations.append(float(np.sum(hd <= 3.2)))

    if clearances:
        out["path_bottleneck_min"]=float(np.min(clearances))
        out["path_bottleneck_mean"]=float(np.mean(clearances))
        out["path_bottleneck_std"]=float(np.std(clearances))
        out["path_bottleneck_max"]=float(np.max(clearances))
    if coordinations:
        out["path_coordination_mean"]=float(np.mean(coordinations))
        out["path_coordination_std"]=float(np.std(coordinations))
        out["path_coordination_change"]=float(
            coordinations[-1]-coordinations[0]
        )
    return out

def main():
    setup()
    index=load_index()
    files=sorted(DATA.rglob("*_init.xyz"))
    logging.info(f"XYZ files found: {len(files)}")
    rows=[]
    matched=0

    for n,path in enumerate(files,1):
        event_key=parse_event_key_from_filename(path,index)
        if event_key is None:
            continue
        frames=parse_xyz_frames(path)
        if not frames:
            continue

        symbols0,coords0=frames[0]
        desc=frame_geometry_descriptors(symbols0,coords0)
        desc.update(trajectory_descriptors(frames))
        desc["event_key"]=event_key
        desc["xyz_file"]=str(path)
        desc["xyz_n_frames"]=len(frames)
        rows.append(desc)
        matched+=1

        if n % 250 == 0:
            logging.info(f"Descriptors {n}/{len(files)} | matched={matched}")

    out=pd.DataFrame(rows)
    if len(out) != 1681:
        raise RuntimeError(
            f"Expected 1681 descriptor rows, got {len(out)}"
        )
    if out["event_key"].duplicated().any():
        raise RuntimeError("Duplicate event keys detected.")

    numeric=[
        c for c in out.columns
        if c not in {"event_key","xyz_file","xyz_n_frames"}
        and pd.api.types.is_numeric_dtype(out[c])
    ]

    if len(numeric) != 36:
        raise RuntimeError(
            f"Expected exactly 36 numerical descriptors, got {len(numeric)}"
        )

    out.to_csv(DESCRIPTORS,index=False)

    merged=index.merge(out,on="event_key",how="left",validate="one_to_one")
    merged.to_csv(OUT/"official_descriptors.csv",index=False)

    (OUT/"feature_list.txt").write_text(
        "\n".join(numeric),
        encoding="utf-8"
    )

    logging.info("="*90)
    logging.info("DESCRIPTOR GENERATION COMPLETE")
    logging.info("Events       : %d",len(out))
    logging.info("Columns      : %d",len(out.columns))
    logging.info("Numerical X  : %d",len(numeric))
    logging.info("Relaxed used : NO")
    logging.info("PIGNet used  : NO")
    logging.info("Graph used   : NO")
    logging.info("Output       : %s",DESCRIPTORS)
    logging.info("="*90)

if __name__ == "__main__":
    main()
