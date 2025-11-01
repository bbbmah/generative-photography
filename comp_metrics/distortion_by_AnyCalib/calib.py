import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageSequence
from anycalib import AnyCalib

# =========================
# 설정
# =========================
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 사용할 모델 목록과 cam_id 매핑 (원본 코드 유지)
MODEL_PLAN = [
    ("anycalib_pinhole", "pinhole"),
    ("anycalib_gen", "simple_radial:4"),
    ("anycalib_dist", "simple_radial:4"),
    ("anycalib_edit", "simple_radial:4"),
]

out_root = Path("./anycalib_gif_results")
out_root.mkdir(parents=True, exist_ok=True)


# =========================
# 유틸
# =========================
def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """
    PIL.Image (RGB) -> (3,H,W) float32 in [0,1] Tensor
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.to(device=device, dtype=torch.float32)


def extract_frames(gif_path: str) -> List[Image.Image]:
    """
    GIF에서 모든 프레임을 추출하여 RGB PIL 이미지 리스트로 반환
    """
    gif = Image.open(gif_path)
    frames = [frame.convert('RGB') for frame in ImageSequence.Iterator(gif)]
    return frames


def save_intrinsics_series_as_json_csv(
    series: List[List[float]],
    out_dir: Path,
    stem: str,
    frame_names: Optional[List[str]] = None,
) -> None:
    """
    프레임별 intrinsics 시퀀스를 JSON/CSV로 저장
    - series: List[frame_index -> List[param]]
    - frame_names: 각 프레임의 식별자(예: f"{i:04d}")
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_items = []
    for i, params in enumerate(series):
        name = frame_names[i] if frame_names else f"frame_{i:04d}"
        json_items.append({"frame": name, "intrinsics": params})

    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_items, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] JSON -> {json_path}")

    # CSV
    max_len = max((len(p) for p in series), default=0)
    fieldnames = ["frame"] + [f"param_{i}" for i in range(max_len)]
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, params in enumerate(series):
            name = frame_names[i] if frame_names else f"frame_{i:04d}"
            row = {"frame": name}
            for j, v in enumerate(params):
                row[f"param_{j}"] = v
            writer.writerow(row)
    print(f"[SAVE] CSV  -> {csv_path}")


# =========================
# 2-1. 한 이미지의 왜곡 계수 계산
# =========================
@torch.inference_mode()
def estimate_intrinsics_on_image(
    model: AnyCalib,
    cam_id: str,
    image: Image.Image,
) -> List[float]:
    """
    요구사항 2-1. 한 이미지의 왜곡 계수(= intrinsics) 계산
    - 입력: PIL.Image (RGB)
    - 출력: List[float] (모델이 반환하는 intrinsics 벡터)
    """
    img_t = pil_to_tensor(image, dev)  # (3,H,W), float32 [0,1]
    output = model.predict(img_t, cam_id=cam_id)

    intr = output.get("intrinsics", None)
    if isinstance(intr, torch.Tensor):
        intr = [float(x) for x in intr.detach().cpu().flatten()]
    elif intr is None:
        intr = []
    return intr


# =========================
# 2-2. 한 GIF의 왜곡 계수 리스트 계산
# =========================
def intrinsics_list_from_gif(
    gif_path: str,
    model_id: str = "anycalib_gen",
    cam_id: str = "simple_radial:4",
    save_dir: Optional[Path] = None,
) -> List[List[float]]:
    """
    요구사항 2-2. GIF → 프레임별 intrinsics 리스트
    - model_id/cam_id로 AnyCalib을 내부에서 로드하여 사용
    - 필요 시 JSON/CSV 저장
    """
    frames = extract_frames(gif_path)
    model = AnyCalib(model_id=model_id).to(dev).eval()

    series = []
    for i, frame in enumerate(frames):
        intr = estimate_intrinsics_on_image(model, cam_id, frame)
        print(f"[{Path(gif_path).name}] frame {i:04d}: intrinsics_dim={len(intr)}")
        series.append(intr)

    if save_dir is not None:
        stem = f"{Path(gif_path).stem}__{model_id.replace('/', '_')}"
        frame_names = [f"{i:04d}" for i in range(len(series))]
        save_intrinsics_series_as_json_csv(series, save_dir, stem, frame_names)

    return series


# =========================
# 2-3. 두 GIF의 왜곡 계수 리스트 비교
# =========================
def _l2_distance_same_len(a: List[float], b: List[float]) -> float:
    """두 벡터의 공통 길이까지만 L2 거리 계산"""
    m = min(len(a), len(b))
    if m == 0:
        return float("nan")
    va = np.array(a[:m], dtype=np.float64)
    vb = np.array(b[:m], dtype=np.float64)
    return float(np.linalg.norm(va - vb, ord=2))


def _componentwise_pearson(
    seq1: List[List[float]],
    seq2: List[List[float]],
) -> Tuple[List[float], List[int]]:
    """
    파라미터 차원별(공통 차원까지만) 프레임 축 방향 피어슨 상관계수 계산
    - 반환: (corr_list, valid_counts) 각 차원별 유효 프레임 수
    """
    n_frames = min(len(seq1), len(seq2))
    if n_frames == 0:
        return [], []

    dim = min(
        max(len(v) for v in seq1 if v),
        max(len(v) for v in seq2 if v),
    )

    corr_list = []
    valid_counts = []
    for d in range(dim):
        a = []
        b = []
        for t in range(n_frames):
            if len(seq1[t]) > d and len(seq2[t]) > d:
                a.append(seq1[t][d])
                b.append(seq2[t][d])

        if len(a) >= 2 and len(b) >= 2:
            c = float(np.corrcoef(a, b)[0, 1])
            corr_list.append(c)
            valid_counts.append(len(a))
        else:
            corr_list.append(float("nan"))
            valid_counts.append(len(a))
    return corr_list, valid_counts


def _l2_norm_series(seq: List[List[float]]) -> List[float]:
    """각 프레임 벡터의 공통 길이 기준 L2 norm 시퀀스(시퀀스 간 비교용)"""
    dim = max((len(v) for v in seq if v), default=0)
    # 다른 리스트와 맞출 때는 외부에서 min-dim을 적용함
    norms = []
    for v in seq:
        m = min(len(v), dim)
        if m == 0:
            norms.append(float("nan"))
        else:
            norms.append(float(np.linalg.norm(np.array(v[:m], dtype=np.float64), ord=2)))
    return norms


def compare_intrinsics_lists(
    seq1: List[List[float]],
    seq2: List[List[float]],
    *,
    per_frame_metric: str = "l2",  # "l2" 또는 "l1" 확장 가능
) -> Dict:
    """
    요구사항 2-3. 두 GIF의 intrinsics 리스트 비교
    결과:
      - frame_errors: 프레임 인덱스별 오차(기본 L2). 길이는 min(len(seq1), len(seq2))
      - component_corrs: 파라미터 차원별 피어슨 r (공통 차원까지만)
      - l2norm_trend_corr: 프레임별 L2-norm 시퀀스 간 상관계수(추가 요약 지표)
    """
    n = min(len(seq1), len(seq2))
    frame_errors = []
    for i in range(n):
        if per_frame_metric == "l2":
            e = _l2_distance_same_len(seq1[i], seq2[i])
        else:  # 간단 확장: L1
            m = min(len(seq1[i]), len(seq2[i]))
            if m == 0:
                e = float("nan")
            else:
                va = np.array(seq1[i][:m], dtype=np.float64)
                vb = np.array(seq2[i][:m], dtype=np.float64)
                e = float(np.linalg.norm(va - vb, ord=1))
        frame_errors.append(e)

    component_corrs, valid_counts = _componentwise_pearson(seq1, seq2)

    # 요약: L2-norm 시퀀스의 상관계수(프레임별 크기 경향 비교)
    # 공통 파라미터 차원을 보장하기 위해 다시 한 번 min-dim을 적용
    dim1 = max((len(v) for v in seq1 if v), default=0)
    dim2 = max((len(v) for v in seq2 if v), default=0)
    common_dim = min(dim1, dim2)

    def l2_series_with_dim(seq, d):
        out = []
        for v in seq[:n]:
            m = min(len(v), d)
            if m == 0:
                out.append(float("nan"))
            else:
                out.append(float(np.linalg.norm(np.array(v[:m], dtype=np.float64), ord=2)))
        return out

    l2_1 = l2_series_with_dim(seq1, common_dim)
    l2_2 = l2_series_with_dim(seq2, common_dim)
    # NaN 제거
    mask = [np.isfinite(a) and np.isfinite(b) for a, b in zip(l2_1, l2_2)]
    l2_1v = np.array([a for a, m in zip(l2_1, mask) if m], dtype=np.float64)
    l2_2v = np.array([b for b, m in zip(l2_2, mask) if m], dtype=np.float64)
    if len(l2_1v) >= 2 and len(l2_2v) >= 2:
        l2norm_trend_corr = float(np.corrcoef(l2_1v, l2_2v)[0, 1])
    else:
        l2norm_trend_corr = float("nan")

    return {
        "num_frames_compared": n,
        "frame_errors": frame_errors,           # 4-1 요구사항
        "component_corrs": component_corrs,     # 4-2 요구사항(차원별)
        "component_valid_counts": valid_counts, # 각 차원의 유효 프레임 수
        "l2norm_trend_corr": l2norm_trend_corr # 경향성 요약
    }


# =========================
# 예시 실행부
# =========================
if __name__ == "__main__":
    # 예시 입력
    gif_a = "/path/to/A_reference.gif"
    gif_b = "/path/to/A_sample.gif"

    # 1) 단일 모델로 실행 (원한다면 MODEL_PLAN에서 골라 사용)
    model_id = "anycalib_gen"
    cam_id = "simple_radial:4"

    # 2) 각 GIF에 대해 프레임별 intrinsics 시퀀스 추출 및 저장
    seq_a = intrinsics_list_from_gif(
        gif_a, model_id=model_id, cam_id=cam_id, save_dir=out_root
    )
    seq_b = intrinsics_list_from_gif(
        gif_b, model_id=model_id, cam_id=cam_id, save_dir=out_root
    )

    # 3) 두 시퀀스 비교
    compare = compare_intrinsics_lists(seq_a, seq_b, per_frame_metric="l2")

    # 4) 결과 저장
    comp_path = out_root / f"compare__{Path(gif_a).stem}__vs__{Path(gif_b).stem}__{model_id.replace('/','_')}.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] COMPARE -> {comp_path}")

    # 5) 필요시 4개 모델 전부 반복 실행 (옵션)
    # for mid, cid in MODEL_PLAN:
    #     sa = intrinsics_list_from_gif(gif_a, model_id=mid, cam_id=cid, save_dir=out_root)
    #     sb = intrinsics_list_from_gif(gif_b, model_id=mid, cam_id=cid, save_dir=out_root)
    #     comp = compare_intrinsics_lists(sa, sb)
    #     comp_path = out_root / f"compare__{Path(gif_a).stem}__vs__{Path(gif_b).stem}__{mid.replace('/','_')}.json"
    #     with open(comp_path, "w", encoding="utf-8") as f:
    #         json.dump(comp, f, ensure_ascii=False, indent=2)
    #     print(f"[SAVE] COMPARE -> {comp_path}")
