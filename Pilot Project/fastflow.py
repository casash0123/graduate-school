# fastflow.py
#  - PS 결과 이미지를 512x512 타일로 분할 (MVTec AD 구조)
#  - anomalib FastFlow 학습/평가 (--run)
#  - test 결과(score/heatmap) 타일을 원본 좌표로 stitch 후
#    원본(타일 재조립) 위에 overlay + 컨투어 라벨 저장
#
#  * 패딩(예: 4000 -> 4032)으로 생기는 테두리 영역은 0(검정) 처리하고
#    파일 저장 시 원본 크기로 크롭하여 경계 아티팩트를 제거합니다.

import os, math, glob as pyglob, yaml, shutil, random, argparse, subprocess, json
from pathlib import Path
import cv2, numpy as np
import torch

# 4090 최적화(가능 시)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =====================================
# 공통 유틸
# =====================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(d: Path, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    if not d.exists(): return []
    out = []
    for e in exts:
        out.extend(pyglob.glob(str(d / f"*{e}")))
    return [Path(x) for x in sorted(out)]

def pad_to_multiple(img, tile, overlap=0):
    h, w = img.shape[:2]
    stride = tile - overlap if 0 <= overlap < tile else tile

    def padded_len(sz):
        if sz <= tile:
            return tile
        return ((max(sz - tile, 0) + stride - 1) // stride) * stride + tile

    nh = padded_len(h); nw = padded_len(w)
    if nh == h and nw == w:
        return img
    # 🔧 상수(0) 패딩으로 변경
    return cv2.copyMakeBorder(
        img, 0, nh - h, 0, nw - w,
        cv2.BORDER_CONSTANT, value=0
    )

def tiles(img, tile, overlap=0):
    """overlap 적용 타일 생성. 마지막 시작점(size - tile) 강제 포함."""
    h, w = img.shape[:2]
    stride = tile - overlap if 0 <= overlap < tile else tile

    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    # 마지막 시작점 보장
    last_y = max(h - tile, 0)
    last_x = max(w - tile, 0)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)
    if not xs or xs[-1] != last_x:
        xs.append(last_x)

    for y in ys:
        for x in xs:
            yield (y, x), img[y:y + tile, x:x + tile]

# =====================================
# 1) PS → MVTec 변환
# =====================================
def build_mvtec(raw: Path, outd: Path, tile: int = 512, train_ratio: float = 0.8, overlap: int = 0):
    good = list_images(raw / "good")
    defect = list_images(raw / "defect")
    gt_dir = raw / "gt"

    if not good:
        raise SystemExit(f"[ERROR] 양품 이미지가 없습니다: {raw/'good'}")

    # 새로 생성
    if outd.exists():
        shutil.rmtree(outd)

    random.seed(2025)
    random.shuffle(good)
    n_tr = int(len(good) * train_ratio)
    trainG, testG = good[:n_tr], good[n_tr:]

    # 원본 사이즈 기록용
    sizes = {}  # base(stem) -> (h0, w0)

    # train/test good 타일 생성
    for split, paths, dst in [
        ("train", trainG, outd / "train" / "good"),
        ("test",  testG,  outd / "test"  / "good"),
    ]:
        ensure_dir(dst)
        print(f"[GOOD-{split}] {len(paths)}장 처리 중…")
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  !! 읽기 실패: {p}")
                continue
            h0, w0 = img.shape[:2]          # 패딩 전 원본 크기
            sizes[p.stem] = (h0, w0)
            img = pad_to_multiple(img, tile, overlap)
            for (y, x), t in tiles(img, tile, overlap):
                name = f"{p.stem}_y{y}_x{x}.png"
                cv2.imwrite(str(dst / name), t)

    # defect 타일 생성 (+ GT 정렬)
    if defect:
        dstD = outd / "test" / "defect"; ensure_dir(dstD)
        dstM = outd / "ground_truth" / "defect"; ensure_dir(dstM)
        print(f"[DEFECT-test] {len(defect)}장 처리 중…")
        for p in defect:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  !! 읽기 실패: {p}")
                continue
            h0, w0 = img.shape[:2]          # 패딩 전 원본 크기
            sizes[p.stem] = (h0, w0)
            img = pad_to_multiple(img, tile, overlap)

            mask = None
            if (gt_dir).exists():
                mp = gt_dir / f"{p.stem}.png"
                if mp.exists():
                    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = pad_to_multiple(mask, tile, overlap)

            for (y, x), t in tiles(img, tile, overlap):
                name = f"{p.stem}_y{y}_x{x}.png"
                cv2.imwrite(str(dstD / name), t)
                if mask is not None:
                    cv2.imwrite(str(dstM / name), mask[y:y + tile, x:x + tile])
                else:
                    # GT가 없어도 anomalib이 기대하는 파일 존재하도록 제로마스크 생성
                    zero = np.zeros((tile, tile), dtype=np.uint8)
                    cv2.imwrite(str(dstM / name), zero)

    # 사이즈 메타 저장
    ensure_dir(outd / "meta")
    with open(outd / "meta" / "meta_sizes.json", "w", encoding="utf-8") as f:
        json.dump(sizes, f)

def fix_gt_alignment(outd: Path, tile: int = 512):
    """test/defect 타일과 ground_truth/defect 타일을 1:1로 강제 정렬"""
    d_def = outd / "test" / "defect"
    d_gt  = outd / "ground_truth" / "defect"
    ensure_dir(d_gt)

    def_files = {p.name for p in list_images(d_def)}
    gt_files  = {p.name for p in list_images(d_gt)}

    # 1) 누락된 GT 마스크는 제로마스크로 생성
    missing = def_files - gt_files
    if missing:
        print(f"[FIX] GT 누락 {len(missing)}개 → 제로마스크 생성")
        zero = np.zeros((tile, tile), dtype=np.uint8)
        for name in missing:
            cv2.imwrite(str(d_gt / name), zero)

    # 2) 고아(고립) 마스크는 삭제
    orphan = gt_files - def_files
    if orphan:
        print(f"[FIX] 고아 마스크 {len(orphan)}개 삭제")
        for name in orphan:
            try:
                (d_gt / name).unlink()
            except Exception as e:
                print(f"  !! 삭제 실패: {(d_gt/name)} -> {e}")

    # 3) 최종 검증
    def_files2 = {p.name for p in list_images(d_def)}
    gt_files2  = {p.name for p in list_images(d_gt)}
    if def_files2 != gt_files2:
        diff1 = len(def_files2 - gt_files2)
        diff2 = len(gt_files2 - def_files2)
        raise SystemExit(f"[ERROR] 보정 실패: defect↔gt 불일치 (def-miss={diff1}, gt-orphan={diff2})")
    else:
        print(f"[OK] defect({len(def_files2)}) ↔ gt({len(gt_files2)}) 1:1 정렬 완료")

# =====================================
# FastFlow 예측 → score/heatmap 저장
# =====================================
def export_test_scores(trainer, model, dataset, save_dir: Path):
    """FastFlow 예측 결과를 타일 단위 raw score(0~1)로 저장(.npy와 8bit PNG)"""
    ensure_dir(save_dir)
    model.eval()
    preds = trainer.predict(model=model, datamodule=dataset)

    def to_numpy(t):
        import torch
        if isinstance(t, torch.Tensor):
            return t.detach().float().cpu().numpy()
        return t

    def pick(obj, *candidates):
        for k in candidates:
            if isinstance(obj, dict) and k in obj: return obj[k]
            if hasattr(obj, k): return getattr(obj, k)
        return None

    for batch_out in preds:
        items = [batch_out] if isinstance(batch_out, dict) else (
            list(batch_out) if isinstance(batch_out, (list, tuple)) else [batch_out]
        )
        for out in items:
            paths = pick(out, "image_paths", "image_path", "paths", "path")
            if paths is None:
                inputs = pick(out, "inputs", "input")
                paths = pick(inputs, "image_paths", "image_path", "paths", "path") if inputs is not None else None
                if paths is None:
                    continue
            if isinstance(paths, (str, Path)):
                paths = [paths]

            m = pick(out, "anomaly_maps", "anomaly_map", "pred_masks", "prediction", "preds")
            if m is None:
                outputs = pick(out, "outputs", "output")
                m = pick(outputs, "anomaly_maps", "anomaly_map", "pred_masks") if outputs is not None else None
                if m is None:
                    continue

            m = to_numpy(m)  # [N,C,H,W] or [N,H,W] or [H,W]
            if m.ndim == 2:
                m = m[None, ...]
            if m.ndim == 4:
                m = m[:, 0, :, :]

            for i, p in enumerate(paths):
                if i >= m.shape[0]:
                    break
                tile_score = m[i].astype(np.float32)  # 🔒 원시 스코어 유지(정규화 X)
                stem = Path(p).stem
                np.save(str(save_dir / f"{stem}.npy"), tile_score)

                # 시각화용 PNG만 클리핑
                png = (np.clip(tile_score, 0.0, 1.0) * 255).astype(np.uint8)
                cv2.imwrite(str(save_dir / f"{stem}.png"), png)

def export_test_heatmaps(trainer, model, dataset, save_dir: Path):
    """FastFlow 예측 결과를 타일 단위 heatmap 이미지로 저장"""
    ensure_dir(save_dir)
    model.eval()
    preds = trainer.predict(model=model, datamodule=dataset)  # 배치 리스트

    def to_numpy(t):
        import torch
        if isinstance(t, torch.Tensor):
            return t.detach().float().cpu().numpy()
        return t

    def pick(obj, *candidates):
        for k in candidates:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
            if hasattr(obj, k):
                return getattr(obj, k)
        return None

    for batch_out in preds:
        items = [batch_out] if isinstance(batch_out, dict) else (list(batch_out) if isinstance(batch_out, (list, tuple)) else [batch_out])
        for out in items:
            # 경로
            paths = pick(out, "image_paths", "image_path", "paths", "path")
            if paths is None:
                inputs = pick(out, "inputs", "input")
                paths = pick(inputs, "image_paths", "image_path", "paths", "path") if inputs is not None else None
                if paths is None:
                    continue
            if isinstance(paths, (str, Path)):
                paths = [paths]

            # heatmap 후보
            hm = pick(out, "anomaly_maps", "anomaly_map", "heatmap", "pred_masks", "prediction", "preds")
            if hm is None:
                outputs = pick(out, "outputs", "output")
                hm = pick(outputs, "anomaly_maps", "anomaly_map", "heatmap", "pred_masks") if outputs is not None else None
                if hm is None:
                    continue

            hm_np = to_numpy(hm)  # [N,1,H,W] 또는 [N,H,W] 또는 [H,W]
            if hm_np.ndim == 2:
                hm_np = hm_np[None, ...]
            elif hm_np.ndim == 4:
                hm_np = hm_np[:, 0, :, :]

            for i, p in enumerate(paths):
                if i >= hm_np.shape[0]:
                    break
                stem = Path(p).stem
                h = hm_np[i]
                hmin, hmax = float(h.min()), float(h.max())
                rng = max(hmax - hmin, 1e-6)
                h = (h - hmin) / rng
                h_u8 = (np.clip(h, 0.0, 1.0) * 255).astype(np.uint8)
                h_color = cv2.applyColorMap(h_u8, cv2.COLORMAP_JET)
                cv2.imwrite(str(save_dir / f"{stem}.png"), h_color)

# =====================================
# 2) FastFlow 학습 및 평가 (anomalib 2.2.0)
# =====================================
from pathlib import Path
from anomalib.models.image.fastflow import Fastflow
from anomalib.data import MVTecAD as MVTec
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

# 1) 학습하고 ckpt 남기는 쪽
def run_fastflow_train(outd: Path,
                       image_size=512,
                       backbone="wide_resnet50_2",
                       flow_steps=8) -> Path:
    category = "metal_case"
    logger = TensorBoardLogger(save_dir="runs", name="fastflow_metal_case")

    datamodule = MVTec(
        root=str(outd.parent if outd.name == category else outd),
        category=category,
        train_batch_size=32,
        eval_batch_size=96,   # 4090이면 이 정도
        num_workers=8,
    )

    model = Fastflow(backbone=backbone, flow_steps=flow_steps)

    ckpt_dir = Path("runs") / "fastflow_metal_case" / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=50,
        logger=logger,
        default_root_dir=str(ckpt_dir),
    )

    trainer.fit(model=model, datamodule=datamodule)

    ckpt_path = ckpt_dir / "fastflow_metal_case_last.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    print(f"[INFO] model saved → {ckpt_path}")
    return ckpt_path


# 2) 저장해 둔 걸로 검사만 하는 쪽
def run_fastflow_test(outd: Path,
                      ckpt_path: Path,
                      image_size=512):
    from fastflow import export_test_scores  # 네가 위에 만든 함수 그대로
    category = "metal_case"

    datamodule = MVTec(
        root=str(outd.parent if outd.name == category else outd),
        category=category,
        eval_batch_size=96,
        num_workers=8,
    )

    model = Fastflow.load_from_checkpoint(str(ckpt_path))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
    )

    trainer.test(model=model, datamodule=datamodule)

    manual_dir = Path("runs") / "fastflow_metal_case" / "manual_scores"
    export_test_scores(trainer, model, datamodule, manual_dir)
    print(f"[INFO] saved scores to: {manual_dir.resolve()}")
    return manual_dir


# =====================================
# 3) heatmap 스티치 + 원본 overlay (패딩 0 처리 + 크롭)
# =====================================
def merge_fastflow_results(outd: Path, result_dir: Path, tile=512, alpha=0.5):
    """
    저장된 heatmap/score 타일을 원본 크기로 stitch 후, 원본 타일을 다시 붙여 overlay 저장
    패딩으로 추가된 영역(예: 4032-4000)은 0으로 마스킹하고 저장 시 크롭.
    """
    heatmap_dir = Path(result_dir)
    if not heatmap_dir.exists():
        raise SystemExit(f"[ERROR] FastFlow 결과 폴더를 찾을 수 없습니다: {heatmap_dir}")

    # score(.npy) 또는 8bit png 로드
    score_files = list(heatmap_dir.glob("*.npy")) + [p for p in list_images(heatmap_dir) if p.suffix.lower() == ".png"]
    if not score_files:
        raise SystemExit(f"[ERROR] heatmap/score 파일이 없습니다: {heatmap_dir}")

    # 원본 사이즈 로드
    sizes = {}
    meta_path = outd / "meta" / "meta_sizes.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            sizes = json.load(f)

    merged = {}
    for f in score_files:
        stem = f.stem  # ex) foo_y512_x1024
        if "_y" not in stem or "_x" not in stem:
            continue
        base = stem.split("_y")[0]
        y = int(stem.split("_y")[1].split("_x")[0])
        x = int(stem.split("_x")[1])

        # 스코어 불러오기
        if f.suffix.lower() == ".npy":
            sc = np.load(str(f)).astype(np.float32)  # 0~1 가정
            sc_u8 = (np.clip(sc, 0, 1) * 255).astype(np.uint8)
            hm = cv2.applyColorMap(sc_u8, cv2.COLORMAP_JET)
        else:
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            hm = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        if hm.shape[0] != tile or hm.shape[1] != tile:
            hm = cv2.resize(hm, (tile, tile), interpolation=cv2.INTER_LINEAR)

        merged.setdefault(base, []).append((y, x, hm))

    save_dir = outd / "merged_overlay"
    ensure_dir(save_dir)

    # 코사인 윈도우(경계 페더링)
    win1d = np.hanning(tile).astype(np.float32)
    w2d = np.outer(win1d, win1d).astype(np.float32)[:, :, None]
    w2d /= (w2d.max() + 1e-6)

    for name, patches in merged.items():
        H = max(y for y, _, _ in patches) + tile
        W = max(x for _, x, _ in patches) + tile

        # heatmap 스티치
        acc = np.zeros((H, W, 3), np.float32)
        wsum = np.zeros((H, W, 1), np.float32)
        for y, x, hm in patches:
            tile_img = hm
            if tile_img.shape[:2] != (tile, tile):
                tile_img = cv2.resize(tile_img, (tile, tile), interpolation=cv2.INTER_LINEAR)
            acc[y:y + tile, x:x + tile, :] += tile_img.astype(np.float32) * w2d
            wsum[y:y + tile, x:x + tile, :] += w2d
        wsum[wsum == 0] = 1.0
        canvas = (acc / wsum).astype(np.uint8)

        # 원본 타일 재조립
        base_canvas = np.zeros_like(canvas)
        found_any = False
        for src_dir in ["test/defect", "test/good"]:
            tile_paths = list((outd / src_dir).glob(f"{name}_y*_x*.png"))
            if not tile_paths:
                continue
            for p in tile_paths:
                pstem = p.stem
                try:
                    yy = int(pstem.split("_y")[1].split("_x")[0])
                    xx = int(pstem.split("_x")[1])
                except Exception:
                    continue
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                if im.shape[0] != tile or im.shape[1] != tile:
                    im = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_LINEAR)
                base_canvas[yy:yy+tile, xx:xx+tile] = im
            found_any = True
            break

        if not found_any:
            print(f"[WARN] 원본 타일을 찾지 못했습니다: {name} → heatmap만 저장합니다.")
            out_path_hm = save_dir / f"{name}_heatmap.png"
            cv2.imwrite(str(out_path_hm), canvas)
            continue

        # 패딩(원본 외곽) 0 처리 + 저장 시 크롭
        h0, w0 = sizes.get(name, (None, None))
        if h0 is not None and w0 is not None:
            canvas[h0:, :, :] = 0
            canvas[:, w0:, :] = 0
            base_canvas[h0:, :, :] = 0
            base_canvas[:, w0:, :] = 0

        blend = cv2.addWeighted(base_canvas, 1 - alpha, canvas, alpha, 0)
        if h0 is not None and w0 is not None:
            blend = blend[:h0, :w0]

        out_path = save_dir / f"{name}_overlay.png"
        cv2.imwrite(str(out_path), blend)
        print(f"[MERGED] {out_path}")

# =====================================
# 3b) Score 스티치 → 이진화 → 컨투어(정보표시) (패딩 0 처리 + 크롭)
# =====================================
def merge_fastflow_contours(outd: Path, result_dir: Path, tile=512,
                            thresh='percentile', pct=99.2,    # 또는 'otsu'
                            min_area=1500, draw_thickness=3, overlap=0):
    """
    저장된 score 타일(.npy 또는 8bit png)을 코사인 페더링으로 스티치 → 이진화 → 컨투어 →
    원본에 외곽선/센터점/라벨(센터, 면적, 둘레, 박스) 그려 저장
    패딩으로 추가된 영역은 0 처리 후, 저장 시 원본 크기로 크롭.
    """
    def _cosine_w(tile_size):
        y = np.hanning(tile_size); x = np.hanning(tile_size)
        w = np.outer(y, x).astype(np.float32)
        w /= (w.max() + 1e-6)
        return w

    d = Path(result_dir)
    if not d.exists():
        raise SystemExit(f"[ERROR] 점수 폴더 없음: {d}")

    # base별로 점수 타일 묶기 (.npy 우선, 없으면 png)
    files = sorted(d.glob("*.npy"))
    if not files:
        files = [p for p in list_images(d) if p.suffix.lower()==".png"]
    if not files:
        raise SystemExit(f"[ERROR] score 파일이 없습니다 (.npy/.png): {d}")

    # 사이즈 로드
    sizes = {}
    meta_path = outd / "meta" / "meta_sizes.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            sizes = json.load(f)

    buckets = {}
    for f in files:
        s = f.stem
        if "_y" not in s or "_x" not in s:
            continue
        base = s.split("_y")[0]
        y = int(s.split("_y")[1].split("_x")[0])
        x = int(s.split("_x")[1])
        buckets.setdefault(base, []).append((y, x, f))

    save_dir = outd / "merged_overlay"
    ensure_dir(save_dir)
    w2d = _cosine_w(tile)  # (tile, tile)

    for name, items in buckets.items():
        H = max(y for y,_,_ in items) + tile
        W = max(x for _,x,_ in items) + tile

        # 1) 스코어 스티치 (경계 페더링)
        acc  = np.zeros((H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        for y, x, f in items:
            if f.suffix.lower()==".npy":
                sc = np.load(str(f)).astype(np.float32)          # 0~1 가정
            else:
                g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                sc = (g.astype(np.float32)/255.0) if g is not None else None
            if sc is None:
                continue
            if sc.shape[:2] != (tile, tile):
                sc = cv2.resize(sc, (tile, tile), interpolation=cv2.INTER_LINEAR)
            acc[y:y+tile, x:x+tile]  += sc * w2d
            wsum[y:y+tile, x:x+tile] += w2d
        wsum[wsum==0] = 1.0
        score = np.clip(acc/wsum, 0.0, 1.0)

        # 1-a) 패딩 영역 0 처리
        h0, w0 = sizes.get(name, (None, None))
        if h0 is not None and w0 is not None:
            score[h0:, :] = 0.0
            score[:, w0:] = 0.0

        # 2) 원본 타일 재조립
        base = np.zeros((H, W, 3), np.uint8)
        found = False
        for src in ["test/defect", "test/good"]:
            tps = list((outd/src).glob(f"{name}_y*_x*.png"))
            if not tps:
                continue
            for p in tps:
                st = p.stem
                yy = int(st.split("_y")[1].split("_x")[0])
                xx = int(st.split("_x")[1])
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                if im.shape[:2] != (tile, tile):
                    im = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_LINEAR)
                base[yy:yy+tile, xx:xx+tile] = im
            found = True
            break
        if not found:
            print(f"[WARN] 원본 타일을 찾지 못함: {name} → score만 저장")
            out_raw = (score*255).astype(np.uint8)
            if h0 is not None and w0 is not None:
                out_raw = out_raw[:h0, :w0]
            cv2.imwrite(str(save_dir/f"{name}_score.png"), out_raw)
            continue

        # 2-a) 원본 패딩 0 처리
        if h0 is not None and w0 is not None:
            base[h0:, :, :] = 0
            base[:, w0:, :] = 0

        # 3) 후처리 + 이진화
        sc8 = (score * 255).astype(np.uint8)
        sc8 = cv2.GaussianBlur(sc8, (5, 5), 0)
        if thresh == "otsu":
            _, binm = cv2.threshold(sc8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            valid = sc8[sc8 > 0]
            base_arr = sc8 if valid.size < 1000 else valid  # 유효 픽셀 충분할 때만 사용
            t = int(np.percentile(base_arr, pct))
            _, binm = cv2.threshold(sc8, t, 255, cv2.THRESH_BINARY)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k, iterations=1)
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k, iterations=2)

        # 4) 컨투어 → 외곽선/정보 표시
        out = base.copy()
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            perim = cv2.arcLength(c, True)
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

            cv2.drawContours(out, [c], -1, (0,0,255), draw_thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (cx,cy), 4, (0,0,255), -1, lineType=cv2.LINE_AA)
            label = f"Center:({cx},{cy})  Area:{int(area)}  Perim:{int(perim)}  Box:{w}x{h}"
            tx, ty = max(10, x), max(20, y-10)
            cv2.putText(out, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

        # 5) 저장 전에 크롭
        if h0 is not None and w0 is not None:
            out = out[:h0, :w0]
        cv2.imwrite(str(save_dir/f"{name}_overlay.png"), out)
        print(f"[MERGED] {save_dir/f'{name}_overlay.png'}")

# =====================================
# 4) 메인
# =====================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--flow-steps", type=int, default=8)
    ap.add_argument("--merge", action="store_true", help="추론 후 이상맵 병합 및 overlay 수행")
    ap.add_argument("--overlap", type=int, default=0, help="타일 오버랩(픽셀)")
    ap.add_argument("--thresh", default="percentile", choices=["percentile", "otsu"])
    ap.add_argument("--pct", type=float, default=99.2)  # percentile 사용 시
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--draw-thickness", type=int, default=3)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    outd = Path(args.out_dir)
    category = "metal_case"

    # out_dir 하위에 category 폴더 맞춰 생성
    outd_cat = outd if outd.name == category else outd / category

    # 1) 데이터셋 생성 + GT 정렬 (이 부분은 그대로 둠)
    build_mvtec(raw, outd_cat, tile=args.tile,
                train_ratio=args.train_ratio,
                overlap=args.overlap)
    fix_gt_alignment(outd_cat, tile=args.tile)

    heatmap_dir = None

    # 2-A) 학습 모드
    if args.run:
        ckpt_path = run_fastflow_train(
            outd_cat,
            image_size=args.tile,
            backbone=args.backbone,
            flow_steps=args.flow_steps,
        )
        print(f"[INFO] ckpt saved: {ckpt_path}")

    # 2-B) 추론 모드
    if args.test:
        if not args.ckpt:
            raise SystemExit("--test 할 때는 --ckpt <파일> 필요합니다.")
        heatmap_dir = run_fastflow_test(
            outd_cat,
            Path(args.ckpt),
            image_size=args.tile,
        )

    # 3) 병합
    if args.merge:
        result_dir = Path(heatmap_dir) if heatmap_dir else Path("runs/fastflow_metal_case/manual_scores")
        if not result_dir.exists():
            raise SystemExit(f"[ERROR] heatmap/score 디렉터리를 찾지 못했습니다: {result_dir}")

        merge_fastflow_contours(
            outd_cat, result_dir,
            tile=args.tile,
            thresh=args.thresh,
            pct=args.pct,
            min_area=args.min_area,
            draw_thickness=args.draw_thickness,
            overlap=args.overlap,
        )

if __name__ == "__main__":
    main()
