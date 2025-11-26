# prep_and_run_patchcore.py
#  - 5120x1180 같은 와이드 이미지를 512x512 타일로 분할
#  - MVTec AD 폴더 구조로 배치
#  - (옵션) anomalib PatchCore 학습/평가까지 실행

import os, math, glob, yaml, shutil, random, argparse, subprocess
from pathlib import Path
import cv2, numpy as np

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(d: Path, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    if not d.exists(): return []
    out = []
    for e in exts:
        out.extend(glob.glob(str(d / f"*{e}")))
    return [Path(x) for x in sorted(out)]

def pad_to_multiple(img, tile):
    """가로/세로를 tile(예: 512)의 배수로 패딩(가장자리 복제)"""
    h, w = img.shape[:2]
    nh = math.ceil(h / tile) * tile
    nw = math.ceil(w / tile) * tile
    if nh == h and nw == w:
        return img
    return cv2.copyMakeBorder(img, 0, nh - h, 0, nw - w, cv2.BORDER_REPLICATE)

def tiles(img, tile):
    """tile x tile 타일 생성 (패딩된 이미지를 입력)"""
    h, w = img.shape[:2]
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            yield (y, x), img[y:y+tile, x:x+tile]

def build_mvtec(raw: Path, outd: Path, tile: int = 512, train_ratio: float = 0.8):
    """
    입력(raw) 구조 예:
      raw/
        good/    (필수) 양품 원본 이미지 (3채널)
        defect/  (선택) 불량 원본 이미지
        gt/      (선택) 불량 픽셀 마스크(단일채널), 파일명 stem 동일
    출력(outd) 구조(MVTec AD):
      outd/
        train/good/*.png
        test/good/*.png
        test/defect/*.png
        ground_truth/defect/*.png
    """
    good = list_images(raw / "good")
    defect = list_images(raw / "defect")
    gt_dir = raw / "gt"

    if not good:
        raise SystemExit(f"[ERROR] 양품 이미지가 없습니다: {raw/'good'}")

    # 출력 폴더 새로 만들기
    if outd.exists():
        shutil.rmtree(outd)

    # good을 이미지 단위로 train/test 분할
    random.shuffle(good)
    n_tr = int(len(good) * train_ratio)
    trainG, testG = good[:n_tr], good[n_tr:]

    # train/good, test/good 타일링
    for split, paths, dst in [
        ("train", trainG, outd / "train" / "good"),
        ("test",  testG,  outd / "test"  / "good"),
    ]:
        ensure_dir(dst)
        print(f"[GOOD-{split}] {len(paths)}장 처리 중…")
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  !! 읽기 실패: {p}"); continue
            img = pad_to_multiple(img, tile)
            for (y, x), t in tiles(img, tile):
                name = f"{p.stem}_y{y}_x{x}.png"
                cv2.imwrite(str(dst / name), t)

    # defect 타일링 (있을 때만)
    if defect:
        dstD = outd / "test" / "defect"; ensure_dir(dstD)
        dstM = outd / "ground_truth" / "defect"; ensure_dir(dstM)
        print(f"[DEFECT-test] {len(defect)}장 처리 중…")
        for p in defect:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  !! 읽기 실패: {p}"); continue
            img = pad_to_multiple(img, tile)

            # 매칭 마스크(optional)
            mask = None
            mp = gt_dir / f"{p.stem}.png"
            if mp.exists():
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = pad_to_multiple(mask, tile)

            for (y, x), t in tiles(img, tile):
                name = f"{p.stem}_y{y}_x{x}.png"
                cv2.imwrite(str(dstD / name), t)
                if mask is not None:
                    cv2.imwrite(str(dstM / name), mask[y:y+tile, x:x+tile])

def write_cfg(cfg_path: Path, dataset_path: Path, image_size=512,
              backbone="wide_resnet50_2", coreset=0.01, use_faiss=True):
    """anomalib PatchCore 설정파일 생성"""
    cfg = {
        "model": {
            "name": "patchcore",
            "backbone": backbone,
            "layers": ["layer2", "layer3"],
            "image_size": image_size,
            "coreset_sampling_ratio": coreset,
        },
        "dataset": {
            "name": "metal_case",
            "format": "mvtec",
            "path": str(dataset_path),
            "category": "metal_case",
            "image_size": image_size,
            "task": "segmentation",
            "train_batch_size": 8,
            "eval_batch_size": 8
        },
        "trainer": {
            "max_epochs": 1,
            "accelerator": "gpu",
            "devices": 1,
            "log_every_n_steps": 5
        },
        "project": {"seed": 42},
        "logging": {"log_graph": False}
    }
    if use_faiss:
        cfg["model"]["nn_method"] = {"name": "faiss", "index": "ivfflat", "nlist": 64}
    ensure_dir(cfg_path.parent)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="원본 루트 (raw/good, raw/defect, raw/gt)")
    ap.add_argument("--out-dir", required=True, help="출력 데이터셋 루트 (MVTec AD 구조)")
    ap.add_argument("--tile", type=int, default=512, help="타일 크기")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="good의 train 비율")
    ap.add_argument("--run", action="store_true", help="전처리 후 PatchCore 학습/평가까지 실행")
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--coreset", type=float, default=0.01)
    ap.add_argument("--no-faiss", action="store_true", help="faiss 비활성화")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    outd = Path(args.out_dir)

    # 1) 전처리: 타일링 + MVTec 구조 생성
    build_mvtec(raw, outd, tile=args.tile, train_ratio=args.train_ratio)

    # 2) (선택) PatchCore 학습/평가까지 실행
    if args.run:
        cfg_path = Path("configs") / "metal_case_patchcore.yaml"
        write_cfg(cfg_path, outd, image_size=args.tile, backbone=args.backbone,
                  coreset=args.coreset, use_faiss=(not args.no_faiss))
        subprocess.run(["anomalib", "train", "--config", str(cfg_path)], check=True)
        subprocess.run(["anomalib", "test",  "--config", str(cfg_path)], check=True)
        print("\n[OK] PatchCore train/test 완료. 결과는 runs/ 또는 results/ 하위 폴더 확인")

if __name__ == "__main__":
    main()
