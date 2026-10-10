"""校验 mair_benchmark_v1 数据集：无泄漏、无重复、文件齐全、尺寸正确、哈希一致。"""
import csv
import json
import hashlib
import sys
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
FIELDS = ["sample_id", "split", "hq_id", "hq_path", "lq_path", "hq_sha256",
          "lq_sha256", "degradation_sequence", "degradation_parameters",
          "random_seed", "width", "height"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    errors = []
    val = read_manifest(BASE / "validation_manifest.csv")
    test = read_manifest(BASE / "test_manifest.csv")
    all_rows = val + test
    print(f"validation 行数: {len(val)}, test 行数: {len(test)}")

    # 1) HQ 无交集
    val_hq = {r["hq_id"] for r in val}
    test_hq = {r["hq_id"] for r in test}
    if val_hq & test_hq:
        errors.append(f"HQ 跨 split 泄漏: {val_hq & test_hq}")
    else:
        print(f"OK: validation/test HQ 无交集（val {len(val_hq)} / test {len(test_hq)}）")

    # 2) 文件无重复
    lq_paths = [r["lq_path"] for r in all_rows]
    if len(lq_paths) != len(set(lq_paths)):
        errors.append("LQ 路径存在重复")
    else:
        print("OK: LQ 路径无重复")
    lq_hashes = [r["lq_sha256"] for r in all_rows]
    if len(lq_hashes) != len(set(lq_hashes)):
        errors.append("LQ 内容哈希存在重复（可能同图）")
    else:
        print("OK: LQ 内容哈希无重复")

    # 3) 同一 HQ 不跨 split（hq_id 集合已覆盖，再核对 hq_sha256）
    val_hq_hash = {r["hq_sha256"] for r in val}
    test_hq_hash = {r["hq_sha256"] for r in test}
    if val_hq_hash & test_hq_hash:
        errors.append("HQ 内容哈希跨 split 泄漏")
    else:
        print("OK: HQ 内容哈希无跨 split 交集")

    # 4) 文件存在且可读 + 尺寸符合退化配置 + 哈希一致
    hq_shapes = {}
    missing = []
    size_mismatch = []
    hash_mismatch = []
    for r in all_rows:
        hq_p = BASE / r["hq_path"]
        lq_p = BASE / r["lq_path"]
        for p in (hq_p, lq_p):
            if not p.exists():
                missing.append(str(p))
        if missing:
            continue
        img = cv2.imread(str(lq_p))
        if img is None:
            missing.append(f"{lq_p}（不可读）")
            continue
        if r["hq_id"] not in hq_shapes:
            hq_img = cv2.imread(str(hq_p))
            hq_shapes[r["hq_id"]] = hq_img.shape[:2]
        hh, hw = hq_shapes[r["hq_id"]]
        h, w = img.shape[:2]
        seq = json.loads(r["degradation_sequence"])
        if "low resolution" in seq:
            exp_h, exp_w = round(hh * 0.25), round(hw * 0.25)
        else:
            exp_h, exp_w = hh, hw
        if abs(h - exp_h) > 2 or abs(w - exp_w) > 2:
            size_mismatch.append((r["sample_id"], (h, w), (exp_h, exp_w)))
        # 哈希
        if sha256(hq_p) != r["hq_sha256"]:
            hash_mismatch.append(r["hq_path"])
        if sha256(lq_p) != r["lq_sha256"]:
            hash_mismatch.append(r["lq_path"])

    if missing:
        errors.append(f"缺失/不可读文件 {len(missing)} 个: {missing[:5]}")
    else:
        print(f"OK: 全部 {len(all_rows)} 张 LQ + 关联 HQ 存在且可读")
    if size_mismatch:
        errors.append(f"尺寸不符 {len(size_mismatch)} 个: {size_mismatch[:5]}")
    else:
        print("OK: 尺寸符合退化配置（含 low-resolution 的为 HQ 的 0.25×）")
    if hash_mismatch:
        errors.append(f"哈希不一致 {len(hash_mismatch)} 个")
    else:
        print("OK: manifest 中的 sha256 与磁盘文件一致")

    # 5) hash_manifest.csv 覆盖所有文件且无重复
    hash_rows = list(csv.DictReader(open(BASE / "hash_manifest.csv")))
    hash_files = [r["file_path"] for r in hash_rows]
    if len(hash_files) != len(set(hash_files)):
        errors.append("hash_manifest 有重复文件路径")
    expected_files = {r["hq_path"] for r in all_rows} | {r["lq_path"] for r in all_rows}
    if set(hash_files) != expected_files:
        errors.append("hash_manifest 文件集合与 manifest 不一致")
    else:
        print(f"OK: hash_manifest 覆盖 {len(hash_files)} 个文件且无重复")

    print("\n===== 校验结果 =====")
    if errors:
        print("FAILED")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
