"""Strict parsers; preserve acquisition/mirror families when splitting."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from src.config import DATA, RESULTS, SHAPE, SEED


@dataclass
class Dataset:
    pressure: np.ndarray
    people: np.ndarray
    actions: np.ndarray
    frames: np.ndarray
    poses: np.ndarray
    groups: np.ndarray
    boxes: np.ndarray | None = None


def pose_for_action(action):
    if not 1 <= action <= 21:
        raise ValueError('动作必须在 1–21 之间')
    return 0 if action <= 6 else 1 if action <= 9 else 2 if action <= 15 else 3


def canonical_action(action):
    return action - 6 if action >= 16 else action


def parse_txt(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip().rstrip(',')
        if not line or line in {'0', '1', '2'}:
            continue
        values = np.fromstring(line, sep=',', dtype=np.float32)
        if len(values) != SHAPE[1] or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{path}:{line_no}: 非法压力行')
        rows.append(values)
    if not rows or len(rows) % SHAPE[0]:
        raise ValueError(f'{path}: 不完整的 44×24 帧')
    return np.asarray(rows).reshape(-1, *SHAPE)


def load_raw():
    cache = RESULTS / 'raw_data.npz'
    if cache.exists() and cache.stat().st_mtime > max(p.stat().st_mtime for p in (DATA/'Spos_data').rglob('*.txt')):
        with np.load(cache) as z:
            return Dataset(**{k: z[k] for k in z.files})
    pressure, people, actions, frames, poses, groups = [], [], [], [], [], []
    for path in sorted((DATA/'Spos_data').glob('*/*.txt')):
        match = re.search(r'_(\d+)\.txt$', path.name)
        if not match or not 1 <= int(match[1]) <= 21:
            continue
        action = int(match[1])
        data = parse_txt(path)
        if len(data) % 2:
            raise ValueError(f'{path}: 镜像帧数应为偶数')
        n = len(data)
        pressure.extend(data)
        people.extend([path.parent.name]*n)
        actions.extend([action]*n)
        frames.extend(range(n))
        poses.extend([pose_for_action(action)]*n)
        groups.extend(f'{path.parent.name}:{canonical_action(action)}:{i % (n//2)}' for i in range(n))
    dataset = Dataset(*(np.asarray(x) for x in (pressure, people, actions, frames, poses, groups)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{k:v for k,v in vars(dataset).items() if v is not None})
    return dataset


def load_regions():
    records = json.loads((DATA/'structure_data/data.json').read_text())
    pressure, people, actions, frames, poses, groups, boxes = [], [], [], [], [], [], []
    for r in records:
        p = np.fromstring(r['data'], sep=',', dtype=np.float32)
        if p.size != 1056 or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('结构化数据必须是有限、非负的 44×24 压力值')
        coords = r['region'].split()
        if len(coords) != 24:
            raise ValueError('region 应为 24 个坐标')
        b = np.array([[float(coords[2*i]), float(coords[12+2*i]),
                       float(coords[2*i+1]), float(coords[13+2*i])] for i in range(5)])
        if not np.isfinite(b).all() or np.any(b[:,2:] <= b[:,:2]) or np.any(b < 0) or np.any(b > [24,44,24,44]):
            raise ValueError(f'非法五区域标注: {r["people_name"]} {r["action"]}')
        pressure.append(p.reshape(SHAPE)); people.append(r['people_name'])
        actions.append(r['action']); frames.append(r['frame']); poses.append(r['sleep_pos'])
        groups.append(f'{r["people_name"]}:{canonical_action(r["action"])}:{r["frame"]}')
        boxes.append(b)
    return Dataset(*(np.asarray(x) for x in (pressure, people, actions, frames, poses, groups, boxes)))


def split_groups(groups, fraction=.3, seed=SEED):
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(groups, groups=groups))
    assert not set(groups[train]) & set(groups[test])
    return train, test


def augment(pressure, seed=SEED):
    """Sensor gain/noise only: no repeated reflection; rectangle labels unchanged."""
    rng = np.random.default_rng(seed)
    gain = rng.uniform(.9, 1.1, (len(pressure), 1, 1))
    noise = rng.normal(0, 1.5, pressure.shape) * (pressure > 0)
    return np.maximum(pressure*gain+noise, 0).astype(np.float32)
