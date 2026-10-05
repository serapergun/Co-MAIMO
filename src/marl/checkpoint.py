"""Checkpoint I/O for the lightweight B4 MAPPO policy."""

import json
import hashlib
from pathlib import Path
import numpy as np

from .mappo import LinearMAPPO


def save_checkpoint(policy, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        actor_w=policy.actor_w,
        critic_w=policy.critic_w,
        obs_dim=policy.obs_dim,
        state_dim=policy.state_dim,
        action_dim=policy.action_dim,
    )
    meta_path = path.with_suffix(path.suffix + ".json")
    meta_path.write_text(
        json.dumps(metadata or {}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(path)


def checkpoint_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_metadata(path):
    meta_path = Path(path).with_suffix(Path(path).suffix + ".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"checkpoint metadata missing: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_checkpoint(path):
    data = np.load(Path(path), allow_pickle=False)
    policy = LinearMAPPO(
        int(data["obs_dim"]),
        int(data["state_dim"]),
        int(data["action_dim"]),
        seed=0,
    )
    policy.actor_w = np.asarray(data["actor_w"], dtype=float)
    policy.critic_w = np.asarray(data["critic_w"], dtype=float)
    return policy
