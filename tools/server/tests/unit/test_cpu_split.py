# Covers the "Track A" locality-aware CPU device split (--cpu-split auto|off),
# see ~/.claude/plans/create-a-plan-to-abundant-whistle.md for the full design.
# GGML_CPU_LOCALITY_SYSFS_ROOT lets the auto case exercise a real >1-domain split
# deterministically, since the CI/dev host itself may only have one L3 domain.

import os

import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def make_mock_sysfs_2domain(tmp_path, cores_per_domain=4) -> str:
    root = tmp_path / "mock-sysfs"
    for domain in range(2):
        lo = domain * cores_per_domain
        hi = lo + cores_per_domain - 1
        shared_list = f"{lo}-{hi}"
        for cpu in range(lo, hi + 1):
            d = root / f"cpu{cpu}" / "cache" / "index3"
            d.mkdir(parents=True)
            (d / "shared_cpu_list").write_text(shared_list + "\n")
    return str(root)


def test_cpu_split_off_serves_completion():
    global server
    server.cpu_split = "off"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_cpu_split_auto_single_domain_serves_completion():
    # No mocked sysfs: real host topology, expected to be a single-domain no-op here.
    global server
    server.cpu_split = "auto"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_cpu_split_auto_mocked_multi_domain_serves_completion(tmp_path):
    # Mocked 2-domain sysfs: exercises the real N>1 registry + weighted-split path.
    global server
    server.cpu_split = "auto"
    server.cpu_locality_sysfs_root = make_mock_sysfs_2domain(tmp_path)
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
