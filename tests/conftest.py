import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
SMOKE = ROOT / "configs" / "smoke.yaml"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="session")
def smoke_assets():
    """Tiny Qwen3 + synthetic prepared data, built once per test session."""
    from capprune.config import data_dir, load_config
    from capprune.data.prepare import prepare
    from capprune.model import build_tiny_model, load_tokenizer, tokenizer_id

    build_tiny_model(ROOT / ".smoke" / "tiny-qwen3")
    cfg = load_config(SMOKE)
    prepare(cfg, load_tokenizer(cfg), data_dir(cfg), tokenizer_id(cfg))
    return cfg


def endpoint_override(port: int) -> str:
    return f"controller.endpoints=[{{name: mock, api: openai, base_url: 'http://127.0.0.1:{port}/v1'}}]"
