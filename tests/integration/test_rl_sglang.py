"""
Integration tests for RL training with SGLang backend.

This module tests the complete RL training pipeline using SGLang as the
inference backend instead of vLLM, verifying that:
- The SGLang server starts correctly with prime-rl config
- Weight updates work during training
- Reward improves over training steps
"""

from functools import partial
from pathlib import Path
from typing import Callable

import pytest

from tests.conftest import ProcessResult
from tests.utils import check_no_error, check_number_goes_up_or_down, check_number_in_range, strip_escape_codes

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


TIMEOUT = 600  # 10 minutes


@pytest.fixture(scope="module")
def wandb_name(branch_name: str) -> str:
    """Fixture for W&B name for SGLang RL CI integration tests."""
    return f"test-rl-sglang-{branch_name}"


@pytest.fixture(scope="module")
def rl_sglang_process(
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    """Run RL training with SGLang backend."""
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl/start.toml",
        "--inference.backend",
        "sglang",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]
    return run_process(cmd, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def rl_sglang_resume_process(
    rl_sglang_process,  # Resume training can only start when regular RL process is finished
    run_process: Callable[..., ProcessResult],
    output_dir: Path,
    wandb_project: str,
    wandb_name: str,
) -> ProcessResult:
    """Resume RL training with SGLang backend."""
    if rl_sglang_process.returncode != 0:
        pytest.skip("SGLang RL process failed")
    wandb_name = f"{wandb_name}-resume"
    cmd = [
        "uv",
        "run",
        "rl",
        "@",
        "configs/ci/integration/rl/resume.toml",
        "--inference.backend",
        "sglang",
        "--wandb.project",
        wandb_project,
        "--wandb.name",
        wandb_name,
        "--output-dir",
        output_dir.as_posix(),
    ]

    return run_process(cmd, timeout=TIMEOUT)


check_reward_goes_up = partial(check_number_goes_up_or_down, go_up=True, pattern=r"Reward:\s*(\d+\.\d{4})")
check_reward_in_range = partial(check_number_in_range, pattern=r"Reward:\s*(\d+\.\d{4})")


@pytest.fixture(scope="module")
def test_no_error(rl_sglang_process: ProcessResult, output_dir: Path):
    """Tests that the SGLang RL process does not fail."""
    check_no_error(rl_sglang_process, output_dir)


def test_sglang_reward_goes_up(rl_sglang_process: ProcessResult, test_no_error, output_dir: Path):
    """Tests that the reward goes up in the SGLang RL process."""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_goes_up(orchestrator_stdout)


def test_sglang_reward_in_range(rl_sglang_process: ProcessResult, test_no_error, output_dir: Path):
    """Tests that the reward is in range in the SGLang RL process."""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.65)


@pytest.fixture(scope="module")
def test_no_error_resume(rl_sglang_resume_process: ProcessResult, output_dir: Path):
    """Tests that the SGLang RL resume process does not fail."""
    check_no_error(rl_sglang_resume_process, output_dir)


def test_sglang_reward_in_range_resume(
    rl_sglang_resume_process: ProcessResult, test_no_error_resume, output_dir: Path
):
    """Tests that the reward is in range in the SGLang RL resume process."""
    with open(output_dir / "logs" / "orchestrator.stdout", "r") as f:
        orchestrator_stdout = strip_escape_codes(f.read()).splitlines()
    check_reward_in_range(orchestrator_stdout, min_threshold=0.65)
