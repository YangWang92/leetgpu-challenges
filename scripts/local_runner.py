#!/usr/bin/env python3
"""Local judge: run LeetGPU challenges on your own GPU without the leetgpu.com backend.

Difference from run_challenge.py: run_challenge.py submits a solution to the remote
grading service; this script grades correctness and measures performance locally,
using the challenge's own reference_impl and test cases.

Supported languages: cuda / triton / pytorch
  - cuda   : compile solution.cu into a .so with nvcc, call solve() via ctypes
             (tensors are passed as device pointers).
  - triton : import solve() from solution.py and call it with GPU tensors.
  - pytorch: same as triton.

Solution file convention (matches run_challenge.py):
    <challenge_dir>/solution/solution.cu    # cuda
    <challenge_dir>/solution/solution.py     # triton / pytorch
Use --solution to point at a different path.

Usage:
    python scripts/local_runner.py challenges/easy/1_vector_add --language cuda
    python scripts/local_runner.py challenges/easy/1_vector_add --language triton --action perf
    python scripts/local_runner.py challenges/easy/1_vector_add --language pytorch --action all
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CHALLENGES_DIR = REPO_ROOT / "challenges"

LANG_EXT = {"cuda": "cu", "triton": "py", "pytorch": "py"}


# ── Load challenge.py ──────────────────────────────────────────────
def load_challenge(challenge_dir: Path, device: str):
    """Import challenge.py dynamically and instantiate Challenge."""
    # challenge.py uses `from core.challenge_base import ...`, so challenges/ must be on the path.
    sys.path.insert(0, str(CHALLENGES_DIR))
    spec = importlib.util.spec_from_file_location("challenge", challenge_dir / "challenge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Challenge(device=device)


def load_python_solve(solution_path: Path) -> Callable:
    """Import the solve function from solution.py (shared by triton / pytorch)."""
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "solve"):
        raise AttributeError(f"{solution_path} has no solve() function")
    return mod.solve


# ── Compile + load a CUDA solution ────────────────────────────────
def find_nvcc() -> str:
    return shutil.which("nvcc") or "/usr/local/cuda-13.3/bin/nvcc"


def build_cuda_solve(solution_path: Path, arch: str) -> Callable:
    """Compile solution.cu into a shared library and return a callable solve (ctypes)."""
    so_path = solution_path.with_suffix(".so")
    cmd = [
        find_nvcc(), "-O3", "-shared", "-Xcompiler", "-fPIC",
        f"-arch={arch}", "-o", str(so_path), str(solution_path),
    ]
    print(f"  compiling: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("nvcc compilation failed")
    lib = ctypes.CDLL(str(so_path))
    return lib.solve


def call_cuda_solve(solve, sig: Dict[str, tuple], args: Dict[str, Any]) -> None:
    """Call solve in signature order: tensors as device pointers, scalars by their ctype."""
    argtypes, call = [], []
    for name, (ctype, _dir) in sig.items():
        v = args[name]
        if isinstance(v, torch.Tensor):
            argtypes.append(ctypes.c_void_p)              # any pointer has the same ABI
            call.append(ctypes.c_void_p(v.data_ptr()))    # data_ptr() is the device address
        else:
            argtypes.append(ctype)
            call.append(v)
    solve.argtypes = argtypes
    solve.restype = None
    solve(*call)
    torch.cuda.synchronize()


# ── Run a single test case ────────────────────────────────────────
def clone_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in case.items()}


def run_case(challenge, sig, run_solution: Callable[[Dict[str, Any]], None],
             case: Dict[str, Any]) -> tuple[bool, float]:
    """Run one case: solve writes outputs, then compare against reference_impl.

    Returns (passed, max_abs_err).
    """
    order = list(sig.keys())

    # Reference implementation (independent copies of inputs/outputs).
    ref = clone_case(case)
    challenge.reference_impl(*[ref[n] for n in order])

    # User solution (a separate copy; solve writes into its out/inout tensors).
    sol = clone_case(case)
    run_solution(sol)

    ok, max_err = True, 0.0
    for name, (_ctype, direction) in sig.items():
        if direction in ("out", "inout") and isinstance(sol[name], torch.Tensor):
            a, b = sol[name].float(), ref[name].float()
            max_err = max(max_err, (a - b).abs().max().item())
            if not torch.allclose(sol[name], ref[name], atol=challenge.atol, rtol=challenge.rtol):
                ok = False
    return ok, max_err


def make_runner(language: str, sig, solution_path: Path, arch: str) -> Callable:
    """Return a run_solution(args_dict) closure dispatched by language."""
    order = list(sig.keys())
    if language == "cuda":
        solve = build_cuda_solve(solution_path, arch)
        return lambda args: call_cuda_solve(solve, sig, args)
    else:  # triton / pytorch: pass positionally (triton's solve may use lowercase param names)
        solve = load_python_solve(solution_path)
        return lambda args: (solve(*[args[n] for n in order]), torch.cuda.synchronize())


# ── Actions ───────────────────────────────────────────────────────
def action_functional(challenge, sig, run_solution) -> bool:
    cases = challenge.generate_functional_test()
    print(f"\n== Functional tests ({len(cases)} cases) ==")
    passed = 0
    for i, case in enumerate(cases):
        try:
            ok, err = run_case(challenge, sig, run_solution, case)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:02d}] ERROR: {type(e).__name__}: {e}")
            continue
        print(f"  [{i:02d}] {'PASS' if ok else 'FAIL'}  max_abs_err={err:.3e}")
        passed += ok
    print(f"  -> {passed}/{len(cases)} passed")
    return passed == len(cases)


def action_perf(challenge, sig, run_solution, iters: int = 50) -> None:
    case = challenge.generate_performance_test()
    print("\n== Performance test ==")
    ok, err = run_case(challenge, sig, run_solution, case)  # verify correctness first
    print(f"  correctness: {'PASS' if ok else 'FAIL'}  max_abs_err={err:.3e}")

    work = clone_case(case)
    run_solution(work)  # warm up
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_solution(work)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"  avg latency: {ms:.3f} ms  (over {iters} iters; input size defined in challenge.py)")


def main() -> int:
    p = argparse.ArgumentParser(description="LeetGPU local judge")
    p.add_argument("challenge_path", type=Path,
                   help="challenge directory, e.g. challenges/easy/1_vector_add")
    p.add_argument("--language", default="cuda", choices=list(LANG_EXT), help="solution language")
    p.add_argument("--action", default="test", choices=["test", "perf", "all"],
                   help="test=functional, perf=performance, all=both")
    p.add_argument("--solution", type=Path, default=None, help="explicit path to the solution file")
    p.add_argument("--arch", default="sm_120", help="CUDA target arch (default sm_120 = Blackwell)")
    p.add_argument("--device", default="cuda", help="torch device")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. A CUDA-enabled PyTorch is required (see README_LOCAL.md).")
        return 1

    challenge_dir = args.challenge_path.resolve()
    if not (challenge_dir / "challenge.py").exists():
        print(f"Error: no challenge.py found in {challenge_dir}")
        return 1

    solution_path = args.solution or (
        challenge_dir / "solution" / f"solution.{LANG_EXT[args.language]}"
    )
    if not solution_path.exists():
        print(f"Error: solution file not found: {solution_path}")
        print(f"Hint: create {solution_path.parent}/ and add solution.{LANG_EXT[args.language]}")
        return 1

    challenge = load_challenge(challenge_dir, args.device)
    sig = challenge.get_solve_signature()
    print(f"challenge : {challenge.name}")
    print(f"language  : {args.language}   solution: {solution_path}")
    print(f"tolerance : atol={challenge.atol}, rtol={challenge.rtol}")

    run_solution = make_runner(args.language, sig, solution_path, args.arch)

    ok = True
    if args.action in ("test", "all"):
        ok = action_functional(challenge, sig, run_solution)
    if args.action in ("perf", "all"):
        action_perf(challenge, sig, run_solution)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
