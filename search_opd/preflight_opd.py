"""Command-line teacher/student readiness check used by ``train.sh``."""

from __future__ import annotations

import argparse

from search_opd.config_opd import resolve_teacher_specs
from search_opd.teacher_readiness_opd import validate_teacher_specs_ready


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate immutable Search-OPD teachers before Ray startup"
    )
    parser.add_argument("--student", required=True, help="Student Hugging Face directory")
    parser.add_argument(
        "--output",
        required=True,
        help="Training checkpoint/output directory",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teacher-manifest", help="Ordered JSON teacher manifest")
    source.add_argument(
        "--teacher-paths",
        help="Ordered comma-separated or JSON teacher paths",
    )
    parser.add_argument(
        "--no-verify-safetensors",
        action="store_true",
        help="Skip safetensors header/inventory reads (test-only escape hatch)",
    )
    parser.add_argument(
        "--compute-digest",
        action="store_true",
        help="Compute and print a deterministic digest for every teacher",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs, _ = resolve_teacher_specs(
        manifest_path=args.teacher_manifest,
        model_paths=args.teacher_paths,
    )
    infos = validate_teacher_specs_ready(
        specs,
        student_path=args.student,
        output_path=args.output,
        verify_safetensors=not args.no_verify_safetensors,
        compute_digest=args.compute_digest,
    )
    print(
        "STUDENT_ELIGIBLE=1 "
        f"path={args.student}",
        flush=True,
    )
    for spec, info in zip(specs, infos, strict=True):
        digest = info.directory_sha256 or "not-computed"
        print(
            f"TEACHER_ELIGIBLE=1 name={spec.name} path={info.path} "
            f"shards={len(info.shard_names)} sha256={digest}",
            flush=True,
        )


if __name__ == "__main__":
    main()
