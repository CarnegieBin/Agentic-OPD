"""Search-OPD trainer with best-average validation checkpoint retention."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from verl_tool.trainer.ppo.ray_trainer import AgentRayPPOTrainer as BaseAgentRayPPOTrainer

from .adaptive_teacher_opd import select_and_publish_teacher
from .checkpoint_retention_opd import average_bamboogle_musique_triviaqa_em
from .config_opd import TeacherSpec


class AgentRayPPOTrainerOPD(BaseAgentRayPPOTrainer):
    """Reuse the Search-R1 trainer while selecting one best HF actor checkpoint.

    The inherited trainer still owns dataset construction, rollout, validation,
    and FSDP export. This subclass selects the retained student checkpoint by
    equal-weight held-out EM across Bamboogle, Musique, and TriviaQA; ties
    prefer the newer checkpoint.
    """

    _CHECKPOINT_SELECTION_STRATEGY = "best_equal_weight_bamboogle_musique_triviaqa_test_em"

    def _validate(self):
        metrics = super()._validate()
        self._last_checkpoint_selection = average_bamboogle_musique_triviaqa_em(metrics)
        metrics.update(self._update_adaptive_teacher_selection())
        return metrics

    def _update_adaptive_teacher_selection(self) -> dict[str, object]:
        """Select a teacher and return scalar/string metrics for every logger."""

        opd_config = self.config.actor_rollout_ref.get("opd", {})
        state_path = opd_config.get("teacher_selection_state_path", None)
        teacher_root = opd_config.get("teacher_trajectory_root", None)
        teacher_metrics_path = opd_config.get("teacher_validation_metrics_path", None)
        trajectory_root = self.config.trainer.get("validation_trajectory_dir", None)
        step = int(self.global_steps)
        if not state_path or not teacher_root or not trajectory_root:
            return {}

        teacher_paths = [str(path) for path in opd_config.teacher_model_paths]
        teacher_names = [str(name) for name in opd_config.teacher_names]
        teacher_steps = list(opd_config.get("teacher_steps", [None] * len(teacher_paths)))
        specs = tuple(
            TeacherSpec(
                name=name,
                path=path,
                step=None if teacher_steps[index] is None else int(teacher_steps[index]),
            )
            for index, (name, path) in enumerate(zip(teacher_names, teacher_paths, strict=True))
        )
        student_step_dir = Path(str(trajectory_root)).expanduser() / f"global_step_{step}"
        payload = select_and_publish_teacher(
            specs,
            student_trajectory_dir=student_step_dir,
            teacher_trajectory_root=teacher_root,
            teacher_metrics_path=teacher_metrics_path,
            state_path=state_path,
            global_step=step,
            pruned_teachers=opd_config.get("pruned_teachers", []),
        )
        selected = payload["selected"]
        print(
            f"Adaptive teacher selection at student step {step}: "
            f"{selected['name']} (index={selected['index']}, "
            f"weighted_marginal_gain={int(selected['weighted_marginal_gain'])}, "
            f"marginal_questions={int(selected['marginal_question_count'])}, "
            f"switch_count={int(payload.get('switch_count', 0))})",
            flush=True,
        )
        selected_indices = [
            int(index)
            for index in payload.get("selected_teacher_indices", [selected["index"]])
        ]
        checkpoint = str(selected.get("checkpoint", selected.get("path", "")))
        teacher_global_step = int(
            selected.get("global_step", selected.get("step", -1))
        )
        weighted_gain = float(selected.get("weighted_marginal_gain", 0.0))
        marginal_count = int(selected.get("marginal_question_count", 0))
        switch_count = int(payload.get("switch_count", 0))
        # These are intentionally separate fields: SwanLab can chart the
        # numeric values while retaining the exact selected index list and
        # immutable checkpoint path as text metadata.
        return {
            "adaptive_teacher/selected_teacher_indices": json.dumps(
                selected_indices,
                separators=(",", ":"),
            ),
            "adaptive_teacher/selected_teacher_index": float(selected_indices[0]),
            "adaptive_teacher/teacher_checkpoint": checkpoint,
            "adaptive_teacher/teacher_global_step": float(teacher_global_step),
            "adaptive_teacher/weighted_marginal_gain": weighted_gain,
            "adaptive_teacher/marginal_question_count": float(marginal_count),
            "adaptive_teacher/teacher_switch_count": float(switch_count),
        }

    def _initialize_checkpoint_selection_state(self) -> None:
        if not hasattr(self, "_best_checkpoint_selection"):
            self._best_checkpoint_selection: dict[str, Any] | None = None
        if not hasattr(self, "_checkpoint_selection_history"):
            self._checkpoint_selection_history: list[dict[str, Any]] = []

    @staticmethod
    def _json_float(value: object) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Checkpoint selection value must be finite, got {value!r}")
        return result

    def _write_best_checkpoint_manifest(self) -> None:
        root = self._checkpoint_root()
        root.mkdir(parents=True, exist_ok=True)
        best = self._best_checkpoint_selection
        payload = {
            "version": 2,
            "strategy": self._CHECKPOINT_SELECTION_STRATEGY,
            "keep_limit": 1,
            "metric": "mean(Bamboogle EM, Musique EM, TriviaQA EM)",
            "best": best,
            "retained": [] if best is None else [best],
            "history": self._checkpoint_selection_history,
        }
        manifest_path = root / "checkpoint_selection.json"
        temporary_path = root / ".checkpoint_selection.json.tmp"
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)

    def _append_checkpoint_validation_record(
        self,
        step: int,
        *,
        candidate_selected: bool,
        evicted_steps: list[int],
    ) -> None:
        configured_path = self.config.trainer.get("validation_checkpoint_metrics_path", None)
        if not configured_path:
            return

        selection = dict(self._last_checkpoint_selection)
        metrics_path = Path(str(configured_path)).expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_step": int(step),
            "checkpoint": f"global_step_{int(step)}",
            "candidate_selected": bool(candidate_selected),
            "evicted_steps": [int(item) for item in evicted_steps],
            "selection": selection,
            "datasets": {
                name: {
                    "em": self._json_float(info["em"]),
                    "metric_key": str(info["metric_key"]),
                    "response_count": int(info["response_count"]),
                }
                for name, info in selection["datasets"].items()
            },
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _remove_other_checkpoints(self, root: Path, keep_path: Path) -> list[int]:
        removed_steps: list[int] = []
        for candidate_path in sorted(root.glob("global_step_*")):
            if candidate_path == keep_path or not candidate_path.is_dir():
                continue
            try:
                removed_steps.append(int(candidate_path.name.removeprefix("global_step_")))
            except ValueError:
                continue
            shutil.rmtree(candidate_path)
        return removed_steps

    def _save_checkpoint(self):
        """Export only a new best average-EM checkpoint."""

        save_freq = int(self.config.trainer.save_freq)
        if save_freq <= 0 or self.global_steps % save_freq != 0:
            print(
                f"Skipping non-periodic checkpoint request at step {self.global_steps}; "
                f"HF checkpoints are evaluated every {save_freq} steps."
            )
            return

        if getattr(self, "_last_validation_step", None) != self.global_steps:
            raise RuntimeError(
                f"Checkpoint step {self.global_steps} has no matching validation result; "
                "validation must run immediately before checkpoint selection."
            )
        if self.config.trainer.default_hdfs_dir is not None:
            raise ValueError("Validation-selected HF-only checkpoints support local storage only")

        self._initialize_checkpoint_selection_state()
        step = int(self.global_steps)
        candidate = dict(self._last_checkpoint_selection)
        candidate["global_step"] = step
        candidate["path"] = f"global_step_{step}"
        candidate["mean_em"] = self._json_float(candidate["mean_em"])

        best = self._best_checkpoint_selection
        candidate_selected = best is None or (
            candidate["mean_em"] >= self._json_float(best["mean_em"])
        )
        evicted_steps = [] if best is None else [int(best["global_step"])]
        decision = {
            "global_step": step,
            "candidate": candidate,
            "candidate_selected": candidate_selected,
            "evicted_steps": evicted_steps if candidate_selected else [],
        }

        if not candidate_selected:
            self._checkpoint_selection_history.append(decision)
            self._write_best_checkpoint_manifest()
            self._append_checkpoint_validation_record(
                step,
                candidate_selected=False,
                evicted_steps=[],
            )
            print(
                f"Rejected checkpoint candidate step {step}: "
                f"mean_em={candidate['mean_em']:.6f}, "
                f"best_step={best['global_step']}, best_mean_em={best['mean_em']:.6f}"
            )
            return

        root = self._checkpoint_root()
        root.mkdir(parents=True, exist_ok=True)
        staging_root = root / f".global_step_{step}.staging"
        actor_staging_path = staging_root / "actor"
        final_path = root / f"global_step_{step}"
        if staging_root.exists():
            shutil.rmtree(staging_root)

        try:
            self.actor_rollout_wg.save_checkpoint(
                str(actor_staging_path),
                None,
                step,
                max_ckpt_to_keep=None,
            )
            hf_staging_path = actor_staging_path / "huggingface"
            weight_files = list(hf_staging_path.glob("*.safetensors")) + list(
                hf_staging_path.glob("*.bin")
            )
            if not hf_staging_path.is_dir() or not weight_files:
                raise RuntimeError(f"HuggingFace checkpoint export is incomplete: {hf_staging_path}")

            if final_path.exists():
                shutil.rmtree(final_path)
            os.replace(hf_staging_path, final_path)

            removed_steps = self._remove_other_checkpoints(root, final_path)
            if removed_steps:
                evicted_steps = sorted(set(evicted_steps) | set(removed_steps))
                decision["evicted_steps"] = evicted_steps
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)

        self._best_checkpoint_selection = candidate
        self._checkpoint_selection_history.append(decision)
        self._write_best_checkpoint_manifest()
        self._append_checkpoint_validation_record(
            step,
            candidate_selected=True,
            evicted_steps=evicted_steps,
        )
        print(
            f"Saved best-average-EM HF checkpoint step {step} to {final_path}; "
            f"bamboogle_em={candidate['bamboogle_em']:.6f}, "
            f"musique_em={candidate['musique_em']:.6f}, "
            f"triviaqa_em={candidate['triviaqa_em']:.6f}, "
            f"mean_em={candidate['mean_em']:.6f}"
        )
