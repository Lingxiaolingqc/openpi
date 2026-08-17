"""Run ACT smoke gates, the 10-episode overfit gate, or full training.

This wrapper uses the repository-pinned Hugging Face LeRobot ACT implementation.
It adds SO-101 schema/audit gates, deterministic episode splits, gradient
accumulation, auditable checkpoint safety markers, and explicit resume handling.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.so101.act import audit_dataset
from examples.so101.act import common
from examples.so101.act.policy_utils import move_batch_to_device


def _config_preparse() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=common.CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return args.config


def _build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    dataset_cfg = config.get("dataset", {})
    split_cfg = config.get("split", {})
    policy_cfg = config.get("policy", {})
    training_cfg = config.get("training", {})
    overfit_cfg = config.get("overfit", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(config["_path"]))
    parser.add_argument(
        "--mode",
        choices=("forward", "backward", "one-step", "overfit", "train"),
        required=True,
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("OPENPI_ACT_REPO_ID", dataset_cfg.get("repo_id")),
    )
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("HF_LEROBOT_HOME"))
    parser.add_argument(
        "--action-semantics",
        default=dataset_cfg.get("action_semantics", common.ACTION_SEMANTICS),
    )
    parser.add_argument("--expected-camera-key", action="append", dest="expected_camera_keys")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=float(split_cfg.get("validation_fraction", 0.2)),
    )
    parser.add_argument("--split-seed", type=int, default=int(split_cfg.get("seed", 42)))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OPENPI_ACT_OUTPUT_DIR", training_cfg.get("output_dir", "outputs/act"))),
    )
    parser.add_argument("--run-name", default=os.environ.get("OPENPI_ACT_RUN_NAME", "act"))
    parser.add_argument("--device", default=policy_cfg.get("device", "cuda"))
    parser.add_argument("--batch-size", type=int, default=int(training_cfg.get("batch_size", 8)))
    parser.add_argument("--num-workers", type=int, default=int(training_cfg.get("num_workers", 4)))
    parser.add_argument("--steps", type=int, default=int(training_cfg.get("steps", 100_000)))
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=int(training_cfg.get("save_frequency", 5000)),
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=int(training_cfg.get("log_frequency", 100)),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(training_cfg.get("gradient_accumulation_steps", 1)),
    )
    parser.add_argument("--seed", type=int, default=int(training_cfg.get("seed", 42)))
    parser.add_argument("--chunk-size", type=int, default=int(policy_cfg.get("chunk_size", 100)))
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=int(policy_cfg.get("n_action_steps", 10)),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=float(policy_cfg.get("temporal_ensemble_coeff", -1.0)),
        help="Negative disables temporal ensembling; enabling it requires n-action-steps=1.",
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=bool(policy_cfg.get("use_amp", False)),
    )
    parser.add_argument("--vision-backbone", default=policy_cfg.get("vision_backbone", "resnet18"))
    parser.add_argument(
        "--pretrained-backbone-weights",
        default=policy_cfg.get("pretrained_backbone_weights", "ResNet18_Weights.IMAGENET1K_V1"),
        help="Use an empty string to disable pretrained backbone weights.",
    )
    parser.add_argument(
        "--replace-final-stride-with-dilation",
        action=argparse.BooleanOptionalAction,
        default=bool(policy_cfg.get("replace_final_stride_with_dilation", False)),
    )
    parser.add_argument(
        "--pre-norm",
        action=argparse.BooleanOptionalAction,
        default=bool(policy_cfg.get("pre_norm", False)),
    )
    parser.add_argument("--dim-model", type=int, default=int(policy_cfg.get("dim_model", 512)))
    parser.add_argument("--n-heads", type=int, default=int(policy_cfg.get("n_heads", 8)))
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=int(policy_cfg.get("dim_feedforward", 3200)),
    )
    parser.add_argument(
        "--feedforward-activation",
        default=policy_cfg.get("feedforward_activation", "relu"),
    )
    parser.add_argument(
        "--n-encoder-layers",
        type=int,
        default=int(policy_cfg.get("n_encoder_layers", 4)),
    )
    parser.add_argument(
        "--n-decoder-layers",
        type=int,
        default=int(policy_cfg.get("n_decoder_layers", 1)),
    )
    parser.add_argument(
        "--use-vae",
        action=argparse.BooleanOptionalAction,
        default=bool(policy_cfg.get("use_vae", True)),
    )
    parser.add_argument("--latent-dim", type=int, default=int(policy_cfg.get("latent_dim", 32)))
    parser.add_argument(
        "--n-vae-encoder-layers",
        type=int,
        default=int(policy_cfg.get("n_vae_encoder_layers", 4)),
    )
    parser.add_argument("--dropout", type=float, default=float(policy_cfg.get("dropout", 0.1)))
    parser.add_argument("--kl-weight", type=float, default=float(policy_cfg.get("kl_weight", 10.0)))
    parser.add_argument(
        "--optimizer-lr",
        type=float,
        default=float(policy_cfg.get("optimizer_lr", 1e-5)),
    )
    parser.add_argument(
        "--optimizer-weight-decay",
        type=float,
        default=float(policy_cfg.get("optimizer_weight_decay", 1e-4)),
    )
    parser.add_argument(
        "--optimizer-lr-backbone",
        type=float,
        default=float(policy_cfg.get("optimizer_lr_backbone", 1e-5)),
    )
    parser.add_argument(
        "--use-imagenet-stats",
        action=argparse.BooleanOptionalAction,
        default=bool(training_cfg.get("use_imagenet_stats", True)),
    )
    parser.add_argument(
        "--image-augmentation",
        action=argparse.BooleanOptionalAction,
        default=bool(training_cfg.get("image_augmentation", False)),
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--overfit-gate-report", type=Path)
    parser.add_argument("--allow-train-without-overfit-gate", action="store_true")
    parser.add_argument(
        "--overfit-episodes",
        type=int,
        default=int(overfit_cfg.get("episode_count", 10)),
    )
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=int(overfit_cfg.get("steps", 2000)),
    )
    parser.add_argument(
        "--overfit-evaluation-batches",
        type=int,
        default=int(overfit_cfg.get("evaluation_batches", 16)),
    )
    parser.add_argument(
        "--overfit-maximum-loss-ratio",
        type=float,
        default=float(overfit_cfg.get("maximum_final_to_initial_loss_ratio", 0.7)),
    )
    parser.add_argument("--audit-scan-batch-size", type=int, default=4096)
    parser.add_argument("--audit-frame0-sample-episodes", type=int, default=0)
    return parser


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "save_frequency": args.save_frequency,
        "log_frequency": args.log_frequency,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "dim_model": args.dim_model,
        "n_heads": args.n_heads,
        "dim_feedforward": args.dim_feedforward,
        "n_encoder_layers": args.n_encoder_layers,
        "n_decoder_layers": args.n_decoder_layers,
        "latent_dim": args.latent_dim,
        "n_vae_encoder_layers": args.n_vae_encoder_layers,
        "overfit_episodes": args.overfit_episodes,
        "overfit_steps": args.overfit_steps,
        "overfit_evaluation_batches": args.overfit_evaluation_batches,
        "audit_scan_batch_size": args.audit_scan_batch_size,
    }
    invalid = {key: value for key, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"training arguments must be positive: {invalid}")
    if args.num_workers < 0 or args.audit_frame0_sample_episodes < 0:
        raise ValueError("worker and frame0 sample counts must be non-negative")
    if args.n_action_steps > args.chunk_size:
        raise ValueError("n_action_steps cannot exceed chunk_size")
    if args.temporal_ensemble_coeff >= 0 and args.n_action_steps != 1:
        raise ValueError("temporal ensembling requires n_action_steps=1")
    if args.dim_model % args.n_heads:
        raise ValueError("dim_model must be divisible by n_heads")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if args.kl_weight < 0 or args.optimizer_lr <= 0 or args.optimizer_lr_backbone <= 0:
        raise ValueError("KL weight must be non-negative and optimizer learning rates must be positive")
    if args.optimizer_weight_decay < 0:
        raise ValueError("optimizer weight decay must be non-negative")
    if not 0.0 < args.overfit_maximum_loss_ratio < 1.0:
        raise ValueError("overfit maximum loss ratio must be strictly between 0 and 1")
    if args.mode not in {"overfit", "train"} and args.resume_checkpoint:
        raise ValueError("only overfit or full training may resume a checkpoint")


def _prepare_output_dir(path: Path, *, resume: bool) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a new path or pass --resume-checkpoint"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _validate_resume_checkpoint(
    checkpoint: Path,
    contract: common.DatasetContract,
) -> None:
    pretrained_dir, _ = common.resolve_pretrained_model_path(checkpoint)
    marker = common.require_simulation_only_marker(pretrained_dir)
    if marker.get("repo_id") != contract.repo_id:
        raise ValueError("resume checkpoint repo_id does not match the selected dataset")
    if marker.get("metadata_fingerprint") != contract.metadata_fingerprint:
        raise ValueError("resume checkpoint metadata fingerprint does not match the selected dataset")


def _prepare_split(
    args: argparse.Namespace,
    contract: common.DatasetContract,
    output_dir: Path,
) -> common.EpisodeSplit:
    if args.split_file:
        split = common.load_episode_split(args.split_file.expanduser().resolve(), contract)
    elif args.resume_checkpoint:
        split = common.load_episode_split(output_dir / common.SPLIT_FILENAME, contract)
    else:
        split = common.deterministic_episode_split(
            contract,
            validation_fraction=args.validation_fraction,
            seed=args.split_seed,
        )
    common.write_json(output_dir / common.SPLIT_FILENAME, split.to_dict())
    return split


def _require_overfit_gate(
    path: Path | None,
    contract: common.DatasetContract,
    split: common.EpisodeSplit,
) -> None:
    if path is None:
        raise ValueError(
            "full training requires --overfit-gate-report from a passed 10-episode gate; "
            "use --allow-train-without-overfit-gate only for an explicitly audited exception"
        )
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("passed") is not True:
        raise ValueError(f"overfit gate did not pass: {path}")
    if report.get("repo_id") != contract.repo_id:
        raise ValueError("overfit gate repo_id does not match the training dataset")
    if report.get("metadata_fingerprint") != contract.metadata_fingerprint:
        raise ValueError("overfit gate metadata fingerprint does not match the current dataset")
    if tuple(report.get("validation_episodes", ())) != split.validation_episodes:
        raise ValueError("overfit gate used a different held-out episode split")


def _create_lerobot_pipeline(
    args: argparse.Namespace,
    *,
    location: common.DatasetLocation,
    episodes: tuple[int, ...],
    output_dir: Path,
):
    try:
        from lerobot.common.datasets.factory import make_dataset
        from lerobot.common.optim.factory import make_optimizer_and_scheduler
        from lerobot.common.policies.act.configuration_act import ACTConfig
        from lerobot.common.policies.factory import make_policy
        from lerobot.configs.default import DatasetConfig
        from lerobot.configs.train import TrainPipelineConfig
    except ImportError as exc:
        raise RuntimeError("LeRobot ACT is required; run this command with `uv run`") from exc

    resume_pretrained = None
    resume_run_root = None
    if args.resume_checkpoint:
        resume_pretrained, resume_run_root = common.resolve_pretrained_model_path(args.resume_checkpoint)
        if resume_run_root.resolve() != output_dir.resolve():
            raise ValueError(f"resume checkpoint belongs to run {resume_run_root}, but --output-dir is {output_dir}")
        policy_config = ACTConfig.from_pretrained(resume_pretrained)
        if policy_config.type != "act":
            raise ValueError(f"resume checkpoint is not ACT: {policy_config.type!r}")
        policy_config.device = args.device
        policy_config.use_amp = args.use_amp
        policy_config.pretrained_path = str(resume_pretrained)
    else:
        temporal_coeff = None if args.temporal_ensemble_coeff < 0 else args.temporal_ensemble_coeff
        policy_config = ACTConfig(
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            temporal_ensemble_coeff=temporal_coeff,
            device=args.device,
            use_amp=args.use_amp,
            vision_backbone=args.vision_backbone,
            pretrained_backbone_weights=(args.pretrained_backbone_weights or None),
            replace_final_stride_with_dilation=args.replace_final_stride_with_dilation,
            pre_norm=args.pre_norm,
            dim_model=args.dim_model,
            n_heads=args.n_heads,
            dim_feedforward=args.dim_feedforward,
            feedforward_activation=args.feedforward_activation,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            use_vae=args.use_vae,
            latent_dim=args.latent_dim,
            n_vae_encoder_layers=args.n_vae_encoder_layers,
            dropout=args.dropout,
            kl_weight=args.kl_weight,
            optimizer_lr=args.optimizer_lr,
            optimizer_weight_decay=args.optimizer_weight_decay,
            optimizer_lr_backbone=args.optimizer_lr_backbone,
        )

    dataset_config = DatasetConfig(
        repo_id=location.repo_id,
        root=str(location.dataset_path),
        episodes=list(episodes),
        use_imagenet_stats=args.use_imagenet_stats,
    )
    dataset_config.image_transforms.enable = args.image_augmentation
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        output_dir=output_dir,
        job_name=args.run_name,
        resume=bool(args.resume_checkpoint),
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_frequency,
        save_checkpoint=True,
        save_freq=args.save_frequency,
        use_policy_training_preset=True,
    )
    train_config.optimizer = policy_config.get_optimizer_preset()
    train_config.scheduler = policy_config.get_scheduler_preset()
    dataset = make_dataset(train_config)
    policy = make_policy(cfg=policy_config, ds_meta=dataset.meta)
    optimizer, scheduler = make_optimizer_and_scheduler(train_config, policy)
    return train_config, dataset, policy, optimizer, scheduler, resume_run_root


def _make_dataloader(dataset: Any, args: argparse.Namespace, *, shuffle: bool):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        pin_memory=str(args.device).startswith("cuda"),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )


def _next_batch(iterator: Any, dataloader: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def _forward_gate(policy: Any, batch: dict[str, Any], *, device: str) -> tuple[Any, dict[str, Any]]:
    import torch

    moved = move_batch_to_device(batch, device)
    policy.train()
    loss, details = policy.forward(moved)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError(f"ACT forward gate produced invalid loss: {loss}")
    print(f"act_forward_loss: {float(loss.detach().cpu()):.8f}")
    print("ACT_FORWARD_GATE_OK")
    return loss, details or {}


def _backward_gate(policy: Any, loss: Any) -> float:
    import torch

    policy.zero_grad(set_to_none=True)
    loss.backward()
    finite_gradients = []
    squared_norm = 0.0
    for parameter in policy.parameters():
        if parameter.grad is None:
            continue
        finite_gradients.append(bool(torch.isfinite(parameter.grad).all()))
        squared_norm += float(torch.sum(parameter.grad.detach().float().square()).cpu())
    if not finite_gradients or not all(finite_gradients):
        raise RuntimeError("ACT backward gate found missing or non-finite gradients")
    grad_norm = math.sqrt(squared_norm)
    if not math.isfinite(grad_norm) or grad_norm <= 0.0:
        raise RuntimeError(f"ACT backward gate produced invalid gradient norm: {grad_norm}")
    print(f"act_backward_grad_norm: {grad_norm:.8f}")
    print("ACT_BACKWARD_GATE_OK")
    return grad_norm


def _optimizer_step(
    *,
    policy: Any,
    optimizer: Any,
    scheduler: Any,
    grad_scaler: Any,
    dataloader: Any,
    iterator: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    import torch

    device_type = torch.device(args.device).type
    optimizer.zero_grad(set_to_none=True)
    losses = []
    started = time.perf_counter()
    for _ in range(args.gradient_accumulation_steps):
        batch, iterator = _next_batch(iterator, dataloader)
        batch = move_batch_to_device(batch, args.device)
        policy.train()
        amp_context = torch.autocast(device_type=device_type) if args.use_amp else nullcontext()
        with amp_context:
            loss, _ = policy.forward(batch)
            scaled_loss = loss / args.gradient_accumulation_steps
        if not torch.isfinite(loss):
            raise RuntimeError(f"ACT training produced non-finite loss: {loss}")
        grad_scaler.scale(scaled_loss).backward()
        losses.append(float(loss.detach().cpu()))
    grad_scaler.unscale_(optimizer)
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0, error_if_nonfinite=True)
    grad_norm = float(grad_norm_tensor.detach().cpu())
    changed_parameter = None
    changed_before = None
    for parameter in policy.parameters():
        if parameter.grad is not None and bool(torch.any(parameter.grad != 0)):
            changed_parameter = parameter
            changed_before = parameter.detach().clone()
            break
    if changed_parameter is None:
        raise RuntimeError("ACT optimizer gate could not find a parameter with a nonzero gradient")
    grad_scaler.step(optimizer)
    grad_scaler.update()
    if scheduler is not None:
        scheduler.step()
    if torch.equal(changed_before, changed_parameter.detach()):
        raise RuntimeError("ACT optimizer gate did not update a parameter with a nonzero gradient")
    return iterator, {
        "loss": float(np.mean(losses)),
        "grad_norm": grad_norm,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "update_seconds": time.perf_counter() - started,
    }


def _evaluate_loss(policy: Any, dataloader: Any, *, device: str, maximum_batches: int, seed: int) -> float:
    import torch

    torch.manual_seed(seed)
    policy.eval()
    losses = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index >= maximum_batches:
                break
            moved_batch = move_batch_to_device(batch, device)
            loss, _ = policy.forward(moved_batch)
            losses.append(float(loss.detach().cpu()))
    if not losses or not np.isfinite(losses).all():
        raise RuntimeError("ACT overfit evaluation did not produce finite losses")
    return float(np.mean(losses))


def _save_checkpoint(
    *,
    train_config: Any,
    policy: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: Path,
    contract: common.DatasetContract,
    split: common.EpisodeSplit,
    step: int,
    total_steps: int,
) -> Path:
    from lerobot.common.utils.train_utils import get_step_checkpoint_dir
    from lerobot.common.utils.train_utils import save_checkpoint
    from lerobot.common.utils.train_utils import update_last_checkpoint

    checkpoint_dir = get_step_checkpoint_dir(output_dir, total_steps, step)
    save_checkpoint(checkpoint_dir, step, train_config, policy, optimizer, scheduler)
    pretrained_dir = checkpoint_dir / "pretrained_model"
    common.write_contract_and_safety(pretrained_dir, contract)
    common.write_json(pretrained_dir / common.SPLIT_FILENAME, split.to_dict())
    try:
        update_last_checkpoint(checkpoint_dir)
    except OSError as exc:
        raise RuntimeError(
            "LeRobot could not update the checkpoints/last symlink; run training on a filesystem that supports symlinks"
        ) from exc
    return checkpoint_dir


def _load_resume_state(args: argparse.Namespace, optimizer: Any, scheduler: Any) -> tuple[int, Any, Any]:
    if not args.resume_checkpoint:
        return 0, optimizer, scheduler
    from lerobot.common.utils.train_utils import load_training_state

    pretrained, _ = common.resolve_pretrained_model_path(args.resume_checkpoint)
    checkpoint_dir = pretrained.parent
    return load_training_state(checkpoint_dir, optimizer, scheduler)


def _write_metric(stream: Any, *, step: int, metrics: dict[str, float]) -> None:
    payload = {"step": step, **metrics}
    stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def _resolved_arguments(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value.expanduser().resolve())
        elif isinstance(value, tuple | list):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = value
    return result


def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    import torch
    from torch.amp import GradScaler

    _validate_args(args)
    if not args.repo_id:
        raise ValueError("--repo-id or OPENPI_ACT_REPO_ID is required")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA ACT training was requested but torch.cuda.is_available() is false; "
            "select --device cpu only for lightweight smoke tests"
        )
    _set_seed(args.seed)
    expected_camera_keys = args.expected_camera_keys
    if expected_camera_keys is None:
        expected_camera_keys = config.get("dataset", {}).get("expected_camera_keys")
    location = common.resolve_dataset_location(args.dataset_root, args.repo_id)
    contract, audit_report = audit_dataset.audit_dataset(
        location,
        action_semantics=args.action_semantics,
        expected_camera_keys=(
            None if expected_camera_keys is None else tuple(str(item) for item in expected_camera_keys)
        ),
        scan_batch_size=args.audit_scan_batch_size,
        frame0_sample_episodes=args.audit_frame0_sample_episodes,
    )
    if not audit_report.get("numeric_gate_passed"):
        raise RuntimeError("ACT dataset numeric gate failed")
    print("ACT_SCHEMA_GATE_OK")
    output_dir = _prepare_output_dir(args.output_dir, resume=bool(args.resume_checkpoint))
    if args.resume_checkpoint:
        _validate_resume_checkpoint(args.resume_checkpoint, contract)
    common.write_contract_and_safety(output_dir, contract)
    common.write_json(
        output_dir / "resolved_run_config.json",
        {
            "arguments": _resolved_arguments(args),
            "config_file": config["_path"],
            "repo_id": contract.repo_id,
            "metadata_fingerprint": contract.metadata_fingerprint,
        },
    )
    common.write_json(output_dir / "dataset_audit.json", audit_report)
    common.write_json(
        output_dir / "normalization_stats_source.json",
        {
            "repo_id": contract.repo_id,
            "metadata_fingerprint": contract.metadata_fingerprint,
            "source": str(location.dataset_path / "meta" / "stats.json"),
            "policy": "LeRobot dataset statistics embedded in every ACT checkpoint",
        },
    )
    split = _prepare_split(args, contract, output_dir)
    if args.mode == "train" and not args.allow_train_without_overfit_gate:
        _require_overfit_gate(args.overfit_gate_report, contract, split)

    train_episodes = split.train_episodes
    if args.mode == "overfit":
        if len(train_episodes) < args.overfit_episodes:
            raise ValueError(
                f"overfit gate requires {args.overfit_episodes} training episodes, only "
                f"{len(train_episodes)} are available after the validation split"
            )
        rng = np.random.default_rng(args.seed)
        train_episodes = tuple(
            sorted(int(item) for item in rng.choice(train_episodes, args.overfit_episodes, replace=False))
        )

    train_config, dataset, policy, optimizer, scheduler, _ = _create_lerobot_pipeline(
        args,
        location=location,
        episodes=train_episodes,
        output_dir=output_dir,
    )
    print(f"act_dataset_repo_id: {contract.repo_id}")
    print(f"act_dataset_path: {contract.dataset_path}")
    print(f"act_dataset_total_episodes: {contract.num_episodes}")
    print(f"act_dataset_total_frames: {contract.num_frames}")
    print(f"act_training_episode_count: {len(train_episodes)}")
    print(f"act_camera_keys: {contract.camera_keys}")
    print(f"act_fps: {contract.fps:g}")
    print(f"act_device: {args.device}")
    print(f"act_cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"act_deployment_scope: {common.DEPLOYMENT_SCOPE}")

    dataloader = _make_dataloader(dataset, args, shuffle=True)
    iterator = iter(dataloader)
    first_batch, iterator = _next_batch(iterator, dataloader)
    loss, _ = _forward_gate(policy, first_batch, device=args.device)
    if args.mode == "forward":
        return 0
    _backward_gate(policy, loss)
    optimizer.zero_grad(set_to_none=True)
    if args.mode == "backward":
        return 0

    step, optimizer, scheduler = _load_resume_state(args, optimizer, scheduler)
    grad_scaler = GradScaler(torch.device(args.device).type, enabled=args.use_amp)
    overfit_initial_loss = None
    overfit_eval_loader = None
    if args.mode == "overfit":
        overfit_eval_loader = _make_dataloader(dataset, args, shuffle=False)
        overfit_initial_loss = _evaluate_loss(
            policy,
            overfit_eval_loader,
            device=args.device,
            maximum_batches=args.overfit_evaluation_batches,
            seed=args.seed,
        )
        print(f"act_overfit_initial_loss: {overfit_initial_loss:.8f}")

    total_steps = 1 if args.mode == "one-step" else (args.overfit_steps if args.mode == "overfit" else args.steps)
    train_config.steps = total_steps
    train_config.save_freq = min(args.save_frequency, total_steps)
    metrics_path = output_dir / "train_metrics.jsonl"
    with metrics_path.open("a" if args.resume_checkpoint else "w", encoding="utf-8", newline="\n") as metrics:
        while step < total_steps:
            iterator, values = _optimizer_step(
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_scaler=grad_scaler,
                dataloader=dataloader,
                iterator=iterator,
                args=args,
            )
            step += 1
            _write_metric(metrics, step=step, metrics=values)
            if step == 1:
                print(
                    f"act_one_step: loss={values['loss']:.8f}:grad_norm={values['grad_norm']:.8f}:"
                    f"lr={values['learning_rate']:.3e}"
                )
                print("ACT_ONE_STEP_GATE_OK")
            if step % args.log_frequency == 0 or step == total_steps:
                print(
                    f"act_train_step: step={step}:loss={values['loss']:.8f}:"
                    f"grad_norm={values['grad_norm']:.8f}:lr={values['learning_rate']:.3e}:"
                    f"update_s={values['update_seconds']:.3f}",
                    flush=True,
                )
            if step % args.save_frequency == 0 or step == total_steps:
                checkpoint_dir = _save_checkpoint(
                    train_config=train_config,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    contract=contract,
                    split=split,
                    step=step,
                    total_steps=total_steps,
                )
                print(f"act_checkpoint_saved: {checkpoint_dir}", flush=True)

    if args.mode == "overfit":
        final_loss = _evaluate_loss(
            policy,
            overfit_eval_loader,
            device=args.device,
            maximum_batches=args.overfit_evaluation_batches,
            seed=args.seed,
        )
        ratio = final_loss / overfit_initial_loss
        passed = bool(math.isfinite(ratio) and ratio <= args.overfit_maximum_loss_ratio)
        report = {
            "passed": passed,
            "repo_id": contract.repo_id,
            "metadata_fingerprint": contract.metadata_fingerprint,
            "episode_count": len(train_episodes),
            "episodes": train_episodes,
            "validation_episodes": split.validation_episodes,
            "steps": total_steps,
            "evaluation_batches": args.overfit_evaluation_batches,
            "initial_loss": overfit_initial_loss,
            "final_loss": final_loss,
            "final_to_initial_loss_ratio": ratio,
            "maximum_allowed_ratio": args.overfit_maximum_loss_ratio,
        }
        common.write_json(output_dir / common.OVERFIT_GATE_FILENAME, report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        if not passed:
            raise RuntimeError(
                f"10-episode overfit gate failed: final/initial loss ratio {ratio:.4f} exceeds "
                f"{args.overfit_maximum_loss_ratio:.4f}"
            )
        print("ACT_OVERFIT_GATE_OK")
    elif args.mode == "train":
        print("ACT_TRAIN_OK")
    return 0


def main() -> int:
    config = common.load_config(_config_preparse())
    args = _build_parser(config).parse_args()
    try:
        return run(args, config)
    except Exception as exc:
        print(f"ACT_TRAIN_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
