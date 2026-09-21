"""
Lab 3 - Quantization: Dynamic PTQ, Static PTQ, and ONNX

Course: Big Data at the Edge - Week 3 - Graded: 10%

Goal
----
Following the floating-point-representation lecture, this lab applies
quantization - reducing the numerical precision used to store and compute
with weights/activations (typically FP32 -> INT8) - to the same CIFAR-10
MobileNetV2 you worked with in Lab 2.

You will implement, measure, and compare:

  1. Dynamic post-training quantization (PTQ) - weights quantized ahead
     of time, activations quantized on the fly at inference. No
     calibration data required.
  2. Static post-training quantization (PTQ) - both weights *and*
     activations quantized ahead of time, using a small calibration set
     to estimate activation ranges. Usually a bigger win than dynamic PTQ
     for convolutional networks, but needs representative data and a
     fuse step.
  3. ONNX conversion of a quantized model - quantizing the *ONNX graph
     itself* with onnxruntime.quantization, which is the more portable
     alternative to exporting an already-quantized PyTorch model.

At each stage, measure model size, accuracy, and CPU latency (int8
quantized kernels in PyTorch are CPU-only - GPUs are out of scope for the
quantized models themselves, though a GPU still speeds up training the
FP32 baseline in Section 1).

This is a plain Python script version of `lab3_quantization.ipynb`, for
running as a SLURM batch job on DAS-5 (see `submit_lab3_quantization.sbatch`)
- Jupyter notebooks can't be submitted as SLURM batch jobs directly. If
you're working locally/interactively instead, `lab3_quantization.ipynb`
covers the exact same material and may be more convenient to iterate in.

Fill in every `# TODO` below, then either run this directly:
    python lab3_quantization.py
or submit it as a batch job:
    mkdir -p logs
    sbatch submit_lab3_quantization.sbatch

Deliverables (see Lab-3.pdf for the full grading rubric)
----------------------------------------------------------
- This completed script (or the completed notebook).
- A table comparing FP32 baseline vs. dynamic PTQ vs. static PTQ across
  size / accuracy / latency.
- A quantized ONNX model (via the ONNX Runtime quantization route),
  verified against the FP32 ONNX baseline.
- Written answers to the reflection questions (in a separate report -
  see the "Reflection" comments throughout this script for the actual
  questions).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from utils import (
    benchmark_latency,
    build_model,
    calibrate,
    clone_model,
    dynamic_quantize,
    evaluate,
    fit,
    get_dataloaders,
    get_device,
    load_checkpoint,
    model_size_mb,
    save_checkpoint,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lab 3 - Quantization assignment (script version, for SLURM batch jobs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="./data", help="Where CIFAR-10 is downloaded/cached.")
    parser.add_argument("--output-dir", default="./outputs", help="Where to write checkpoints, the ONNX exports, and the summary table.")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Optional path to a pretrained checkpoint to load instead of training the FP32 baseline "
             "from scratch - e.g. a fine-tuned checkpoint from Lab 2 (utils.load_checkpoint). "
             "Check with your instructor whether this is expected for your section; the default "
             "(no checkpoint) trains a fresh baseline, matching the notebook's Section 1.",
    )
    parser.add_argument(
        "--train-subset-fraction", type=float, default=0.4,
        help="Fraction of the training set to use when training the baseline from scratch "
             "(ignored if --checkpoint is given). Use 1.0 for the full training set.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--baseline-epochs", type=int, default=3,
        help="Epochs for training the FP32 baseline (Section 1), reusing the same approach as "
             "Lab 2, Section 1. Ignored if --checkpoint is given.",
    )
    parser.add_argument(
        "--calibration-batches", type=int, default=20,
        help="Number of training batches to run through the prepared model for static PTQ "
             "calibration (Section 3), passed to utils.calibrate.",
    )
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset for the FP32 export (Section 5).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", default=None, help="Optional path to additionally write logs to a file.")
    return parser


def setup_logging(log_file: str | None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("lab3_quantization")


def main() -> None:
    args = build_arg_parser().parse_args()
    log = setup_logging(args.log_file)

    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # ======================================================================
    # 0. Setup
    # ======================================================================
    # Same dual-device setup as the notebook: the GPU (if available) is
    # used only to train the FP32 baseline faster in Section 1. Every
    # quantized model from Section 2 onward is measured on CPU, because
    # PyTorch's standard int8 quantized kernels do not run on CUDA.
    log.info("=== Section 0: Setup ===")
    gpu_device = get_device()
    cpu_device = torch.device("cpu")
    log.info(f"Training device: {gpu_device} | Quantized inference device: {cpu_device}")

    # ======================================================================
    # 1. Baseline model (FP32)
    # ======================================================================
    # Train the same CIFAR-10-adapted MobileNetV2 as in Lab 2 (or,
    # else load the fine-tuned checkpoint you already
    # trained in Lab 2 via --checkpoint). Either way, record baseline
    # accuracy, size, and CPU latency - quantized models must be compared
    # on CPU, so the baseline should be too, for a fair comparison.
    log.info("=== Section 1: Baseline model (FP32) ===")

    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_subset_fraction=args.train_subset_fraction,
    )

    fp32_model = build_model()

    if args.checkpoint:
        log.info(f"Loading checkpoint from {args.checkpoint} (skipping training)")
        fp32_model = load_checkpoint(fp32_model, args.checkpoint)
    else:
        # TODO: train fp32_model on gpu_device for args.baseline_epochs
        # epochs (utils.fit), reusing the same approach as Lab 2, Section 1.
        pass  # TODO

    # TODO: move fp32_model to cpu_device and eval() mode, then measure and
    # print accuracy (utils.evaluate), model size in MB (utils.model_size_mb),
    # and CPU latency in ms (utils.benchmark_latency). Store the results in
    # fp32_acc, fp32_size, fp32_latency - these three names are used by the
    # summary table in Section 4, so keep them as-is.
    fp32_acc = None      # TODO
    fp32_size = None     # TODO
    fp32_latency = None  # TODO

    save_checkpoint(fp32_model, output_dir / "checkpoints" / "fp32_baseline.pt")
    log.info(
        f"FP32 baseline - accuracy: {fp32_acc}, size_mb: {fp32_size}, "
        f"latency_ms: {fp32_latency}"
    )

    # ======================================================================
    # 2. Dynamic post-training quantization
    # ======================================================================
    # Apply utils.dynamic_quantize (torch.ao.quantization.quantize_dynamic),
    # which quantizes the weights of the given layer types ahead of time and
    # quantizes activations dynamically at inference. No calibration data
    # is needed.
    #
    # Important limitation to notice and discuss: MobileNetV2 is almost
    # entirely Conv2d layers; dynamic quantization in PyTorch only supports
    # a limited set of layer types (chiefly nn.Linear and RNN layers), so
    # it will only quantize the final classifier Linear layer here. Expect
    # a *small* effect on size/latency - this is expected, not a bug.
    log.info("=== Section 2: Dynamic PTQ ===")

    # TODO: clone the FP32 baseline (utils.clone_model), move to
    # cpu_device, apply utils.dynamic_quantize(...), and measure accuracy,
    # size, and CPU latency exactly as in Section 1. Store the results in
    # dynamic_acc, dynamic_size, dynamic_latency (used in Section 4).
    dynamic_model = None    # TODO
    dynamic_acc = None      # TODO
    dynamic_size = None     # TODO
    dynamic_latency = None  # TODO

    log.info(
        f"Dynamic PTQ - accuracy: {dynamic_acc}, size_mb: {dynamic_size}, "
        f"latency_ms: {dynamic_latency}"
    )

    # Reflection (answer in your written report, not here): How much (if
    # at all) did size/latency change vs. the FP32 baseline? Given what
    # you know about MobileNetV2's layer composition, was this expected?
    # For which architectures would you expect dynamic PTQ to matter much
    # more?

    # ======================================================================
    # 3. Static post-training quantization (FX Graph Mode)
    # ======================================================================
    # Static PTQ quantizes both weights and activations ahead of time,
    # which requires observing typical activation ranges on a small
    # calibration set first. We use PyTorch's FX Graph Mode quantization
    # (torch.ao.quantization.quantize_fx), which automatically fuses
    # Conv+BatchNorm+ReLU patterns for you - doing this fusion by hand for
    # a nested architecture like MobileNetV2's inverted residual blocks is
    # tedious and error-prone, which is exactly why FX mode exists.
    #
    # Steps: (1) set the quantization backend/engine, (2) build a
    # QConfigMapping with a default int8 qconfig, (3) prepare_fx (inserts
    # observers), (4) calibrate by running some batches through in eval
    # mode, (5) convert_fx (produces the quantized model).
    log.info("=== Section 3: Static PTQ (FX graph mode) ===")

    # TODO: import get_default_qconfig_mapping, prepare_fx, convert_fx
    # from torch.ao.quantization (see the solution notebook if you get
    # stuck on the exact import paths).

    # TODO: set torch.backends.quantized.engine = "fbgemm" (targets x86
    # CPUs; use "qnnpack" instead if you were targeting an ARM edge
    # device).

    # TODO: clone the FP32 baseline to cpu_device/eval() as `static_model`.

    # TODO: build a qconfig_mapping (get_default_qconfig_mapping()) and
    # example_inputs (a single batch of images from test_loader, on
    # cpu_device), then call
    # prepare_fx(static_model, qconfig_mapping, example_inputs).

    # TODO: calibrate the prepared model on a handful of training batches
    # using utils.calibrate(prepared_model, train_loader, cpu_device,
    # n_batches=args.calibration_batches).

    # TODO: call convert_fx(...) to produce `static_quantized`, then
    # measure and print accuracy, size, and CPU latency exactly as in
    # Sections 1-2. Store the results in static_acc, static_size,
    # static_latency (used in Section 4).
    static_quantized = None  # TODO
    static_acc = None        # TODO
    static_size = None       # TODO
    static_latency = None    # TODO

    log.info(
        f"Static PTQ (FX) - accuracy: {static_acc}, size_mb: {static_size}, "
        f"latency_ms: {static_latency}"
    )

    # Reflection: Compare static PTQ against both the FP32 baseline and
    # dynamic PTQ on all three metrics. Which trade-off looks best for an
    # edge deployment where storage/bandwidth is the binding constraint?
    # Which would you pick if inference latency were the binding
    # constraint instead? Did accuracy degrade - and if so, would
    # more/better calibration data help?

    # ======================================================================
    # 4. Summary table
    # ======================================================================
    log.info("=== Section 4: Summary table ===")

    summary = pd.DataFrame([
        {"method": "FP32 baseline", "accuracy": fp32_acc, "size_mb": fp32_size, "latency_ms": fp32_latency},
        {"method": "Dynamic PTQ", "accuracy": dynamic_acc, "size_mb": dynamic_size, "latency_ms": dynamic_latency},
        {"method": "Static PTQ (FX)", "accuracy": static_acc, "size_mb": static_size, "latency_ms": static_latency},
    ])

    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved summary table to {summary_path}")
    log.info("\n" + summary.to_string(index=False))

    # ======================================================================
    # 5. ONNX conversion
    # ======================================================================
    # There are two ways to end up with a quantized ONNX model:
    #
    # Option A - export a PyTorch-quantized model directly. Possible in
    # principle, but int8 quantized PyTorch ops don't always map cleanly
    # onto standard ONNX operators/opsets, and support has historically
    # been inconsistent across PyTorch versions. This is exploratory only
    # - it is not graded, so it is intentionally left out of this batch
    # script; feel free to try it interactively in the notebook, but
    # don't spend more than a few minutes on it if it fails.
    #
    # Option B (recommended, and what's implemented below) - quantize the
    # ONNX graph directly using onnxruntime.quantization. Export the
    # *original FP32* model to ONNX (exactly like Lab 1), then run ONNX
    # Runtime's own post-training quantization on that graph. This is the
    # more portable path in practice: the resulting .onnx file quantizes
    # and runs correctly under ONNX Runtime regardless of which framework
    # produced the original graph.
    log.info("=== Section 5: ONNX conversion ===")

    fp32_onnx_path = output_dir / "mobilenet_v2_fp32.onnx"
    int8_onnx_path = output_dir / "mobilenet_v2_int8.onnx"

    # TODO: export `fp32_model` to fp32_onnx_path (reuse the Lab 1
    # pattern: torch.onnx.export with input_names/output_names/
    # dynamic_axes/opset_version=args.opset_version).

    # TODO: import quantize_dynamic and QuantType from
    # onnxruntime.quantization, and quantize fp32_onnx_path into
    # int8_onnx_path (weight_type=QuantType.QInt8).

    # TODO: print the file sizes of both ONNX files (FP32 vs INT8) and
    # compare against your PyTorch static PTQ size from Section 3.

    # TODO: run both ONNX models on the same dummy input with
    # onnxruntime.InferenceSession and print the max absolute difference
    # between their outputs.

    # Reflection (this is the main graded discussion - answer thoroughly
    # in your written report):
    # 1. Compare the ONNX-graph quantization file size against your
    #    PyTorch static PTQ size from Section 3. Are they similar? If
    #    not, what could explain a difference (hint: think about what
    #    exactly gets quantized - weight-only dynamic quantization of the
    #    ONNX graph vs. weights *and* activations in PyTorch static PTQ).
    # 2. Why might quantizing the ONNX graph directly (Option B) be a
    #    more robust choice than exporting an already-quantized PyTorch
    #    model (Option A) when your deployment target is a heterogeneous
    #    fleet of edge devices, each possibly running a different
    #    inference runtime?
    # 3. Tie this back to floating point representation from lecture: in
    #    your own words, what does going from FP32 to INT8 actually
    #    change about how a single weight value is stored and multiplied
    #    during inference, and why does that reduce both memory and, on
    #    suitable hardware, latency?

    log.info("Done.")


if __name__ == "__main__":
    main()
