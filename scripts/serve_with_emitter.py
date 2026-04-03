#!/usr/bin/env python3
"""Launch vLLM OpenAI-compatible server with hidden states emission.

Starts a standard vLLM serving endpoint that also captures hidden states
during prefill and writes them as .pt training data files.

Usage:
    python scripts/serve_with_emitter.py \
        --model /data/models/MiniMax-M2.5 \
        --layer-ids 2 15 29 31 \
        --output-dir /data/serving_data \
        --max-buffer-gb 50 \
        --tensor-parallel-size 4 \
        --port 8000

The emitter runs as a background thread within the vLLM worker process.
It produces .pt files compatible with train_streaming.py.
"""

import argparse
import asyncio
import logging
import signal
import sys

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM server with hidden states emission"
    )

    # Emitter arguments
    emitter = parser.add_argument_group("Emitter")
    emitter.add_argument(
        "--layer-ids",
        type=int,
        nargs="+",
        required=True,
        help="Layer indices to capture hidden states from",
    )
    emitter.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write .pt training data files",
    )
    emitter.add_argument(
        "--max-buffer-gb",
        type=float,
        default=100.0,
        help="Maximum buffer size in GB (default: 100)",
    )
    emitter.add_argument(
        "--min-seq-len",
        type=int,
        default=64,
        help="Minimum sequence length to capture (default: 64)",
    )
    emitter.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Fraction of requests to capture (default: 1.0)",
    )

    # vLLM server arguments
    server = parser.add_argument_group("Server")
    server.add_argument(
        "--model", type=str, required=True, help="Model name or path"
    )
    server.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path (default: same as model)",
    )
    server.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    server.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    server.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    server.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length",
    )
    server.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    server.add_argument(
        "--speculative-config",
        type=str,
        default=None,
        help="JSON speculative decoding config (optional)",
    )

    return parser.parse_args()


async def run_server(args):
    """Start vLLM server with serving emitter extension."""
    # Import vLLM here to allow the script to parse args without vLLM installed
    from vllm import LLM, SamplingParams  # noqa: F401
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args,
        build_app,
        init_app_state,
        serve_http,
    )

    tokenizer_name = args.tokenizer or args.model

    # Build engine args with our worker extension
    engine_args = AsyncEngineArgs(
        model=args.model,
        tokenizer=tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        worker_extension_cls=(
            "speculators.serving.worker_extension"
            ".ServingHiddenStatesExtension"
        ),
        enforce_eager=True,  # required for forward patching
    )

    logger.info("Starting vLLM engine with serving emitter extension...")

    async with build_async_engine_client_from_engine_args(
        engine_args, disable_frontend_multiprocessing=True
    ) as engine_client:
        # Initialize the serving capture on all workers
        logger.info(
            "Initializing serving capture: layers=%s, output=%s, "
            "buffer=%.1fGB, sample_rate=%.2f",
            args.layer_ids,
            args.output_dir,
            args.max_buffer_gb,
            args.sample_rate,
        )
        await engine_client.collective_rpc(
            "_setup_serving_capture",
            args=(
                args.layer_ids,
                args.output_dir,
                tokenizer_name,
                args.max_buffer_gb,
                args.min_seq_len,
                args.sample_rate,
            ),
        )

        # Build the OpenAI-compatible API app
        # Note: The exact API may vary by vLLM version. This targets vLLM 0.12-0.17.
        app = build_app(args)
        await init_app_state(engine_client, app.state, args)

        # Set up graceful shutdown
        shutdown_event = asyncio.Event()

        def _signal_handler():
            logger.info("Shutdown signal received")
            shutdown_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

        # Start serving
        shutdown_task = await serve_http(
            app, host=args.host, port=args.port
        )

        # Wait for shutdown
        await shutdown_task

        # Clean shutdown of emitter
        logger.info("Shutting down serving capture...")
        await engine_client.collective_rpc("_shutdown_serving_capture")

    logger.info("Server stopped.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()

    logger.info("=" * 60)
    logger.info("vLLM Server with Hidden States Emission")
    logger.info("  Model:       %s", args.model)
    logger.info("  Layers:      %s", args.layer_ids)
    logger.info("  Output:      %s", args.output_dir)
    logger.info("  Buffer:      %.1f GB", args.max_buffer_gb)
    logger.info("  Sample Rate: %.2f", args.sample_rate)
    logger.info("  TP Size:     %d", args.tensor_parallel_size)
    logger.info("  Port:        %d", args.port)
    logger.info("=" * 60)

    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
