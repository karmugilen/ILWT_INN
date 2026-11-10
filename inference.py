#!/usr/bin/env python3
"""
Video Steganography Inference Pipeline

Converts .y4m files to MP4, runs the full steganography embedding/recovery pipeline,
generates results, and handles cleanup.

Architecture: Imperative Shell
- This file is a thin CLI wrapper that delegates to:
  - core/config.py: Configuration dataclasses
  - core/metrics.py: Metric registry (add metrics with @metric decorator)
  - core/models.py: Model factory
  - shell/io.py: File I/O operations
  - shell/pipeline_runner.py: Pipeline orchestration

Features:
- Progress bar with tqdm
- Parallel async processing for multiple videos
- Easy to add new metrics (just use @metric decorator)
- Easy to change models (just update ModelConfig)
"""

import sys
import asyncio
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Import configuration from functional core
from core.config import PipelineConfig, ModelConfig, ProcessingConfig

# Import pipeline runner from imperative shell
from shell.pipeline_runner import process_single_video


def process_single_y4m_wrapper(args_tuple) -> dict:
    """
    Worker function for parallel processing.
    
    Unpacks args and delegates to shell/pipeline_runner.
    """
    y4m_path, hide_dir, config = args_tuple
    return process_single_video(y4m_path, hide_dir, config)


async def process_videos_parallel(
    y4m_files, 
    hide_dir: Path,
    config: PipelineConfig,
    max_workers: int = 2
):
    """Process multiple videos in parallel with progress bar."""
    args_list = [(y4m_file, hide_dir, config) for y4m_file in y4m_files]
    results = []
    
    loop = asyncio.get_event_loop()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(y4m_files), desc="Processing videos", unit="video",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            futures = [
                loop.run_in_executor(executor, process_single_y4m_wrapper, args)
                for args in args_list
            ]
            
            for future in asyncio.as_completed(futures):
                result = await future
                results.append(result)
                status = "✅" if result['success'] else "❌"
                pbar.set_postfix_str(f"{status} {result['file']}")
                pbar.update(1)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Video Steganography Inference: convert to MP4, run pipeline, verify recovery"
    )
    
    # File selection
    parser.add_argument(
        '--y4m', nargs='+', 
        help='Y4M file(s) to process. If not specified, processes all .y4m in current dir'
    )
    parser.add_argument(
        '--hide_dir', default='secrets', 
        help='Directory containing images to hide (default: secrets)'
    )
    
    # Model configuration (easy to change!)
    parser.add_argument(
        '--model', default='ilwt_steganography_model_research.pth', 
        help='Model path'
    )
    parser.add_argument(
        '--size', type=int, default=256, 
        help='Output video size (default: 256)'
    )
    parser.add_argument(
        '--num_blocks', type=int, default=8, 
        help='Number of StegoWaveFlow blocks (default: 8)'
    )
    parser.add_argument(
        '--hidden_channels', type=int, default=128, 
        help='Hidden channels (default: 128)'
    )
    
    # Processing configuration
    parser.add_argument(
        '--cleanup', action='store_true', 
        help='Cleanup intermediate files after processing (default: False)'
    )
    parser.add_argument(
        '--batch', action='store_true',
        help='Batch mode: disable interactive prompts and run smoothly'
    )
    parser.add_argument(
        '--workers', type=int, default=2, 
        help='Number of parallel workers (default: 2)'
    )
    parser.add_argument(
        '--sequential', action='store_true',
        help='Process videos sequentially instead of in parallel'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='GPU batch size for embedding/recovery (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Build configuration from args (Functional Core pattern)
    model_config = ModelConfig(
        path=args.model,
        num_blocks=args.num_blocks,
        hidden_channels=args.hidden_channels,
        size=args.size
    )
    
    # Invert batch flag to get interactive mode
    is_interactive = not args.batch
    
    processing_config = ProcessingConfig(
        batch_size=args.batch_size,
        workers=args.workers,
        cleanup=args.cleanup,
        sequential=args.sequential or is_interactive,
        interactive=is_interactive
    )
    
    config = PipelineConfig(
        model=model_config,
        processing=processing_config,
        hide_dir=args.hide_dir
    )
    
    # Find Y4M files
    if args.y4m:
        y4m_files = [Path(f) for f in args.y4m]
    else:
        y4m_files = list(Path('.').glob('*.y4m'))
    
    if not y4m_files:
        print("No Y4M files found to process.")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"\n{'='*70}")
    print(f"VIDEO STEGANOGRAPHY PIPELINE (Functional Core / Imperative Shell)")
    print(f"{'='*70}")
    print(f"Found {len(y4m_files)} Y4M file(s) to process:")
    for f in y4m_files:
        print(f"  • {f.name}")
    print(f"\nModel: {model_config.path}")
    print(f"  Blocks: {model_config.num_blocks}, Channels: {model_config.hidden_channels}")
    print(f"  Blocks: {model_config.num_blocks}, Channels: {model_config.hidden_channels}")
    print(f"\nMode: {'Interactive Sequential' if is_interactive else ('Sequential' if args.sequential else f'Parallel ({args.workers} workers)')}")
    
    print(f"Cleanup: {'Enabled' if processing_config.cleanup else 'Disabled'}")
    print(f"{'='*70}\n")
    
    hide_dir = Path(args.hide_dir)
    
    if args.sequential or is_interactive:
        # Sequential processing with progress bar
        results = []
        with tqdm(y4m_files, desc="Processing videos", unit="video",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for y4m_file in pbar:
                pbar.set_postfix_str(f"📹 {y4m_file.name}")
                result = process_single_video(y4m_file, hide_dir, config)
                results.append(result)
                status = "✅" if result['success'] else "❌"
                pbar.set_postfix_str(f"{status} {result['file']}")
    else:
        # Parallel async processing
        results = asyncio.run(process_videos_parallel(
            y4m_files, hide_dir, config, args.workers
        ))
    
    # Print summary
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count
    
    print(f"\n{'='*70}")
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"  ✅ Successful: {success_count}/{len(results)}")
    if failed_count > 0:
        print(f"  ❌ Failed: {failed_count}")
        for r in results:
            if not r['success']:
                print(f"     • {r['file']}: {r['error']}")
    
    # Print average metrics if available
    # Check results.txt/json in output folder if metrics were enabled
    pass
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
