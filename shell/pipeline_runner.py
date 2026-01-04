"""
Imperative Shell - Pipeline Runner

Orchestrates the full video steganography pipeline by:
1. Using pure functions from the functional core
2. Delegating I/O operations to shell/io.py
3. Managing state and side effects

This is the main entry point for processing videos.
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm import tqdm
import time

# Core imports (pure functions)
from core.config import ModelConfig, ProcessingConfig, PipelineConfig
from core.models import create_model


# Shell imports (side effects)
from shell.io import (
    convert_y4m_to_mp4, extract_frames, frames_to_video, get_video_fps,
    cleanup_output, get_file_size, load_model_weights,
    load_image, load_16bit_image, save_16bit_png, list_images_in_dir,
    save_per_frame_metrics_csv, save_average_metrics_csv, save_results_txt,
    save_results_json
)

# Core imports for model info
from core.models import get_model_info


# ============================================================================
# Pipeline Core Operations
# ============================================================================

async def embed_all_frames(
    model,
    cover_frames: List[Path],
    hide_files: List[Path],
    stego_frames_dir: Path,
    size: int,
    device: torch.device,
    batch_size: int = 8
) -> List[torch.Tensor]:
    """
    Embed secret images into cover frames using batched GPU processing.
    
    Args:
        model: Loaded steganography model
        cover_frames: List of cover frame paths
        hide_files: List of secret image paths (cycled if shorter)
        stego_frames_dir: Output directory for stego frames
        size: Image size
        device: GPU device
        batch_size: Batch size for GPU processing
    
    Returns:
        List of original hide tensors for metric computation
    """
    stego_frames_dir.mkdir(parents=True, exist_ok=True)
    original_hide_tensors = []
    num_frames = len(cover_frames)
    loop = asyncio.get_event_loop()
    
    for batch_start in tqdm(range(0, num_frames, batch_size), desc="Embedding", 
                            total=(num_frames + batch_size - 1) // batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        batch_indices = range(batch_start, batch_end)
        
        # Load pairs concurrently
        async def load_pair(i):
            def _load():
                cover = load_image(cover_frames[i], size)
                hide = load_image(hide_files[i % len(hide_files)], size)
                return cover, hide
            return await loop.run_in_executor(None, _load)
        
        pairs = await asyncio.gather(*[load_pair(i) for i in batch_indices])
        
        # Batch GPU inference
        cover_batch = torch.stack([p[0] for p in pairs]).to(device)
        hide_batch = torch.stack([p[1] for p in pairs]).to(device)
        input_batch = torch.cat([cover_batch, hide_batch], dim=1)
        
        with torch.no_grad():
            stego_output, _ = model(input_batch)
        stego_hosts = stego_output[:, :3, :, :]
        
        # Save outputs concurrently
        async def save_output(idx, tensor, hide_tensor):
            frame_idx = batch_start + idx
            output_path = stego_frames_dir / f"frame_{frame_idx+1:05d}.png"
            def _save():
                save_16bit_png(tensor, output_path)
            await loop.run_in_executor(None, _save)
            return hide_tensor.cpu()
        
        hide_tensors = await asyncio.gather(*[
            save_output(idx, stego_hosts[idx], hide_batch[idx].unsqueeze(0)) 
            for idx in range(len(batch_indices))
        ])
        original_hide_tensors.extend(hide_tensors)
    
    return original_hide_tensors


async def recover_all_frames(
    model,
    stego_frames: List[Path],
    recovered_dir: Path,
    size: int,
    device: torch.device,
    batch_size: int = 8
) -> None:
    """
    Recover hidden images from stego frames using batched GPU processing.
    """
    recovered_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(stego_frames)
    loop = asyncio.get_event_loop()
    
    for batch_start in tqdm(range(0, num_frames, batch_size), desc="Recovering",
                            total=(num_frames + batch_size - 1) // batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        batch_indices = range(batch_start, batch_end)
        
        # Load stego frames concurrently
        async def load_stego(i):
            def _load():
                return load_16bit_image(stego_frames[i], size)
            return await loop.run_in_executor(None, _load)
        
        stego_tensors = await asyncio.gather(*[load_stego(i) for i in batch_indices])
        
        # Batch GPU inference
        stego_batch = torch.stack(stego_tensors).to(device).float()
        zeros_batch = torch.zeros_like(stego_batch)
        stego_like = torch.cat([stego_batch, zeros_batch], dim=1)
        
        with torch.no_grad():
            reconstructed = model.inverse(stego_like)
        recovered_secrets = reconstructed[:, 3:, :, :]
        
        # Save outputs concurrently
        async def save_recovered(idx, tensor):
            frame_idx = batch_start + idx
            output_path = recovered_dir / f"recovered_{frame_idx+1:05d}.png"
            def _save():
                save_16bit_png(tensor, output_path)
            await loop.run_in_executor(None, _save)
        
        await asyncio.gather(*[
            save_recovered(idx, recovered_secrets[idx]) 
            for idx in range(len(batch_indices))
        ])


    return []



# ============================================================================
# Main Pipeline Runner
# ============================================================================

async def run_pipeline(
    video_path: Path,
    hide_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    device: torch.device
) -> Dict[str, Any]:
    """
    Run the complete video steganography pipeline.
    
    This is the main orchestrating function that:
    1. Creates model from config (core)
    2. Loads weights (shell/io)
    3. Extracts frames (shell/io)
    4. Embeds secrets (batched GPU)
    5. Creates stego video (shell/io)
    6. Recovers secrets (batched GPU)
    7. Computes metrics (core)
    8. Cleans up (shell/io)
    
    Args:
        video_path: Input video file
        hide_dir: Directory containing secret images
        output_dir: Output directory for results
        config: Complete pipeline configuration
        device: GPU device
    
    Returns:
        Dict with success status, metrics, and any errors
    """
    result = {
        'success': False,
        'error': None,
        'metrics': None,
        'output_dir': str(output_dir)
    }
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create and load model (core + shell)
        model = create_model(config.model)
        load_model_weights(model, Path(config.model.path), device)
        model = model.to(device)
        model.eval()
        
        # Get hide images
        hide_files = list_images_in_dir(hide_dir, '*.png')
        if not hide_files:
            hide_files = list_images_in_dir(hide_dir, '*.jpg')
        if not hide_files:
            result['error'] = f"No images found in {hide_dir}"
            return result
        
        # Extract frames from video (shell)
        cover_frames_dir = output_dir / "cover_frames"
        
        if config.processing.interactive:
             print(f"\nNext: Extract frames from {video_path}")
             input("Press Enter to continue...")
             
        cover_frames = extract_frames(video_path, cover_frames_dir, config.model.size)
        
        fps = get_video_fps(video_path)
        
        # Embed secrets (batched GPU) - TIMED for research
        stego_frames_dir = output_dir / "stego_frames"
        
        if config.processing.interactive:
             print(f"\nNext: Embed {len(hide_files)} secret images into {len(cover_frames)} frames")
             input("Press Enter to continue...")

        embed_start_time = time.perf_counter()
        original_hide_tensors = await embed_all_frames(
            model, cover_frames, hide_files, stego_frames_dir,
            config.model.size, device, config.processing.batch_size
        )
        embed_end_time = time.perf_counter()
        embed_time = embed_end_time - embed_start_time
        
        # Create stego video (shell)
        stego_video = output_dir / "stego_video.mp4"
        
        if config.processing.interactive:
             print(f"\nNext: Create stego video at {stego_video}")
             input("Press Enter to continue...")
             
        frames_to_video(stego_frames_dir, stego_video, fps)
        
        # Extract frames from stego video for recovery
        extracted_dir = output_dir / "extracted_frames"
        extracted_frames = extract_frames(stego_video, extracted_dir, config.model.size)
        
        # Recover secrets (batched GPU) - TIMED for research
        recovered_dir = output_dir / "recovered"
        
        if config.processing.interactive:
             print(f"\nNext: Recover secrets from stego video")
             input("Press Enter to continue...")
             
        extract_start_time = time.perf_counter()
        await recover_all_frames(
            model, extracted_frames, recovered_dir,
            config.model.size, device, config.processing.batch_size
        )
        extract_end_time = time.perf_counter()
        extract_time = extract_end_time - extract_start_time
        
        # Total pipeline time (embedding + extraction, excluding metrics)
        total_pipeline_time = embed_time + extract_time
        num_frames = len(cover_frames)
        
        # Timing info for results
        timing_info = {
            'embed_time': embed_time,
            'extract_time': extract_time,
            'total_pipeline_time': total_pipeline_time,
            'num_frames': num_frames,
            'fps_embed': num_frames / embed_time if embed_time > 0 else 0,
            'fps_extract': num_frames / extract_time if extract_time > 0 else 0,
        }
        
        # Metrics computation removed as per request (Code completely stripped)
        # Just save results.txt with timing and file sizes
        original_video_size = get_file_size(video_path)
        stego_video_size = get_file_size(stego_video)
        
        save_results_txt(
            {},  # Empty metrics
            video_path.name,
            get_model_info(config.model),
            output_dir / 'results.txt',
            file_sizes={
                'original_video': original_video_size,
                'stego_video': stego_video_size
            },
            timing_info=timing_info
        )
        
        # Cleanup (shell)
        if config.processing.cleanup:
            cleanup_output(output_dir, config.processing.keep_patterns)
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


def process_single_video(
    y4m_path: Path,
    hide_dir: Path,
    config: PipelineConfig,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Process a single Y4M file through the complete pipeline.
    
    This is a synchronous wrapper around run_pipeline for use with
    ProcessPoolExecutor.
    
    Args:
        y4m_path: Path to Y4M file
        hide_dir: Directory with secret images
        config: Pipeline configuration
        device: GPU device (uses CUDA if available)
    
    Returns:
        Result dict with success status and metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    result = {'file': y4m_path.name, 'success': False, 'error': None}
    
    # Create paths
    mp4_path = y4m_path.parent / f"{y4m_path.stem}_{config.model.size}x{config.model.size}.mp4"
    output_dir = y4m_path.parent / f"{y4m_path.stem}_output"
    
    # Convert Y4M to MP4
    if config.processing.interactive:
        print(f"\nNext: Convert {y4m_path} to MP4")
        input("Press Enter to continue...")
        
    success, error = convert_y4m_to_mp4(y4m_path, mp4_path, config.model.size)
    if not success:
        result['error'] = f"Conversion failed: {error}"
        return result
    
    # Run pipeline
    pipeline_result = asyncio.run(run_pipeline(
        mp4_path, hide_dir, output_dir, config, device
    ))
    
    if not pipeline_result['success']:
        result['error'] = f"Pipeline failed: {pipeline_result.get('error', 'Unknown error')}"
        return result
    
    result['success'] = True
    result['metrics'] = pipeline_result.get('metrics')
    result['output'] = str(output_dir)
    
    # Cleanup converted MP4 and original Y4M if configured
    if config.processing.cleanup:
        if mp4_path.exists():
            mp4_path.unlink()
        if y4m_path.exists():
            y4m_path.unlink()
    
    return result
