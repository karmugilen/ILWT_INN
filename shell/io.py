"""
Imperative Shell - I/O Operations

All file system and subprocess operations are isolated here.
This keeps side effects contained and makes the core easily testable.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional, List
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


# ============================================================================
# FFmpeg Operations (Side Effects)
# ============================================================================

def convert_y4m_to_mp4(y4m_path: Path, output_mp4: Path, size: int = 256) -> Tuple[bool, Optional[str]]:
    """
    Convert Y4M file to MP4 at specified resolution.
    
    Side effect: Runs ffmpeg subprocess, writes file.
    """
    cmd = [
        'ffmpeg', '-y', '-i', str(y4m_path),
        '-vf', f'scale={size}:{size}:flags=lanczos',
        '-c:v', 'png', '-pix_fmt', 'rgb48be',
        str(output_mp4)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr
    return True, None


def extract_frames(video_path: Path, output_dir: Path, size: int = 256) -> List[Path]:
    """
    Extract frames from video to 16-bit PNG.
    
    Side effect: Runs ffmpeg, creates directory, writes files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -y -i "{video_path}" -vf "scale={size}:{size}:flags=lanczos" -pix_fmt rgb48be "{output_dir}/frame_%05d.png"'
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    return sorted(output_dir.glob("frame_*.png"))


def frames_to_video(frames_dir: Path, output_video: Path, fps: float = 30) -> None:
    """
    Combine frames to lossless MP4 video.
    
    Side effect: Runs ffmpeg, writes video file.
    """
    cmd = f'ffmpeg -y -framerate {fps} -i "{frames_dir}/frame_%05d.png" -c:v png -pix_fmt rgb48be "{output_video}"'
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def get_video_fps(video_path: Path) -> float:
    """
    Get video framerate using ffprobe.
    
    Side effect: Runs ffprobe subprocess.
    """
    import json
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of json "{video_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data.get('streams', [{}])[0]
    fps_str = stream.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den) if float(den) != 0 else 30
    else:
        fps = float(fps_str)
    return fps


# ============================================================================
# File Operations (Side Effects)
# ============================================================================

def cleanup_output(output_dir: Path, keep_patterns: Tuple[str, ...]) -> None:
    """
    Delete files in directory except those matching keep patterns.
    
    Side effect: Deletes files and directories.
    """
    if not output_dir.exists():
        return
    
    items_to_delete = []
    for item in output_dir.iterdir():
        if item.name not in keep_patterns:
            items_to_delete.append(item)
    
    for item in items_to_delete:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def get_file_size(path: Path) -> int:
    """
    Get file size in bytes.
    
    Side effect: File system read.
    """
    return path.stat().st_size if path.exists() else 0


# ============================================================================
# Model I/O (Side Effects)
# ============================================================================

def load_model_weights(model, path: Path, device: torch.device) -> None:
    """
    Load model weights from checkpoint file.
    
    Handles both formats:
    - Direct state_dict
    - Wrapped with 'model_state_dict' key
    
    Side effect: Reads file, modifies model state.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle both checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's the state_dict directly
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()


# ============================================================================
# Image I/O (Side Effects)
# ============================================================================

def load_image(path: Path, size: int) -> torch.Tensor:
    """
    Load and preprocess image for model input.
    
    Side effect: Reads file from disk.
    
    Returns:
        Tensor in [-1, 1] range, shape (C, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(path).convert('RGB')
    return transform(img)


def load_16bit_image(path: Path, size: int) -> torch.Tensor:
    """
    Load 16-bit PNG image.
    
    Side effect: Reads file from disk.
    
    Returns:
        Tensor in [-1, 1] range, shape (C, H, W)
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1))
    return tensor


def save_16bit_png(tensor: torch.Tensor, path: Path) -> None:
    """
    Save tensor as 16-bit PNG.
    
    Side effect: Writes file to disk.
    
    Args:
        tensor: Tensor in [-1, 1] range, shape (C, H, W)
        path: Output file path
    """
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor_np = (tensor.detach().cpu().numpy() * 65535.0).round().astype(np.uint16)
    tensor_np = tensor_np.transpose(1, 2, 0)
    tensor_np = cv2.cvtColor(tensor_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), tensor_np)


def list_images_in_dir(directory: Path, pattern: str = '*.png') -> List[Path]:
    """
    List image files in directory.
    
    Side effect: Reads directory listing.
    """
    return sorted(directory.glob(pattern))


# ============================================================================
# Metrics Output (Side Effects)
# ============================================================================

import csv
from datetime import datetime
from typing import Dict, Any

def save_per_frame_metrics_csv(
    metrics_list: List[Dict[str, float]], 
    output_path: Path
) -> None:
    """
    Save per-frame metrics to CSV file.
    
    Side effect: Writes CSV file to disk.
    
    Args:
        metrics_list: List of metric dicts, one per frame
        output_path: Path to output CSV file
    """
    if not metrics_list:
        return
    
    # Get all unique keys across all frames
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    fieldnames = ['frame'] + sorted(all_keys)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, metrics in enumerate(metrics_list):
            row = {'frame': i + 1}
            row.update(metrics)
            writer.writerow(row)


def save_average_metrics_csv(
    avg_metrics: Dict[str, float],
    output_path: Path
) -> None:
    """
    Save average metrics to CSV file.
    
    Side effect: Writes CSV file to disk.
    """
    if not avg_metrics:
        return
    
    fieldnames = sorted(avg_metrics.keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(avg_metrics)


def save_results_txt(
    avg_metrics: Dict[str, float],
    video_name: str,
    model_info: Dict[str, Any],
    output_path: Path,
    file_sizes: Optional[Dict[str, int]] = None,
    timing_info: Optional[Dict[str, float]] = None
) -> None:
    """
    Save human-readable results summary to text file.
    
    Side effect: Writes text file to disk.
    
    Args:
        avg_metrics: Averaged metrics dict
        video_name: Name of the processed video
        model_info: Dict with model configuration
        output_path: Path to output text file
        file_sizes: Optional dict with 'original_video', 'stego_video' sizes in bytes
        timing_info: Optional dict with embed_time, extract_time, total_pipeline_time, etc.
    """
    def format_size(bytes_size: int) -> str:
        """Format bytes to human-readable size."""
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.2f} KB"
        else:
            return f"{bytes_size / (1024 * 1024):.2f} MB"
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("VIDEO STEGANOGRAPHY PIPELINE RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video: {video_name}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        for key, value in model_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Video file sizes section
        if file_sizes:
            f.write("-" * 40 + "\n")
            f.write("VIDEO FILE SIZES\n")
            f.write("-" * 40 + "\n")
            orig_size = file_sizes.get('original_video', 0)
            stego_size = file_sizes.get('stego_video', 0)
            f.write(f"  Original Video: {format_size(orig_size)} ({orig_size:,} bytes)\n")
            f.write(f"  Stego Video:    {format_size(stego_size)} ({stego_size:,} bytes)\n")
            if orig_size > 0:
                size_increase = ((stego_size - orig_size) / orig_size) * 100
                f.write(f"  Size Increase:  {size_increase:.2f}%\n")
            f.write("\n")
        
        # Pipeline timing section (for research paper)
        if timing_info:
            f.write("-" * 40 + "\n")
            f.write("PIPELINE TIMING (Excluding Metrics)\n")
            f.write("-" * 40 + "\n")
            embed_time = timing_info.get('embed_time', 0)
            extract_time = timing_info.get('extract_time', 0)
            total_time = timing_info.get('total_pipeline_time', 0)
            num_frames = timing_info.get('num_frames', 0)
            fps_embed = timing_info.get('fps_embed', 0)
            fps_extract = timing_info.get('fps_extract', 0)
            
            f.write(f"  Number of Frames:    {num_frames}\n")
            f.write(f"  Embedding Time:      {embed_time:.3f} s ({fps_embed:.2f} fps)\n")
            f.write(f"  Extraction Time:     {extract_time:.3f} s ({fps_extract:.2f} fps)\n")
            f.write(f"  Total Pipeline Time: {total_time:.3f} s\n")
            if num_frames > 0 and total_time > 0:
                avg_fps = num_frames / total_time
                ms_per_frame = (total_time / num_frames) * 1000
                f.write(f"  Average FPS:         {avg_fps:.2f}\n")
                f.write(f"  Time per Frame:      {ms_per_frame:.2f} ms\n")
            f.write("\n")
        
        hiding_metrics = {k: v for k, v in avg_metrics.items() if k.startswith('hiding_')}
        if hiding_metrics:
            f.write("-" * 40 + "\n")
            f.write("HIDING QUALITY (Cover vs Stego)\n")
            f.write("-" * 40 + "\n")
            for key in sorted(hiding_metrics.keys()):
                clean_key = key.replace('hiding_', '')
                f.write(f"  {clean_key}: {hiding_metrics[key]:.4f}\n")
            f.write("\n")
        
        recovery_metrics = {k: v for k, v in avg_metrics.items() if k.startswith('recovery_')}
        if recovery_metrics:
            f.write("-" * 40 + "\n")
            f.write("RECOVERY QUALITY (Original vs Recovered)\n")
            f.write("-" * 40 + "\n")
            for key in sorted(recovery_metrics.keys()):
                clean_key = key.replace('recovery_', '')
                f.write(f"  {clean_key}: {recovery_metrics[key]:.4f}\n")
            f.write("\n")
        
        other_metrics = {k: v for k, v in avg_metrics.items() 
                        if not k.startswith('hiding_') and not k.startswith('recovery_')}
        if other_metrics:
            f.write("-" * 40 + "\n")
            f.write("CAPACITY & EFFICIENCY\n")
            f.write("-" * 40 + "\n")
            for key in sorted(other_metrics.keys()):
                f.write(f"  {key}: {other_metrics[key]:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")


def save_results_json(
    avg_metrics: Dict[str, float],
    video_name: str,
    model_info: Dict[str, Any],
    output_path: Path,
    per_frame_metrics: Optional[List[Dict[str, float]]] = None,
    file_sizes: Optional[Dict[str, int]] = None,
    timing_info: Optional[Dict[str, float]] = None
) -> None:
    """
    Save all results to a structured JSON file for easy conversion to other formats.
    
    Side effect: Writes JSON file to disk.
    
    Args:
        avg_metrics: Averaged metrics dict
        video_name: Name of the processed video
        model_info: Dict with model configuration
        output_path: Path to output JSON file
        per_frame_metrics: Optional list of per-frame metric dicts
        file_sizes: Optional dict with 'original_video', 'stego_video' sizes in bytes
        timing_info: Optional dict with embed_time, extract_time, total_pipeline_time, etc.
    """
    import json
    
    # Organize metrics by category
    hiding_metrics = {k.replace('hiding_', ''): v for k, v in avg_metrics.items() if k.startswith('hiding_')}
    recovery_metrics = {k.replace('recovery_', ''): v for k, v in avg_metrics.items() if k.startswith('recovery_')}
    capacity_metrics = {k: v for k, v in avg_metrics.items() 
                       if not k.startswith('hiding_') and not k.startswith('recovery_')}
    
    # Build structured result
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "video_name": video_name,
            "pipeline_version": "1.0"
        },
        "model": model_info,
        "metrics": {
            "hiding_quality": hiding_metrics,
            "recovery_quality": recovery_metrics,
            "capacity_and_efficiency": capacity_metrics,
            "all_averaged": avg_metrics
        }
    }
    
    # Add file sizes if available
    if file_sizes:
        orig_size = file_sizes.get('original_video', 0)
        stego_size = file_sizes.get('stego_video', 0)
        size_increase = ((stego_size - orig_size) / orig_size * 100) if orig_size > 0 else 0
        result["file_sizes"] = {
            "original_video_bytes": orig_size,
            "stego_video_bytes": stego_size,
            "size_increase_percent": round(size_increase, 4)
        }
    
    # Add timing info if available
    if timing_info:
        result["timing"] = {
            "num_frames": timing_info.get('num_frames', 0),
            "embed_time_seconds": round(timing_info.get('embed_time', 0), 4),
            "extract_time_seconds": round(timing_info.get('extract_time', 0), 4),
            "total_pipeline_time_seconds": round(timing_info.get('total_pipeline_time', 0), 4),
            "fps_embed": round(timing_info.get('fps_embed', 0), 2),
            "fps_extract": round(timing_info.get('fps_extract', 0), 2)
        }
    
    # Add per-frame metrics if available
    if per_frame_metrics:
        result["per_frame_metrics"] = per_frame_metrics
    
    # Write JSON with pretty formatting
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)


