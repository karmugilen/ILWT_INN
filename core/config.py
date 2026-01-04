"""
Functional Core - Configuration Dataclasses

Immutable configuration objects for the steganography pipeline.
These are pure data structures with no side effects.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable model configuration.
    
    To change models, simply create a new ModelConfig with different values.
    
    Example:
        # Default production model
        config = ModelConfig(path='model.pth')
        
        # Custom model
        config = ModelConfig(
            path='custom_model.pth',
            num_blocks=12,
            hidden_channels=256
        )
    """
    path: str
    num_blocks: int = 8
    hidden_channels: int = 128
    size: int = 256
    channels: int = 6
    transform_type: str = "ilwt53"


@dataclass(frozen=True)
class ProcessingConfig:
    """
    Immutable processing configuration.
    
    Controls how the pipeline processes videos.
    """
    batch_size: int = 8
    workers: int = 2
    cleanup: bool = False
    sequential: bool = False
    interactive: bool = False
    
    # Output patterns to keep during cleanup
    keep_patterns: Tuple[str, ...] = ('per_frame_metrics.csv', 'average_metrics.csv', 'results.txt', 'results.json')






@dataclass(frozen=True)
class PipelineConfig:
    """
    Complete pipeline configuration combining all sub-configs.
    
    This is the top-level config passed to the pipeline runner.
    """
    model: ModelConfig
    processing: ProcessingConfig
    hide_dir: str = 'hide'
    
    @classmethod
    def from_args(cls, args) -> 'PipelineConfig':
        """Create PipelineConfig from argparse namespace."""
        model = ModelConfig(
            path=args.model,
            num_blocks=args.num_blocks,
            hidden_channels=args.hidden_channels,
            size=args.size
        )
        
        # Interactive is default unless --batch is specified
        # Check if attribute exists to be safe (inference.py has it, other scripts might not)
        is_batch = getattr(args, 'batch', False)
        is_interactive = not is_batch
        
        processing = ProcessingConfig(
            batch_size=args.batch_size,
            workers=args.workers,
            cleanup=args.cleanup,
            sequential=args.sequential or is_interactive,  # Interactive implies sequential
            interactive=is_interactive
        )
        
        return cls(
            model=model,
            processing=processing,
            hide_dir=args.hide_dir
        )
