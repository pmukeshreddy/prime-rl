"""
SGLang worker extensions for prime-rl.

This module provides worker-level extensions for the SGLang inference backend,
including support for weight broadcast mechanisms used in distributed RL training.

Weight Broadcast Backends:
- filesystem: Updates weights by reading from shared filesystem (default)
- nccl: Updates weights via NCCL collective operations for lower latency

Note: SGLang natively supports both filesystem and NCCL weight updates via its
built-in `update_weights_from_disk` and `init_weights_update_group` APIs.
The routes in server.py directly use these native SGLang capabilities.

For custom worker implementations (e.g., custom NCCL topologies), extend the
classes below.
"""

from typing import Optional


class WeightBroadcastBase:
    """Base class for weight broadcast implementations."""
    
    async def update_weights(self, weight_path: str) -> tuple[bool, str]:
        """
        Update model weights.
        
        Args:
            weight_path: Path to the weight checkpoint
            
        Returns:
            Tuple of (success, message)
        """
        raise NotImplementedError
    
    async def init_broadcast_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> tuple[bool, str]:
        """
        Initialize broadcast group for distributed weight sync.
        
        Args:
            master_address: Master node address
            master_port: Master node port
            rank_offset: This worker's rank offset in the group
            world_size: Total number of participants
            
        Returns:
            Tuple of (success, message)
        """
        raise NotImplementedError


class FilesystemBroadcast(WeightBroadcastBase):
    """
    Filesystem-based weight broadcast.
    
    This is the default and simplest approach - weights are saved to a shared
    filesystem by the trainer and loaded by inference servers.
    
    SGLang's native `update_weights_from_disk` handles this automatically.
    """
    pass


class NCCLBroadcast(WeightBroadcastBase):
    """
    NCCL-based weight broadcast.
    
    Uses NCCL collective operations to broadcast weights directly from trainer
    GPU memory to inference server GPU memory, avoiding filesystem I/O.
    
    SGLang's native `init_weights_update_group` handles this automatically.
    """
    pass
