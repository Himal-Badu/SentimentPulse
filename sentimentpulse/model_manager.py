"""
SentimentPulse - Model utilities and management
Built by Himal Badu, AI Founder

Utilities for model management, downloading, and caching.
"""

import os
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from loguru import logger


class ModelManager:
    """Manages model files and caching."""
    
    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
    
    def get_model_size(self, model_name: str) -> Tuple[int, str]:
        """Get the size of a model in bytes.
        
        Returns:
            Tuple of (size_bytes, size_human_readable)
        """
        model_path = self.cache_dir / f"models--{model_name.replace('/', '--')}"
        
        if not model_path.exists():
            return 0, "0B"
        
        total_size = sum(
            f.stat().st_size 
            for f in model_path.rglob("*") 
            if f.is_file()
        )
        
        # Convert to human readable
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if total_size < 1024:
                return total_size, f"{total_size:.1f}{unit}"
            total_size /= 1024
        
        return int(total_size), f"{total_size:.1f}PB"
    
    def list_downloaded_models(self) -> List[Dict[str, str]]:
        """List all downloaded models.
        
        Returns:
            List of model info dictionaries
        """
        models = []
        
        if not self.cache_dir.exists():
            return models
        
        for model_folder in self.cache_dir.iterdir():
            if not model_folder.name.startswith("models--"):
                continue
            
            model_name = model_folder.name.replace("models--", "").replace("--", "/")
            size_bytes, size_human = self.get_model_size(model_name)
            
            models.append({
                "name": model_name,
                "path": str(model_folder),
                "size": size_human,
                "size_bytes": size_bytes
            })
        
        return sorted(models, key=lambda x: x["name"])
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model.
        
        Returns:
            True if deleted, False if not found
        """
        model_path = self.cache_dir / f"models--{model_name.replace('/', '--')}"
        
        if not model_path.exists():
            return False
        
        try:
            shutil.rmtree(model_path)
            logger.info(f"Deleted model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def clear_all_models(self) -> int:
        """Clear all downloaded models.
        
        Returns:
            Number of models deleted
        """
        count = 0
        models = self.list_downloaded_models()
        
        for model in models:
            if self.delete_model(model["name"]):
                count += 1
        
        return count
    
    def get_disk_usage(self) -> Dict[str, str]:
        """Get total disk usage of model cache.
        
        Returns:
            Dictionary with total size
        """
        total_size = sum(
            model["size_bytes"] 
            for model in self.list_downloaded_models()
        )
        
        # Convert to human readable
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if total_size < 1024:
                return {"total": f"{total_size:.1f}{unit}", "bytes": total_size}
            total_size /= 1024
        
        return {"total": f"{total_size:.1f}PB", "bytes": int(total_size)}


# Global model manager
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
