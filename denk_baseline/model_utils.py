import glob
import os
from typing import Dict, Any

import torch


class ModelExtractor:
    @staticmethod
    def extract_checkpoints(model: torch.nn.Module, save_path: str) -> None:
        """Extract and save model checkpoints without Lightning prefix.
        
        Args:
            model: The PyTorch Lightning model to extract checkpoints from
            save_path: Directory where checkpoints are saved
        """
        ckpt_paths = glob.glob(os.path.join(save_path, '*.ckpt'))
        ckpt_paths = sorted(ckpt_paths)
        model = model.cpu()

        for ckpt_path in ckpt_paths:
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
            
            # Remove Lightning prefix and batch augmentation layers
            state_dict = {}
            for name, params in model.state_dict().items():
                if 'batch_augs' in name:
                    continue
                new_name = name[6:] if name.startswith('model.') else name
                state_dict[new_name] = params
            
            torch.save({'state_dict': state_dict}, ckpt_path)

    @staticmethod
    def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
        """Load a checkpoint into a model.
        
        Args:
            model: The model to load the checkpoint into
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            The model with loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        return model 