from dataclasses import dataclass
from typing import Optional, Dict, Any
from omegaconf import OmegaConf
import os


@dataclass
class ExperimentConfig:
    exp_name: str
    project_name: str
    save_dir: str
    seed: int
    save_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        general = config_dict.get('general', {})
        exp_name = general.get('exp_name', 'noname_experiment')
        project_name = general.get('project_name', 'noname_project')
        save_dir = general.get('save_dir', 'noname_project_output')
        seed = general.get('seed', 42)
        
        save_path = os.path.join(save_dir, project_name, exp_name)
        
        return cls(
            exp_name=exp_name,
            project_name=project_name,
            save_dir=save_dir,
            seed=seed,
            save_path=save_path
        )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return normalize_config(config)


def _default_loss_name(target: str) -> str:
    return target.split('.')[-1].lower()


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy configs to the current schema."""
    if 'criterions' not in config and 'criterion' in config:
        criterion = config['criterion']
        criterion.setdefault('weight', 1.0)
        criterion.setdefault('name', _default_loss_name(criterion['target']))
        config['criterions'] = [criterion]

    if 'optimizers' not in config and 'optimizer' in config:
        optimizer = config['optimizer']
        if 'lr_scheduler' in config and 'scheduler' not in optimizer:
            optimizer['scheduler'] = config['lr_scheduler']
        config['optimizers'] = [optimizer]

    if 'metrics' not in config:
        config['metrics'] = []

    lightning_model = config.get('lightning_model')
    if lightning_model:
        if lightning_model.endswith('SegmentationModel'):
            classes = config.get('model', {}).get('params', {}).get('classes', 1)
            config['lightning_model'] = (
                'denk_baseline.lightning_models.SegmentationBinaryModel'
                if classes == 1
                else 'denk_baseline.lightning_models.SegmentationMulticlassModel'
            )
        elif lightning_model.endswith('ClassificationModel'):
            num_classes = config.get('model', {}).get('params', {}).get('num_classes', 1)
            config['lightning_model'] = (
                'denk_baseline.lightning_models.ClassificationBinaryModel'
                if num_classes == 1
                else 'denk_baseline.lightning_models.ClassificationMulticlassModel'
            )
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.yaml"), 'w') as file:
        OmegaConf.save(config=config, f=file) 