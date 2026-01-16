from typing import Dict, Any, Optional, List
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from denk_baseline.utils import instantiate_from_config


class LoggerFactory:
    @staticmethod
    def create_loggers(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create and configure loggers based on configuration."""
        str_loggers = config.get('loggers', [])
        loggers = {}
        
        for str_logger in str_loggers:
            logger = LoggerFactory._create_single_logger(str_logger, config)
            if logger:
                loggers[str_logger['target'].split('.')[-1].lower()] = logger
                
        return loggers if loggers else None

    @staticmethod
    def _create_single_logger(logger_config: Dict[str, Any], config: Dict[str, Any]) -> Optional[Any]:
        """Create a single logger instance based on configuration."""
        params = logger_config.get('params', {})
        target = logger_config['target']
        
        if 'TensorBoardLogger' in target:
            params.update({
                'name': config['general'].get('project_name', 'proj0'),
                'version': config['general'].get('exp_name', 'exp0'),
                'save_dir': config['general'].get('save_dir', 'output'),
            })
            return instantiate_from_config(logger_config)
            
        elif 'WandbLogger' in target:
            params.update({
                'name': config['general'].get('exp_name', 'exp0'),
                'project': config['general'].get('project_name', 'proj0'),
                'save_dir': config['general'].get('save_dir', 'output'),
            })
            return instantiate_from_config(logger_config)
            
        elif 'clearml' in target:
            from clearml import Task
            return Task.init(**params)
            
        return None 