TRAIN_CONFIG = {
    'common': {
        'gpus': [0],
        'seed': 17,
        'folds_count': None,
        'batch_size': 2,
        'num_workers': 2,
        'epochs': 256,
        'exp_name': 'test_experiment',
        'wandb': False,
    },
    'model': {
        'target': 'segmentation_models_pytorch.Unet',
        'params': {
            'encoder_name': 'resnet34', 
            'classes': 4,
        }
    },
    'criterions': [
        {
            'target': 'segmentation_models_pytorch.losses.FocalLoss',
            'params': {
                'mode': 'multiclass',
            },
            'weight': 0.5,
            'name': 'focal',
        },
        {
            'target': 'segmentation_models_pytorch.losses.JaccardLoss',
            'params': {
                'mode': 'multiclass',
            },
            'weight': 0.5,
            'name': 'jaccard',
        },
    ],
    'optimizers': [
        {
            'target': 'torch.optim.Adam',
            'params': {
                'lr': 0.002,
            },
            'scheduler': {
                'target': 'pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR',
                'params': {
                    'warmup_epochs': 16,
                    'max_epochs': 256,
                    'warmup_start_lr': 0.001,
                },
                'additional': {
                    'monitor': 'iou_valid',
                },
            },
            
        }
    ],
    'metrics': [
        {
            'target': 'segmentation_models_pytorch.utils.metrics.IoU',
            'params': {
                'threshold': 0.5,
            },
            'name': 'iou',
        },
    ],
    'callbacks': [
        {
            'target': 'pytorch_lightning.callbacks.LearningRateMonitor',
        },
        {
            'target': 'pytorch_lightning.callbacks.ModelCheckpoint', 
            'params': {
                'dirpath': './models',
                'filename': 'best-{epoch:02d}-{iou_valid:2.2f}',
                'monitor': 'iou_valid',
                'mode': 'max',
                'save_top_k': 1,
                'save_last': True,
            }
        }
    ],
    'datasets': {
        'train': {
            'target': 'datasets.TrainDataset',
            'params':{
                'img_h': 256,
                'img_w': 256,
                'labels': [0, 6, 7, 10],
                'images_dir': '/home/user/datasets/denu/images',
                'masks_dir': '/home/user/datasets/denu/mask',
            },
        },
        'valid': {
            'target': 'datasets.TrainDataset',
            'params': {
                'img_h': 256,
                'img_w': 256,
                'labels': [0, 6, 7, 10],
                'images_dir': '/home/user/datasets/denu/images',
                'masks_dir': '/home/user/datasets/denu/mask',
            },
        }
    }
}