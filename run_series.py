from omegaconf import OmegaConf

from train import parse_args, run_experiment

if __name__ == '__main__':
    args = parse_args()
    series_config = OmegaConf.load(args.config)
    for exp_name, changes in series_config['experiments'].items():
        main_config = OmegaConf.load(series_config['main_config'])
        main_config['common']['exp_name'] = main_config['common']['exp_name'] + f'_{exp_name}'
        for ch_name, change in changes.items():
            main_config[ch_name] = change
        run_experiment(main_config)