import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mimic-iv')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)

    # Training Setting
    parser.add_argument('--train_proportion', type=float, default=0.8)
    parser.add_argument('--val_proportion', type=float, default=0.1)
    parser.add_argument('--test_proportion', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=3)

    # Data Paths
    parser.add_argument('--raw_data_path', type=str, default='')
    parser.add_argument('--save_data_path', type=str, default='')

    # Model Params
    

    # Config File
    config_parser = argparse.ArgumentParser(description='Algorithm Config', add_help=False)
    config_parser.add_argument('-c',
                               '--config',
                               default=None,
                               type=str,
                               help='YAML config file')

    args_config, remaining = config_parser.parse_known_args()
    assert args_config.config is not None, 'Config file must be specified'

    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    assert args.train_proportion + args.val_proportion + args.test_proportion == 1., 'train-val-test split should sum to 1'

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)