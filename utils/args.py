import argparse
import yaml
from .misc import get_time_str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mimic-iv')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)

    # Training Setting
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--val_freq', type=int, default=5)

    # KG Configs
    parser.add_argument('--triplet_method', type=str, default='co-occurrence')

    # Model Configs

    # Data Paths
    parser.add_argument('--raw_data_path', type=str, default='')
    parser.add_argument('--save_data_path', type=str, default='')
    parser.add_argument('--graph_construction_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='')

    # Config File
    config_parser = argparse.ArgumentParser(description='Algorithm Config', add_help=False)
    config_parser.add_argument('-c', '--config', default=None, type=str, help='YAML config file')

    args_config, remaining = config_parser.parse_known_args()
    assert args_config.config is not None, 'Config file must be specified'

    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    args.start_time = get_time_str()

    assert args.train_proportion + args.val_proportion + args.test_proportion == 1., 'train-val-test split should sum to 1'
    # if args.code_freq_filter != 0:
    #     assert args.pad_dim >= args.code_freq_filter, 'Padding must exceed max number of codes'

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
