import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser()

    # Model Params
    parser.add_argument('--dataset', type=str, default='mimic-iv')

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

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)