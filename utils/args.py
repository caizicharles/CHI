import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mimic-iv')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)

    # Data Processing
    parser.add_argument('--age_thresh_low', type=int, default=None)
    parser.add_argument('--age_thresh_high', type=int, default=None)
    parser.add_argument('--code_freq_filter', type=int, default=None)
    parser.add_argument('--visit_thresh', type=int, default=None)
    parser.add_argument('--pad_dim', type=int, default=None)

    # Training Setting
    parser.add_argument('--train_proportion', type=float, default=0.8)
    parser.add_argument('--val_proportion', type=float, default=0.1)
    parser.add_argument('--test_proportion', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--val_freq', type=int, default=5)

    # KG Configs
    parser.add_argument('--triplet_method', type=str, default='co-occurrence')

    # Model Configs
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--gnn_layer', type=int, default=1)
    parser.add_argument('--gnn_hidden_dim', type=int, default=128)
    parser.add_argument('--trans_hidden_dim', type=int, default=128)

    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    # Data Paths
    parser.add_argument('--raw_data_path', type=str, default='')
    parser.add_argument('--save_data_path', type=str, default='')
    parser.add_argument('--graph_construction_path', type=str, default='')

    # Config File
    config_parser = argparse.ArgumentParser(description='Algorithm Config', add_help=False)
    config_parser.add_argument('-c', '--config', default=None, type=str, help='YAML config file')

    args_config, remaining = config_parser.parse_known_args()
    assert args_config.config is not None, 'Config file must be specified'

    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    assert args.train_proportion + args.val_proportion + args.test_proportion == 1., 'train-val-test split should sum to 1'
    # if args.code_freq_filter != 0:
    #     assert args.pad_dim >= args.code_freq_filter, 'Padding must exceed max number of codes'

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
