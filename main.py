from utils import *
from dataset import process_dataset

import time


def main(args):

    # start = time.time()
    
    # dataset = load_with_pickle(args.save_data_path, 'mimic-iv.pickle')

    # end = time.time()
    # print(dataset)
    # print(start-end)

    # exit()

    a = time.time()

    dataset = process_dataset(args.dataset)

    b = time.time()
    print(b-a)

    save_with_json(dataset, args.save_data_path, 'mimic-iv.json')

    c = time.time()
    print(c-b)

    dataset = load_with_json(args.save_data_path, 'mimic-iv.json')

    d = time.time()
    print(d-c)
    print(dataset)

    
    return

if __name__ == '__main__':
    args = get_args()
    main(args=args)