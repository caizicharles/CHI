from utils import get_args
from dataset import process_dataset


def main(args):

    dataset = process_dataset(args.dataset)
    print(dataset.__dict__)
    print(dataset.info())
    print(dataset.stat())
    
    return

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args=args)