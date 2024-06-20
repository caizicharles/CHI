import pickle
import csv
import os.path as osp


def save_with_pickle(file: object, save_path: str, file_name: str):

    if not file_name.lower().endswith('.pickle'):
        file_name += '.pickle'

    file_path = osp.join(save_path, file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

    return


def load_with_pickle(save_path: str, file_name: str):

    if not file_name.lower().endswith('.pickle'):
        file_name += '.pickle'

    file_path = osp.join(save_path, file_name)
    assert osp.isfile(file_path), f'Dataset save path: {file_path} is invalid.'

    with open(file_path, "rb") as f:
        file = pickle.load(f)

    return file


def save_with_csv(file: dict, header: list, save_path: str, file_name: str):

    if not file_name.lower().endswith('.csv'):
        file_name += '.csv'

    file_path = osp.join(save_path, file_name)

    with open(file_path, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(file)


def read_csv_file(save_path: str, file_name: str):

    if not file_name.lower().endswith('.csv'):
        file_name += '.csv'

    file_path = osp.join(save_path, file_name)
    assert osp.isfile(file_path), f'Dataset save path: {file_path} is invalid.'

    with open(file_path, "r") as f:
        file = csv.reader(f)
        file = list(file)

    return file