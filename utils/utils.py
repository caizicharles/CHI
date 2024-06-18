import pickle
import json
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


def save_with_json(file, save_path: str, file_name: str):

    if not file_name.lower().endswith('.json'):
        file_name += '.json'

    file_path = osp.join(save_path, file_name)

    if type(file) == object:
        file = file.__dict__    

    with open(file_path, 'w') as f:
        json.dump(file, f)

    return


def load_with_json(save_path: str, file_name: str):

    if not file_name.lower().endswith('.json'):
        file_name += '.json'

    file_path = osp.join(save_path, file_name)
    assert osp.isfile(file_path), f'Dataset save path: {file_path} is invalid.'

    with open(file_path, "rb") as f:
        file = json.load(f)

    return file