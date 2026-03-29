import os
import tempfile

import requests
from tqdm import tqdm
from utils.logging import log_for_all, log_for_0
from hashlib import md5
import pickle

def download(url, target_md5):
    # name = url[url.rfind('/') + 1 : url.rfind('?')]
    # cache file dir
    cache_path = '/tmp/inception_params.pkl'
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, 'rb'))
    
    i = 0
    log_for_0(f'Downloading inception checkpoint...')
    while i < 10:
        i += 1
        # log_for_all(f'Downloading: {url}')
        # assert url.endswith('?dl=1'), 'URL should end with ?dl=1, got: ' + url
        if i > 1:
            log_for_all(f'warning: retrying download {i}/10 ...')
        resp = requests.get(url)
        if resp.status_code != 200:
            log_for_all(f'Failed to download {url}, status code: {resp.status_code}, retrying...')
            continue
        data = resp.content
        md5_hash = md5(data).hexdigest()
        if md5_hash != target_md5:
            log_for_all(f'Checksum mismatch, expected {target_md5}, got {md5_hash}, retrying...')
            continue
        try:
            params_dict = pickle.loads(data)
            log_for_0(f'Downloaded and verified inception checkpoint successfully.')
            with open(cache_path, 'wb') as f:
                pickle.dump(params_dict, f)
            return params_dict
        except Exception as e:
            log_for_all(f'Failed to load pickle data, error: {e}, retrying...')
            continue
    raise RuntimeError(f'Failed to download or validate the file after {i} attempts.')



def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]

