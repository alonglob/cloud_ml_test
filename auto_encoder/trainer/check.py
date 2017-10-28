import os, gzip
import _pickle as pickle

def unpacker(path):
    with gzip.open(path, 'rb') as f:
        print(path + 'is uncompressing and loading...')
        data = pickle.load(f)
    print(path + ' has been uploaded')
    return data

script_dir = os.path.dirname(__file__)
print(script_dir)
