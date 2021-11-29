import ruamel.yaml as yaml


def read_pairs(pairs_file):
    files = []
    with open(pairs_file, 'r') as stream:
        pairs_dic = yaml.load_all(stream)
        # pairs_dic = yaml.load_all(stream, Loader=yaml.RoundTripLoader)

        for pair in pairs_dic:
            files.append(pair)
    return files


def read_cameras(cameras_file):
    files = []
    with open(cameras_file, 'r') as stream:
        pairs_dic = yaml.load_all(stream)
        # pairs_dic = yaml.load_all(stream, Loader=yaml.RoundTripLoader)

        for pair in pairs_dic:
            files.append(pair)
    return files


if __name__ == '__main__':
    cameras_file = './camera_parameters/cameras_calibrated.yaml'
        
    cameras = read_cameras(cameras_file)
    print(cameras)
