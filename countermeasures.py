import numpy as np

# padding
# padding the file volume to the nearest multiple of k

def padding(real_files_volume, k = 500):
    padding_files_volume = real_files_volume.copy()
    if k == 0:
        return padding_files_volume
    for i in range(len(padding_files_volume)):
        file_volume = padding_files_volume[i]
        if file_volume % k == 0:
            padding_file_volume = file_volume
        else:
            padding_file_volume = (file_volume // k + 1) * k
        padding_files_volume[i] = padding_file_volume
    return padding_files_volume


def obfuscate(access_pattern, p, q):
    access_pattern_obfuscated = []
    for i in range(len(access_pattern)):
        tmp = []
        for j in range(len(access_pattern[0])):
            if access_pattern[i][j] == 1:
                if np.random.rand() > p:
                    tmp.append(0)
                else:
                    tmp.append(1)
            else:
                if np.random.rand() < q:
                    tmp.append(1)
                else:
                    tmp.append(0)
        access_pattern_obfuscated.append(tmp)
    return access_pattern_obfuscated