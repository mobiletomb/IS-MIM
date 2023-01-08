import random
import numpy as np
import torch


def load_permutations_3d(
        permutation_path='/home/qlc/model/nnUNet/permute_list.npy'
):

    perms = np.load(permutation_path)
    return perms


def gene_permutations_list():
    permute_list = []

    for i in range(100):
        permute = np.arange(32 * 32 * 32)
        rng = np.random.default_rng()
        rng.shuffle(permute)
        permute_list.append(np.asarray(permute))

    permute_list = np.stack(permute_list)
    np.save('permute_list.npy', permute_list)


def gene_pairs(x):
    perms_list = load_permutations_3d()
    label = []
    idx = []

    B, C, H, W, Z = x.shape
    z = h = w = 4

    x = x.view(B, C, Z // z, z, H // h, h, W // w, w)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
    x = x.contiguous().view(B, C, -1, z, h, w)

    for i in range(B):
        perm = random.choice(perms_list)
        x[i] = x[i, :, perm, :, :, :]
        for j, permutation in enumerate(perms_list):
            if (permutation == perm).all():
                label.append(torch.zeros(100, dtype=torch.float32))
                label[i][j] = 1
                idx.append(j)
                break

    x = x.view(B, C, Z // z, H // h, W // w, z, h, w)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7)
    x = x.contiguous().view(B, C, Z, H, W)

    label = torch.from_numpy(np.stack(label))
    return x, label, idx


if __name__ == '__main__':
    inps = torch.randn((2, 4, 128, 128, 128), dtype=torch.float32)

    x, label, idx = gene_pairs(inps)
    print(label.shape)




