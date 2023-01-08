import numpy as np
import torch


def random_block_masking_generator(x, mask_ratio, mask_size, mask_seq=None):
    B, C, Z, H, W = x.shape

    z, h, w = mask_size
    num_patches = int((Z * H * W) / (z * h * w))
    num_mask = int(num_patches * mask_ratio)

    x = x.view(B, C, Z // z, z, H // h, h, W // w, w)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
    x = x.contiguous().view(B, C, -1, z, h, w)

    if mask_seq is not None:
        mask_seq = mask_seq
    else:
        mask_seq = np.hstack([np.zeros(num_mask),
                              np.ones(num_patches - num_mask)])
        np.random.shuffle(mask_seq)

    mask_seq_mul = np.asarray(mask_seq)
    mask_seq_mul = torch.from_numpy(mask_seq_mul).view(1, 1, mask_seq_mul.shape[0], 1, 1, 1).type(x.dtype).to(x.device)

    x = torch.mul(x, mask_seq_mul)

    x = x.view(B, C, Z // z, H // h, W // w, z, h, w)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7)
    x = x.contiguous().view(B, C, Z, H, W)
    return x, mask_seq


def channel_wise(x, drop_ratio, drop_seq=None):
    """
    x: torch.tensor [B, C, Z, H, W]
    drop_seq: List
    """
    if drop_seq is not None:
        assert type(drop_seq).__name__ == 'list', f'drop_seq received an invalid arguments {type(drop_seq)} but' \
                                                  f'expected "list"'
        z = drop_seq[0]
        h = drop_seq[1]
        w = drop_seq[2]
        drop_list = drop_seq
    else:
        drop_list = []
        B, C, Z, H, W = x.shape

        drop_Z = int(Z * drop_ratio)
        drop_H = int(H * drop_ratio)
        drop_W = int(W * drop_ratio)

        z = np.arange(Z)
        h = np.arange(H)
        w = np.arange(W)

        z = np.random.choice(z, drop_Z, replace=False)
        h = np.random.choice(h, drop_H, replace=False)
        w = np.random.choice(w, drop_W, replace=False)

        drop_list.append(z)
        drop_list.append(h)
        drop_list.append(w)

    for i in z:
        x[:, :, i, :, :] = torch.zeros(x[:, :, i, :, :].shape, dtype=x.dtype)
    for i in h:
        x[:, :, :, i, :] = torch.zeros(x[:, :, :, i, :].shape, dtype=x.dtype)
    for i in w:
        x[:, :, :, :, i] = torch.zeros(x[:, :, :, :, i].shape, dtype=x.dtype)

    return x, drop_list


def constant_channel(x, drop_ratio, width, drop_seq=None):
    if drop_seq is not None:
        assert type(drop_seq).__name__ == 'list', f'drop_seq received an invalid arguments {type(drop_seq)} but' \
                                                  f'expected "list"'
        z = drop_seq[0]
        h = drop_seq[1]
        w = drop_seq[2]
        drop_list = drop_seq
    else:
        drop_list = []
        B, C, Z, H, W = x.shape

        drop_Z = int(Z // width * drop_ratio)
        drop_H = int(H // width * drop_ratio)
        drop_W = int(W // width * drop_ratio)

        z = np.arange(Z // width)
        h = np.arange(H // width)
        w = np.arange(W // width)

        z = np.random.choice(z, drop_Z, replace=False)
        h = np.random.choice(h, drop_H, replace=False)
        w = np.random.choice(w, drop_W, replace=False)

        drop_list.append(z)
        drop_list.append(h)
        drop_list.append(w)

    for i in z:
        x[:, :, i * width:(i + 1) * width, :, :] = torch.zeros(x[:, :, i * width:(i + 1) * width, :, :].shape,
                                                               dtype=x.dtype)
    for i in h:
        x[:, :, :, i * width:(i + 1) * width, :] = torch.zeros(x[:, :, :, i * width:(i + 1) * width, :].shape,
                                                               dtype=x.dtype)
    for i in w:
        x[:, :, :, :, i * width:(i + 1) * width] = torch.zeros(x[:, :, :, :, i * width:(i + 1) * width].shape,
                                                               dtype=x.dtype)

    return x, drop_list


def synthesize_input(x, cube_size, syn_ratio=0.5):
    B, C, Z, H, W = x.shape
    z, h, w = cube_size

    num_patches = int((Z * H * W) // (z * h * w))

    x = x.view(B, C, Z // z, z, H // h, h, W // w, w)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
    x = x.contiguous().view(B, C, -1, z, h, w)

    y = x.clone()

    syn_seq = np.hstack([np.zeros(int(num_patches * syn_ratio)),
                         np.ones(num_patches - int(num_patches * syn_ratio))])
    np.random.shuffle(syn_seq)

    for i in np.where(syn_seq > 0):
        y[:, 0:2, i, :, :, :] = x[:, 2:4, i, :, :, :]
        y[:, 2:4, i, :, :, :] = x[:, 0:2, i, :, :, :]

    y = y.view(B, C, Z // z, H // h, W // w, z, h, w)
    y = y.permute((0, 1, 2, 5, 3, 6, 4, 7))
    y = y.contiguous().view(B, C, Z, H, W)
    del x
    return y


if __name__ == '__main__':
    inps = torch.randn((2, 4, 128, 128, 128), dtype=torch.float32)

    a, maskseq = random_block_masking_generator(inps, mask_ratio=0.5, mask_size=(4, 4, 4))

    out, _ = random_block_masking_generator(a, mask_ratio=0.5, mask_size=(4, 4, 4), mask_seq=[1 - x for x in maskseq])


