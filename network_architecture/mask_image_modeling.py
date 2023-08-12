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

def _masking(x, mask_list):
    return x * torch.from_numpy(mask_list).to(x.device)

def _masking_reverse(x, mask_list):
    return x * (1. - torch.from_numpy(mask_list)).to(x.device)
    
def _get_mask_list(x, br):
    """_summary_

    Args:
        x (torch.tensor): [b, c, token_res_h, token_res_w, token_res_z, token_num]
    """
    num_mask = int(x.size()[-1] * np.min([br, 1]))
    
    mask_seq = np.concatenate([np.zeros(num_mask),
                                np.ones(x.size()[-1] - num_mask)])
    np.random.shuffle(mask_seq)
    mask_list = mask_seq
    return mask_list
        
def _image_to_token(x, mask_size):
    """_summary_
        In the original MIM, images are masked by dropping tokens. 
        For convolutional networks, we first need to partition images 
        into patches, which has a similar function to the tokenization 
        operation.
        
    Args:
        x (torch.tensor): [b, c, h, w, d]

    Returns:
        torch.tensor: [b, c, self.mask_size, self.mask_size, self.mask_size, token_num] 
    """
    b, c, h, w, d = x.size()

    x = x.view(b, c, 
                h // mask_size, mask_size, 
                w // mask_size, mask_size, 
                d // mask_size, mask_size)
    
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
    
    # [b, c, token_res_h, token_res_w, token_res_d, pixels of each token]
    x = rearrange(x, 'b c res_h res_w res_d h w d -> b c (res_h res_w res_d) h w d')
    token = x.permute(0, 1, 3, 4, 5, 2)
    return token
    
def _token_to_image(x, token, mask_size):
    b, c, h, w, d = x.size()
    x = token.view(b, c, 
                mask_size,
                mask_size, 
                mask_size,
                h // mask_size,
                w // mask_size,
                d // mask_size).permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = rearrange(x, 'b c res_h h res_w w res_d d -> b c (res_h h) (res_w w) (res_d d)')
    return x
    
def _synthesize_modalities(x, fusion_iter, syn_ratio):
    for i in range(fusion_iter):
        x = _fusion(x, syn_ratio)
    return x
    
def _fusion(x, syn_ratio):
    syn_seq = np.hstack([np.zeros(int(x.size()[-1] * syn_ratio)),
                np.ones(x.size()[-1] - int(x.size()[-1] * syn_ratio))])
    np.random.shuffle(syn_seq)
    
    # We exchange the patches in (modality_seq[0] th, modality_seq[1] th),
    # and (modality_seq[2] th, modality_seq[3] th) out of 4 modalities
    modality_seq = np.array([0, 1, 2, 3])
    np.random.shuffle(modality_seq)

    y = x.clone()
    for i in np.where(syn_seq > 0):
        y[:, modality_seq[0], :, :, :, i] = x[:, modality_seq[1], :, :, :, i]
        y[:, modality_seq[1], :, :, :, i] = x[:, modality_seq[0], :, :, :, i]
        y[:, modality_seq[2], :, :, :, i] = x[:, modality_seq[3], :, :, :, i]
        y[:, modality_seq[3], :, :, :, i] = x[:, modality_seq[2], :, :, :, i]
        
    return y
    
def synthesize_input(x, cube_size, syn_ratio=0.5):
        token = _image_to_token(x, cube_size)
        mask_list = _get_mask_list(token, syn_ratio)
        token = _synthesize_modalities(token, 2, syn_ratio)
        token = _masking(token, mask_list)
        x = _token_to_image(x, token, cube_size)

if __name__ == '__main__':
    inps = torch.randn((2, 4, 128, 128, 128), dtype=torch.float32)

    a, maskseq = random_block_masking_generator(inps, mask_ratio=0.5, mask_size=(4, 4, 4))

    out, _ = random_block_masking_generator(a, mask_ratio=0.5, mask_size=(4, 4, 4), mask_seq=[1 - x for x in maskseq])


