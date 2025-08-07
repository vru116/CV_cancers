import numpy as np
from matplotlib import pyplot as plt
import random
import torch

def tensor_imshow(inp, title=None, ax=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.detach().cpu().numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    if ax is None:
        ax = plt.gca()
    ax.imshow(inp, **kwargs)
    if title is not None:
        ax.set_title(title)

    # plt.imshow(inp, **kwargs)
    # if title is not None:
    #     plt.title(title)

def visualize_masks_and_masked_images(img, masks, num=6):
    img = img[0].detach().cpu().numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    masks = masks.cpu()

    indices = random.sample(range(len(masks)), num)

    plt.figure(figsize=(10, 3 * num))
    for i, idx in enumerate(indices):
        # print(masks[idx].shape)
        mask = masks[idx].squeeze().numpy()  # (1, H, W) -> (H, W)
        mask_vis = mask.copy()
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # (H, W) -> (H, W, 3)
        masked_img = img * mask  # (H, W, 3) * (H, W, 3)

        plt.subplot(num, 2, 2 * i + 1)
        plt.imshow(mask_vis, cmap='gray')
        plt.title(f'Mask #{idx}')
        plt.axis('off')

        plt.subplot(num, 2, 2 * i + 2)
        plt.imshow(masked_img)
        plt.title(f'Masked Image #{idx}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
