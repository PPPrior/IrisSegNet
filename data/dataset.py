import torch
from torch.utils.data import Dataset

from .preprocess import preprocess


class IrisDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        image, mask = preprocess(image_dir, mask_dir)
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        tensors = [image, mask]
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.shape = mask.shape

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
