"""
Dataset classes for fMoW data loading.
"""

import random

import h5py
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image


class BasefMoWDataset(Dataset):
    """
    Base dataset for fMoW satellite imagery (RGB or multispectral).

    Handles common HDF5 loading, data splitting, and transform application.
    Subclasses specify which bands to extract via the band_indices parameter.
    """

    def __init__(self, h5_file_path, data_split, band_indices, transform=None):
        """
        Args:
            h5_file_path (str): Path to the dataset.h5 file.
            data_split (int): Number of samples to use from the dataset.
            band_indices (list or tuple): Which bands to extract from the image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.h5_file_path = h5_file_path
        self.band_indices = band_indices
        self.transform = transform

        # Open the HDF5 file and load metadata
        with h5py.File(h5_file_path, "r") as h5_file:
            total_images = len(h5_file["images"])
            self.image_keys = list(range(total_images))
            random.shuffle(self.image_keys)  # Randomize the order
            self.image_keys = self.image_keys[:data_split]

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, "r") as h5_file:
            image = h5_file["images"][self.image_keys[idx]][:]

        # Extract specified bands and convert to tensor
        s2_image = torch.from_numpy(image).float()[self.band_indices]

        # Convert to PIL for augmentations (only single-channel or 3-channel inputs)
        if s2_image.shape[0] in (1, 3):
            s2_image = to_pil_image(s2_image)

        if self.transform:
            s2_image = self.transform(s2_image)

        return s2_image, s2_image


class fMoWRGBDataset(BasefMoWDataset):
    """Dataset for fMoW RGB satellite imagery."""

    def __init__(self, h5_file_path, data_split, transform=None):
        """
        Args:
            h5_file_path (str): Path to the dataset.h5 file.
            data_split (int): Number of samples to use from the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # RGB channels from Sentinel-2 bands: B04 (Red), B03 (Green), B02 (Blue)
        band_indices = [3, 2, 1]
        super().__init__(h5_file_path, data_split, band_indices, transform)


class fMoWMSDataset(BasefMoWDataset):
    """Dataset for fMoW multispectral satellite imagery."""

    def __init__(self, h5_file_path, data_split, transform=None):
        """
        Args:
            h5_file_path (str): Path to the dataset.h5 file.
            data_split (int): Number of samples to use from the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # All bands except coastal aerosol (B01), water vapor (B09), and cirrus (B10)
        # Indices: [1,2,3,4,5,6,7,8,11,12] match Sentinel-2 bands [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12]
        band_indices = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        super().__init__(h5_file_path, data_split, band_indices, transform)
