import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class SolarPVDataset(Dataset):
    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "valid", "test"}
        self.root = root
        self.mode = mode
        self.transform = transform
        self.images_directory = os.path.join(self.root, self.mode, "images")
        self.annotations_file = os.path.join(self.root, self.mode, "_annotations.coco.json")
        self.data = self._read_annotations()

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        image_info = self.data["images"][idx]
        image_path = os.path.join(self.images_directory, image_info["file_name"])
        image = np.array(Image.open(image_path).convert("RGB"))
        annotation_ids = [ann["id"] for ann in self.data["annotations"] if ann["image_id"] == image_info["id"]]
        masks = self._get_masks(annotation_ids, image_info)

        sample = {"image": image, "mask": masks, 'info': image_info}
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def _read_annotations(self):
        with open(self.annotations_file) as f:
            data = json.load(f)
        return data

    def _get_masks(self, annotation_ids, image_info):
        masks = []
        for ann_id in annotation_ids:
            annotation = next(ann for ann in self.data["annotations"] if ann["id"] == ann_id)
            mask = self._preprocess_mask(annotation, image_info)
            mask = Image.fromarray(mask.astype(np.uint8))
            masks.append(mask)
        return masks

    @staticmethod
    def _preprocess_mask(annotation, image_info):
        mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.float32)
        for segmentation in annotation["segmentation"]:
            polygon = np.array(segmentation).reshape(-1, 2)
            mask = cv2.fillPoly(mask, [polygon.astype(np.int32)], 1.0)
        return mask


class SimpleSolarPVDataset(SolarPVDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)

        # resize images and masks
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
        masks = np.array([np.array(mask.resize((256, 256), Image.NEAREST)) for mask in sample["mask"]])
        
        combined_mask = np.any(masks, axis=0)
        final_mask = combined_mask.astype(np.uint8)

        # Convert the final mask to a NumPy array
        final_mask_array = np.array(final_mask)

        # Convert the final mask to a PyTorch tensor
        final_mask_tensor = torch.from_numpy(final_mask_array)

        # Add a batch dimension to the tensor
        final_mask_tensor = final_mask_tensor.unsqueeze(0)
        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.array(final_mask_tensor)
        # sample["masks"] = np.expand_dims(final_mask, axis=0)
        # sample["masks"] = np.expand_dims(combined_mask, 1)
        if sample['image'].shape != (3, 256, 256) or sample['mask'].shape != (1, 256, 256):
            print(sample['info'])

        return sample
