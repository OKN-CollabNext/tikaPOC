#!/usr/bin/env python
import os
import torch
import numpy as np
from PIL import Image
from segmentation_main.segmentation_demo import SemanticSegmentation, DLStudio

class SegmentationAgent:
    def __init__(self):
        # Define more descriptive segmentation classes.
        # Here, we use urban scene categories (without the background, which is index 0).
        self.classes = ('person', 'car', 'bus', 'bicycle', 'building')

        # Initialize DLStudio configuration.
        self.config = DLStudio(
            dataroot="./data/",
            image_size=[64, 64],
            path_saved_model="./saved_model",
            momentum=0.9,
            learning_rate=1e-4,
            epochs=6,
            batch_size=4,
            use_gpu=torch.cuda.is_available()
        )

        # Create the segmentation instance (with a maximum of 5 objects per image).
        self.segmenter = SemanticSegmentation(dl_studio=self.config, max_num_objects=5)

        # Build the U-Net–like model (mUnet) with skip connections enabled.
        self.model = self.segmenter.mUnet(skip_connections=True, depth=12)

        # Load pretrained model weights if available.
        self._load_pretrained_model()

        # Set the model to evaluation mode.
        self.model.eval()

        # Define a mapping from output indices to human-readable labels.
        # Index 0 is reserved for background.
        self.label_mapping = {0: "background"}
        for i, cls in enumerate(self.classes, start=1):
            self.label_mapping[i] = cls

    def _load_pretrained_model(self):
        model_file = self.config.path_saved_model
        if os.path.exists(model_file):
            try:
                # Load the state dictionary on the CPU.
                state_dict = torch.load(model_file, map_location=torch.device("cpu"))
                self.model.load_state_dict(state_dict)
                print("Model loaded successfully!")
            except Exception as err:
                print(f"Could not load pretrained model. Error: {err}")
        else:
            print(f"Pretrained model file not found at {model_file}.")

    def segment_image(self, input_image: Image.Image) -> (Image.Image, str):
        """
        Accepts a PIL image, processes it, and returns:
          1. A grayscale segmentation mask image.
          2. A semantic description generated by mapping the predicted indices to class names.
        """
        if input_image.mode == 'RGBA':
            input_image = input_image.convert('RGB')

        target_size = tuple(self.config.image_size)
        resized_img = input_image.resize(target_size)
        img_np = np.array(resized_img)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        img_np = img_np.transpose(2, 0, 1)
        img_tensor = torch.tensor(img_np, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            pred_tensor = self.model(img_tensor)
            pred_tensor = pred_tensor.squeeze(0)

        if pred_tensor.shape[0] > 1:
            pred_np = torch.argmax(pred_tensor, dim=0).cpu().numpy()
        else:
            pred_np = pred_tensor.squeeze(0).cpu().numpy()

        max_val = pred_np.max() if pred_np.max() > 0 else 1
        norm_pred = (pred_np * 255.0 / max_val).astype('uint8')
        mask_image = Image.fromarray(norm_pred, mode='L')

        unique_indices = np.unique(pred_np)
        labels = [self.label_mapping.get(idx, "undefined") for idx in unique_indices if idx != 0]
        if labels:
            semantic_description = "The image contains: " + ", ".join(labels) + "."
        else:
            semantic_description = "No relevant objects detected."

        return mask_image, semantic_description

if __name__ == "__main__":
    agent = SegmentationAgent()
    test_image = Image.open("your_image.jpg")  # Replace with an actual image
    mask, description = agent.segment_image(test_image)
    mask.show()
    print(description)
