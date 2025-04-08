import os
import torch
import numpy as np
from PIL import Image 
from skimage.transform import resize
from torchvision import transforms as T
import matplotlib.pyplot as plt


def preprocess_image(image_path, output_size=256, original_image=False):
    """Drop irrelevant areas from an image using average pixel values."""
    image = Image.open(image_path).convert('L')

    tensor = T.ToTensor()(image)
    array = tensor.data.cpu().numpy()

    original_shape = array.shape

    x_values = np.mean(array, axis=0)
    y_values = np.mean(array, axis=1)

    x_range = (int(0.24 * len(x_values)), int(2.2 * len(x_values)))
    y_range = (int(0.8 * len(y_values)), int(1.8 * len(y_values)))

    threshold = 5

    x_cut = np.argwhere(x_values <= threshold).flatten()
    x_min = x_cut[x_cut <= x_range[0]].max() if x_cut[x_cut <= x_range[0]].size else 0
    x_max = x_cut[x_cut >= x_range[1]].min() if x_cut[x_cut >= x_range[1]].size else original_shape[0]

    y_cut = np.argwhere(y_values <= threshold).flatten()
    y_min = y_cut[y_cut <= y_range[0]].max() if y_cut[y_cut <= y_range[0]].size else 0
    y_max = y_cut[y_cut >= y_range[1]].min() if y_cut[y_cut >= y_range[1]].size else original_shape[1]

    if original_image:
        x_min = 0
        x_max = original_shape[0]
        y_min = 0
        y_max = original_shape[1]

    cut_image = array[x_min:x_max, y_min:y_max]
    cut_image_shape = cut_image.shape

    cut_image = resize(cut_image, (output_size, output_size), order=3)

    cut_image_tensor = torch.tensor(data=cut_image, dtype=tensor.dtype)

    return [
        cut_image_tensor,
        cut_image_shape,
        original_shape,
        [x_min, x_max, y_min, y_max]
    ]
    
    
def preprocess_mask(mask_path, crop_coords, output_size=256):
    """Drop irrelevant areas from a mask using average pixel values."""
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    x_min, x_max, y_min, y_max = crop_coords
    cut_mask = mask_array[x_min:x_max, y_min:y_max]
    resized_mask = resize(
        cut_mask,
        (output_size, output_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )

    return resized_mask


def preprocess_and_save(image_folder, mask_folder, output_folder, output_size=256):
    images_output_dir = os.path.join(output_folder, "images")
    masks_output_dir = os.path.join(output_folder, "masks")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)
    
    image_files = os.listdir(image_folder)
    
    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name)
        
        processed_image, cut_shape, original_shape, crop_coordinates = TNSCUI_preprocess(image_path, output_size)
        processed_image = processed_image.unsqueeze(0).unsqueeze(0).to(device='cpu')
        image_array = processed_image.squeeze().cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        
        mask_array = TNSCUI_mask_preprocess(mask_path, crop_coordinates, output_size)
        
        image_save_path = os.path.join(images_output_dir, image_name)
        mask_save_path = os.path.join(masks_output_dir, image_name)
        
        Image.fromarray(image_array).convert(mode='L').save(image_save_path)
        plt.imsave(mask_save_path, mask_array, cmap='gray')
    
    print("Preprocessing complete! Files saved in", output_folder)
