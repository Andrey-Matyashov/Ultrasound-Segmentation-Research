# С проекта
import torch
from PIL import Image
from skimage.transform import resize
from torchvision import transforms as T
import numpy as np
import os
import matplotlib.pyplot as plt


def TNSCUI_preprocess(image_path, outputsize=256, orimg=False):
    img = Image.open(image_path)
    img = img.convert(mode='L')

    Transform = T.Compose([T.ToTensor()])
    img_tensor = Transform(img)
    img_dtype = img_tensor.dtype
    # print(img_dtype)
    img_array_fromtensor = (torch.squeeze(img_tensor)).data.cpu().numpy()

    img_array = np.array(img, dtype=np.float32)

    or_shape = img_array.shape  #原始图片的尺寸

    value_x = np.mean(img, 1) #% 为了去除多余行，即每一列平均
    value_y = np.mean(img, 0) #% 为了去除多余列，即每一行平均

    x_hold_range = list((len(value_x) * np.array([0.24 / 3, 2.2 / 3])).astype(np.int_))
    y_hold_range = list((len(value_y) * np.array([0.8 / 3, 1.8 / 3])).astype(np.int_))
    # x_hold_range = list((len(value_x) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))
    # y_hold_range = list((len(value_y) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))

    # value_thresold = 0
    value_thresold = 5

    x_cut = np.argwhere((value_x<=value_thresold)==True)
    x_cut_min = list(x_cut[x_cut<=x_hold_range[0]])
    if x_cut_min:
        x_cut_min = max(x_cut_min)
    else:
        x_cut_min = 0

    x_cut_max = list(x_cut[x_cut>=x_hold_range[1]])
    if x_cut_max:
        # print('q')
        x_cut_max = min(x_cut_max)
    else:
        x_cut_max = or_shape[0]

    y_cut = np.argwhere((value_y<=value_thresold)==True)
    y_cut_min = list(y_cut[y_cut<=y_hold_range[0]])
    if y_cut_min:
        y_cut_min = max(y_cut_min)
    else:
        y_cut_min = 0

    y_cut_max = list(y_cut[y_cut>=y_hold_range[1]])
    if y_cut_max:
        # print('q')
        y_cut_max = min(y_cut_max)
    else:
        y_cut_max = or_shape[1]

    if orimg:
        x_cut_max = or_shape[0]
        x_cut_min = 0
        y_cut_max = or_shape[1]
        y_cut_min = 0
    # 截取图像
    cut_image = img_array_fromtensor[x_cut_min:x_cut_max,y_cut_min:y_cut_max]
    cut_image_orshape = cut_image.shape

    cut_image = resize(cut_image, (outputsize, outputsize), order=3)

    cut_image_tensor = torch.tensor(data = cut_image,dtype=img_dtype)
    # print(cut_image_tensor.dtype)

    return [cut_image_tensor, cut_image_orshape,or_shape,[x_cut_min,x_cut_max,y_cut_min,y_cut_max]]


def TNSCUI_mask_preprocess(mask_path, crop_coords, outputsize=256):
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    # print(mask_array.dtype)
    # print(mask_array)
    
    x_min, x_max, y_min, y_max = crop_coords
    cut_mask = mask_array[x_min:x_max, y_min:y_max]
    resized_mask = resize(cut_mask, (outputsize, outputsize), order=0, preserve_range=True, anti_aliasing=False)
    # mask_tensor = torch.tensor(resized_mask, dtype=torch.float32)
    
    # return mask_tensor
    return resized_mask


def preprocess_and_save(image_folder, mask_folder, output_folder, outputsize=256):
    images_output = os.path.join(output_folder, "images")
    masks_output = os.path.join(output_folder, "masks")
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(masks_output, exist_ok=True)
    
    image_files = os.listdir(image_folder)
    
    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        mask_path = os.path.join(mask_folder, img_name)
        
        processed_img, cut_shape, original_shape, crop_coords = TNSCUI_preprocess(img_path, outputsize)
        processed_img = torch.unsqueeze(processed_img, 0)
        processed_img = torch.unsqueeze(processed_img, 0)
        processed_img = processed_img.to(device='cpu')
        img_array = (torch.squeeze(processed_img)).data.cpu().numpy()
        # print(img_array.dtype)
        # print(img_array.shape)
        # print(img_array)
        img_array = (img_array * 255).astype(np.uint8)

        # mask_tensor = TNSCUI_mask_preprocess(mask_path, crop_coords, outputsize)
        mask_array = TNSCUI_mask_preprocess(mask_path, crop_coords, outputsize)
        # print(mask_array.dtype)
        # print(mask_array)
        
        img_save_path = os.path.join(images_output, img_name)
        mask_save_path = os.path.join(masks_output, img_name)
        
        Image.fromarray(img_array).convert(mode='L').save(img_save_path)
        # plt.imsave(img_save_path, img_array, cmap='gray')
        # Image.fromarray(img_array).save("output.png", "PNG")
        # plt.imsave(mask_save_path, mask_tensor.numpy(), cmap='gray')
        plt.imsave(mask_save_path, mask_array, cmap='gray')
    
    print("Предобработка завершена! Файлы сохранены в ", output_folder)


# # LONG
# # train
# preprocess_and_save(
#     image_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\long\\train\\images',
#     mask_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\long\\train\\masks',
#     output_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\2_preprocessed_size256\\long\\train',
#     outputsize=256
# )

# # test
# preprocess_and_save(
#     image_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\long\\test\\images',
#     mask_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\long\\test\\masks',
#     output_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\2_preprocessed_size256\\long\\test',
#     outputsize=256
# )


# # CROSS
# # train
# preprocess_and_save(
#     image_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\cross\\train\\images',
#     mask_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\cross\\train\\masks',
#     output_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\2_preprocessed_size256\\cross\\train',
#     outputsize=256
# )

# # test
# preprocess_and_save(
#     image_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\cross\\test\\images',
#     mask_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\cross\\test\\masks',
#     output_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\2_preprocessed_size256\\cross\\test',
#     outputsize=256
# )


# # ALL
# # train
# preprocess_and_save(
#     image_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\all\\train\\images',
#     mask_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\all\\train\\masks',
#     output_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\2_preprocessed_size256\\all\\train',
#     outputsize=256
# )

# # test
# preprocess_and_save(
#     image_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\all\\test\\images',
#     mask_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\1_no_preprocessing\\all\\test\\masks',
#     output_folder='D:\\Study\\CV_Project\\ACTUAL_DATA_2025\\data\\one_nodule\\step2_segmentation_dataset\\2_preprocessed_size256\\all\\test',
#     outputsize=256
# )
