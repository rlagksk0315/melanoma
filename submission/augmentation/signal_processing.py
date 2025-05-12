import os
import numpy as np
from skimage import io, color, filters
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.feature import canny
from scipy import ndimage as ndi
from tqdm import tqdm


def main():
    input_dir = '../data/HAM10000/images'
    output_dir = '../data/darkHAM_sigproc'
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):

        if not filename.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(input_dir, filename)
        image = io.imread(image_path)

        gray = color.rgb2gray(image)

        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
        distance = ndi.distance_transform_edt(binary)
        markers = ndi.label(binary)[0]
        segmented = watershed(-distance, markers, mask=binary)

        mask = segmented > 0

        dark_image = image.copy().astype(np.float32)
        dark_image[~mask] = dark_image[~mask] * 0.35
        dark_image[mask] = dark_image[mask] * 0.8
        dark_image = np.clip(dark_image, 0, 255).astype(np.uint8)

        noise_image = dark_image.copy().astype(np.float32)

        mean, std = -80, 15
        noise = np.random.normal(mean, std, dark_image.shape).astype(np.float32)
        blurred_mask = gaussian(mask.astype(np.float32), sigma=20)
        blurred_mask_rgb = np.stack([blurred_mask] * 3, axis=-1)

        noise_image += noise * blurred_mask_rgb
        noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)

        lab_image = color.rgb2lab(noise_image.copy())
        a_shift, b_shift = -10, 20
        lab_image[..., 1][mask] += a_shift
        lab_image[..., 2][mask] += b_shift
        final_image = np.clip(color.lab2rgb(lab_image) * 255, 0, 255).astype(np.uint8)

        augmented_filename = filename.replace('.jpg', '_fake_Y.png')

        io.imsave(os.path.join(output_dir, augmented_filename), final_image, check_contrast=False)

if __name__ == "__main__":
    main()
