import numpy as np
import os
import cv2

from config import *
from preprocess import (
    mask_image,
    segment_and_threshold,
    sharpen_image,
    apply_power_rule,
    apply_average_filter,
)
from segmentation import get_connected_component, height_segment_threshold
from utils import create_directory
from region_grow import region_grow_cont, region_split_merge, merge_small_region
from colormap import apply_colormap_to_component


def main(image_path):
    create_directory(OUTPUT_DIR)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Preprocess
    img1 = mask_image(img, MASK_LEFT, MASK_RIGHT)
    img1 = apply_power_rule(img1, GAMMA)
    img2 = segment_and_threshold(img1, NUM_SEGMENTS, THRESHOLD_PERCENT)
    img3 = sharpen_image(img1)

    # Average filter
    avg1 = apply_average_filter(img1, 3)
    avg2 = apply_average_filter(avg1, 3)
    avg3 = apply_average_filter(avg2, 3)
    avg4 = apply_average_filter(avg3, 3)

    # Height segment threshold
    mask1 = height_segment_threshold(avg1, 16, 3)
    mask2 = height_segment_threshold(avg2, 16, 3)
    mask3 = height_segment_threshold(avg3, 16, 3)

    bitwisefilter = cv2.bitwise_and(mask1, mask2)
    bitwisefilter = cv2.bitwise_and(bitwisefilter, mask3)

    # Connected component
    component = get_connected_component(bitwisefilter)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{image_name}_img4_connected_component.jpg"),
        component,
    )

    # Region growing + merge
    region = region_grow_cont(img3, component, 28, vis=False)
    merge1 = region_split_merge(region)
    merge2 = merge_small_region(merge1, 0.5)
    merge3 = merge_small_region(merge2, 0.5)
    merge4 = merge_small_region(merge3, 0.5)
    merge4 = merge_small_region(merge4, 0.7)

    # Save each connected component
    for i in range(1, np.max(merge4) + 1):
        print(f"Connected component {i}: {np.sum(merge4 == i)} pixels")

        single_comp = np.where(merge4 == i, 255, 0).astype(np.uint8)

        if i < len(LABEL):
            out_name = f"{image_name}_{LABEL[i]}.jpg"
        else:
            out_name = f"{image_name}_{i}.jpg"

        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), single_comp)

        # highlight bounding box
        x, y, w, h = cv2.boundingRect(single_comp)
        img_rec = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        img_rec = cv2.rectangle(img_rec, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if i < len(LABEL):
            rec_name = f"{image_name}_{LABEL[i]}_highlight.jpg"
        else:
            rec_name = f"{image_name}_{i}_highlight.jpg"

        cv2.imwrite(os.path.join(OUTPUT_DIR, rec_name), img_rec)

    # Colormap image
    img_cmap = apply_colormap_to_component(merge4)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{image_name}_img5_connected_component_colormap.jpg"),
        img_cmap,
    )

    # Overlay on original grayscale
    img6 = cv2.addWeighted(
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        0.5,
        img_cmap,
        0.5,
        0
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{image_name}_img6_connected_component_on_gray.jpg"),
        img6,
    )

    # Save intermediate images
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}_img1_masked.jpg"), img1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}_img2_thresholded.jpg"), img2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}_img3_sharpened.jpg"), img3)


if __name__ == "__main__":
    image_path = os.path.join(DATA_DIR, "3.jpeg")
    main(image_path)