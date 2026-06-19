import os
import cv2
import numpy as np
import random
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_ubyte
from skimage.morphology import convex_hull_image
from tqdm import tqdm
from collections import deque

def get_neighbors(labels, segments):
    neighbors = set()
    h, w = segments.shape
    for label in labels:
        ys, xs = np.where(segments == label)
        for y, x in zip(ys, xs):
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    nlabel = segments[ny, nx]
                    if nlabel not in labels:
                        neighbors.add(nlabel)
    return list(neighbors)

def process_batch(image_paths, mask_paths,
                  save_mask_dir, save_seg_dir,
                  save_superpixel_dir,   #
                  n_segments=200, compactness=10):

    color_list = [(255, 0, 0), (255, 165, 0),
                  (255, 255, 0), (0, 255, 0), (0, 0, 255)]

    for img_path, mask_path in zip(image_paths, mask_paths):
        img_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)

        # 读取数据
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_bin = (mask > 127).astype(np.uint8)

        # ------------------ SLIC 超像素 ------------------
        segments = slic(image,
                        n_segments=n_segments,
                        compactness=compactness,
                        start_label=0)

        superpixel_vis = mark_boundaries(image, segments, color=(1, 0, 0))
        superpixel_vis = img_as_ubyte(superpixel_vis)

        cv2.imwrite(
            os.path.join(save_superpixel_dir, img_name),
            cv2.cvtColor(superpixel_vis, cv2.COLOR_RGB2BGR)
        )

        all_labels = np.unique(segments)

        high_ratio_labels = []
        lesion_labels = []

        for label in all_labels:
            seg_mask = (segments == label)
            total = seg_mask.sum()
            lesion = np.sum(mask_bin[seg_mask])

            if total > 0 and lesion / total >= 0.5:
                high_ratio_labels.append(label)

            if lesion > 0:
                lesion_labels.append(label)

        rand_val = random.random()
        use_original_mask = rand_val >= 0.5

        selected_labels = []
        selection_order = []

        if use_original_mask:
            STAT["use_original"] += 1

        else:
            STAT["use_bfs"] += 1

            if high_ratio_labels:
                start_label = random.choice(high_ratio_labels)
                selected_labels.append(start_label)
                selection_order.append([start_label])
                queue = deque([start_label])

                while queue:
                    cur = queue.popleft()
                    neighbors = get_neighbors([cur], segments)

                    neighbors = [n for n in neighbors
                                 if n in high_ratio_labels
                                 and n not in selected_labels]

                    if not neighbors:
                        continue

                    new_label = random.choice(neighbors)
                    selected_labels.append(new_label)
                    queue.append(new_label)
                    selection_order.append([new_label])

                    # IoU 判断
                    mask_region = np.isin(segments, selected_labels).astype(np.uint8)
                    mask_hull = convex_hull_image(mask_region).astype(np.uint8)

                    inter = np.sum(mask_hull & mask_bin)
                    union = np.sum(mask_hull | mask_bin)
                    iou = inter / (union + 1e-8)

                    if iou >= 0.5:
                        break

            elif lesion_labels:
                start_label = random.choice(lesion_labels)
                selected_labels.append(start_label)
                selection_order.append([start_label])

        # ------------------ 生成最终 mask ------------------
        if use_original_mask:
            new_mask = mask_bin.copy()
        elif selected_labels:
            mask_region = np.isin(segments, selected_labels).astype(np.uint8)
            mask_hull = convex_hull_image(mask_region).astype(np.uint8)
            new_mask = mask_hull * mask_bin
        else:
            new_mask = np.zeros_like(mask_bin)

        cv2.imwrite(os.path.join(save_mask_dir, mask_name),
                    new_mask * 255)

        # ------------------ 可视化 ------------------
        seg_vis = mark_boundaries(image, segments, color=(1, 0, 0))
        seg_vis = img_as_ubyte(seg_vis)
        seg_with_mask = seg_vis.copy()

        for i, label_group in enumerate(selection_order):
            color = color_list[i % len(color_list)]
            for label in label_group:
                seg_with_mask[segments == label] = color

        if use_original_mask:
            cv2.putText(seg_with_mask,
                        "Original Mask",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2)

        overlay = seg_with_mask.copy()
        overlay[new_mask == 1] = (0, 0, 255)
        seg_with_mask = cv2.addWeighted(
            overlay, 0.4, seg_with_mask, 0.6, 0
        )

        cv2.imwrite(
            os.path.join(save_seg_dir, img_name),
            cv2.cvtColor(seg_with_mask, cv2.COLOR_RGB2BGR)
        )

def process_images_in_batches(image_dir, mask_dir,
                              save_mask_dir, save_seg_dir,
                              save_superpixel_dir,
                              batch_size=16,
                              n_segments=200,
                              compactness=10):

    os.makedirs(save_mask_dir, exist_ok=True)
    os.makedirs(save_seg_dir, exist_ok=True)
    os.makedirs(save_superpixel_dir, exist_ok=True)

    image_list = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png','.jpg','.jpeg'))
    ])

    mask_list = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.lower().endswith(('.png','.jpg','.jpeg'))
    ])

    assert len(image_list) == len(mask_list)

    total_batches = (len(image_list) + batch_size - 1) // batch_size

    for b in tqdm(range(total_batches), desc="Processing batches"):
        s = b * batch_size
        e = min(s + batch_size, len(image_list))

        process_batch(
            image_list[s:e],
            mask_list[s:e],
            save_mask_dir,
            save_seg_dir,
            save_superpixel_dir,
            n_segments,
            compactness
        )

    
if __name__ == "__main__":

    image_dir = "/root/autodl-tmp/TrainDataset/images"
    mask_dir = "/root/autodl-tmp/TrainDataset/masks"

    save_mask_dir = "/root/autodl-tmp/"
    save_seg_dir = "/root/autodl-tmp/"

    # 
    save_superpixel_dir = "/root/autodl-tmp/"

    process_images_in_batches(
        image_dir,
        mask_dir,
        save_mask_dir,
        save_seg_dir,
        save_superpixel_dir,
        batch_size=16,
        n_segments=70,
        compactness=15
    )