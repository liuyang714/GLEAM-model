"""
Step 4+5 (merged): Coordinate restoration + deduplication + high-res extraction.
Input:  YOLO txt labels from cropped patches, original WSI (SVS/NDPI)
Output: Individual glomerulus images at Level 0 full resolution
"""
import os
import json
from PIL import Image, ImageDraw
from openslide import OpenSlide
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ---------- Coordinate helpers ----------

def extract_base_name(filename):
    name_without_extension = os.path.splitext(filename)[0]
    parts = name_without_extension.split('_')
    return '_'.join(parts[:-2])


def convert_yolo_to_shapes(yolo_data, image_width, image_height, x_offset, y_offset):
    shapes = []
    for line in yolo_data.strip().split('\n'):
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, width, height = map(float, parts)

        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        x_min = x_center - width / 2 + x_offset
        y_min = y_center - height / 2 + y_offset
        x_max = x_center + width / 2 + x_offset
        y_max = y_center + height / 2 + y_offset

        shapes.append({
            "points": [[x_min, y_min], [x_max, y_max]],
            "shape_type": "rectangle",
            "area": (x_max - x_min) * (y_max - y_min)
        })
    return shapes


def calculate_overlap_area(box1, box2):
    x1 = max(box1[0][0], box2[0][0])
    y1 = max(box1[0][1], box2[0][1])
    x2 = min(box1[1][0], box2[1][0])
    y2 = min(box1[1][1], box2[1][1])
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0


def deduplicate_shapes(shapes):
    shapes = sorted(shapes, key=lambda x: x["area"], reverse=True)
    filtered_shapes = []

    for shape in shapes:
        points = shape["points"]
        x_min, y_min = points[0]
        x_max, y_max = points[1]
        w, h = x_max - x_min, y_max - y_min
        if min(w, h) <= 0 or max(w, h) / min(w, h) > 2:
            continue

        keep = True
        for kept in filtered_shapes:
            overlap_area = calculate_overlap_area(shape["points"], kept["points"])
            smaller_area = min(shape["area"], kept["area"])
            if smaller_area > 0 and overlap_area / smaller_area > 0.5:
                keep = False
                break
        if keep:
            filtered_shapes.append(shape)

    return filtered_shapes


# ---------- Step 4: restore + deduplicate ----------

def restore_and_deduplicate(base_name, crop_folder, txt_folder, crop_size=400):
    """Stitch crops back, convert YOLO labels to Level 2 coords, deduplicate."""
    coordinates = set()
    for file in os.listdir(crop_folder):
        if file.startswith(base_name) and file.endswith('.jpg'):
            parts = os.path.splitext(file)[0].split('_')
            if len(parts) >= 3:
                coordinates.add((int(parts[-2]), int(parts[-1])))

    if not coordinates:
        return []

    all_shapes = []
    for file in os.listdir(crop_folder):
        if file.startswith(base_name) and file.endswith('.jpg'):
            parts = os.path.splitext(file)[0].split('_')
            x_offset, y_offset = int(parts[-2]), int(parts[-1])

            txt_path = os.path.join(txt_folder, f"{os.path.splitext(file)[0]}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    yolo_data = f.read()
                shapes = convert_yolo_to_shapes(yolo_data, crop_size, crop_size, x_offset, y_offset)
                all_shapes.extend(shapes)

    return deduplicate_shapes(all_shapes)


# ---------- Step 5: extract at Level 0 ----------

def get_level2_to_level0_scale(slide):
    """Compute per-axis scale factor from Level 2 to Level 0."""
    dim0 = slide.level_dimensions[0]
    dim2 = slide.level_dimensions[2]
    return dim0[0] / dim2[0], dim0[1] / dim2[1]


def extract_glomeruli_from_wsi(wsi_path, shapes, output_folder, padding_l2=0):
    """Extract full-resolution glomerulus images from original WSI."""
    slide = OpenSlide(wsi_path)
    slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
    os.makedirs(output_folder, exist_ok=True)

    if wsi_path.lower().endswith('.svs'):
        scale_x, scale_y = get_level2_to_level0_scale(slide)
    else:
        # NDPI: fixed 16x
        scale_x, scale_y = 16, 16

    for shape in shapes:
        x1_l2, y1_l2 = shape["points"][0]
        x2_l2, y2_l2 = shape["points"][1]

        # Optional padding (used by NDPI to add context margin)
        if padding_l2 > 0:
            x1_l2 -= padding_l2
            y1_l2 -= padding_l2
            x2_l2 += padding_l2
            y2_l2 += padding_l2

        x1 = int(x1_l2 * scale_x)
        y1 = int(y1_l2 * scale_y)
        x2 = int(x2_l2 * scale_x)
        y2 = int(y2_l2 * scale_y)
        w, h = x2 - x1, y2 - y1

        if w <= 0 or h <= 0:
            continue

        region = slide.read_region((x1, y1), 0, (w, h))
        region.save(os.path.join(output_folder, f"{slide_name}_{x1}_{y1}.png"))

    slide.close()


# ---------- Save restored image + JSON (optional) ----------

def save_restored_image(base_name, crop_folder, shapes, output_folder, crop_size=400):
    """Stitch crops into a full Level 2 image with bounding boxes drawn."""
    coordinates = set()
    for file in os.listdir(crop_folder):
        if file.startswith(base_name) and file.endswith('.jpg'):
            parts = os.path.splitext(file)[0].split('_')
            if len(parts) >= 3:
                coordinates.add((int(parts[-2]), int(parts[-1])))

    if not coordinates:
        return

    max_x = max(c[0] for c in coordinates)
    max_y = max(c[1] for c in coordinates)
    w, h = max_x + crop_size, max_y + crop_size

    restored = Image.new("RGB", (w, h), (255, 255, 255))
    for file in os.listdir(crop_folder):
        if file.startswith(base_name) and file.endswith('.jpg'):
            parts = os.path.splitext(file)[0].split('_')
            x, y = int(parts[-2]), int(parts[-1])
            restored.paste(Image.open(os.path.join(crop_folder, file)), (x, y))

    draw = ImageDraw.Draw(restored)
    for shape in shapes:
        x1, y1 = shape["points"][0]
        x2, y2 = shape["points"][1]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 150), width=4)

    os.makedirs(output_folder, exist_ok=True)
    restored.save(os.path.join(output_folder, f"{base_name}_restored.jpg"))


def save_json(base_name, shapes, json_folder, w, h):
    os.makedirs(json_folder, exist_ok=True)
    data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{base_name}_restored.jpg",
        "imageWidth": w,
        "imageHeight": h
    }
    with open(os.path.join(json_folder, f"{base_name}.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


# ---------- Main pipeline ----------

def process_pipeline(crop_folder, txt_folder, wsi_folder, output_folder,
                     json_folder=None, restored_folder=None,
                     crop_size=400, padding_l2=0, max_workers=4):
    """
    End-to-end: YOLO labels -> deduplicate -> extract Level 0 glomeruli.
    Optional: save restored Level 2 image and JSON for LabelMe review.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Collect base names from cropped patches
    base_names = set()
    for f in os.listdir(crop_folder):
        if f.endswith('.jpg'):
            base_names.add(extract_base_name(f))

    # Build WSI lookup: base_name -> file path
    wsi_map = {}
    for f in os.listdir(wsi_folder):
        if f.lower().endswith(('.svs', '.ndpi')):
            wsi_map[os.path.splitext(f)[0]] = os.path.join(wsi_folder, f)

    total_extracted = 0

    with tqdm(total=len(base_names), desc="Processing", unit="slide") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for base_name in base_names:
                # Step 4: restore + deduplicate
                shapes = restore_and_deduplicate(base_name, crop_folder, txt_folder, crop_size)

                if not shapes:
                    pbar.update(1)
                    continue

                # Optional: save restored image and JSON
                if restored_folder:
                    save_restored_image(base_name, crop_folder, shapes, restored_folder, crop_size)
                if json_folder:
                    coords = set()
                    for f in os.listdir(crop_folder):
                        if f.startswith(base_name) and f.endswith('.jpg'):
                            parts = os.path.splitext(f)[0].split('_')
                            if len(parts) >= 3:
                                coords.add((int(parts[-2]), int(parts[-1])))
                    if coords:
                        mw = max(c[0] for c in coords) + crop_size
                        mh = max(c[1] for c in coords) + crop_size
                        save_json(base_name, shapes, json_folder, mw, mh)

                # Step 5: extract from WSI
                if base_name in wsi_map:
                    fut = executor.submit(
                        extract_glomeruli_from_wsi,
                        wsi_map[base_name], shapes, output_folder, padding_l2
                    )
                    futures[fut] = base_name
                else:
                    print(f"Warning: No matching WSI for {base_name}")
                    pbar.update(1)

            for fut in as_completed(futures):
                fut.result()
                pbar.update(1)

    # Count output
    n = len([f for f in os.listdir(output_folder) if f.endswith('.png')])
    print(f"\nDone! Extracted {n} glomerulus images from {len(base_names)} slides.")


if __name__ == "__main__":
    crop_folder   = r"/mnt/disk3/liuyang/KidneyData/sc2leve2crops/704/crop"
    txt_folder    = r"/mnt/disk3/liuyang/yolov8/runs/detect/predict28/labels"
    wsi_folder    = r"/mnt/disk3/liuyang/KidneyData/sc2svs/szdmt第3批svs"
    output_folder = r"/mnt/disk3/liuyang/KidneyData/sc2leve2crops/704/cropglos"

    # Optional outputs (set to None to skip)
    json_folder    = r"/mnt/disk3/liuyang/KidneyData/sc2leve2crops/704/json"
    restored_folder = r"/mnt/disk3/liuyang/KidneyData/sc2leve2crops/704/restored"

    process_pipeline(
        crop_folder, txt_folder, wsi_folder, output_folder,
        json_folder=json_folder, restored_folder=restored_folder
    )
