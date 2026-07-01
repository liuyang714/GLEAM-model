"""
WSI -> Level 2 -> 400x400 patches (end-to-end).
Step 1: Convert SVS/NDPI to Level 2 downsampled image.
Step 2: Sliding window crop with 80px overlap, zero-pad at edges.
"""
import os
import openslide
from PIL import Image, ImageOps
from tqdm import tqdm


# ---------- Step 1: WSI -> Level 2 ----------

def wsi_to_level2(slide, filename):
    """Convert an OpenSlide object to a Level 2 RGB PIL Image."""
    if filename.lower().endswith('.svs'):
        level = 2
        return slide.read_region(
            (0, 0), level, slide.level_dimensions[level]
        ).convert("RGB")
    else:
        # NDPI: approximate 1/4 per axis -> 1/16 area
        w, h = slide.dimensions
        return slide.get_thumbnail((w // 16, h // 16)).convert("RGB")


# ---------- Step 2: Crop ----------

def crop_image(img, image_name, output_folder, crop_size=400, overlap=80):
    """Crop a single image into overlapping patches with zero-padding."""
    img_width, img_height = img.size
    step = crop_size - overlap
    count = 0

    for y in range(0, img_height, step):
        for x in range(0, img_width, step):
            crop_box = (x, y, min(x + crop_size, img_width), min(y + crop_size, img_height))
            cropped_img = img.crop(crop_box)

            pad_right = max(0, crop_size - (crop_box[2] - crop_box[0]))
            pad_bottom = max(0, crop_size - (crop_box[3] - crop_box[1]))
            if pad_right > 0 or pad_bottom > 0:
                cropped_img = ImageOps.expand(cropped_img, (0, 0, pad_right, pad_bottom), fill=(0, 0, 0))

            patch_name = f"{os.path.splitext(image_name)[0]}_{x}_{y}.jpg"
            output_path = os.path.join(output_folder, patch_name)
            if not os.path.exists(output_path):
                cropped_img.save(output_path)
                count += 1

    return count


# ---------- End-to-end pipeline ----------

def process_wsi(input_folder, level2_folder, crop_folder, crop_size=400, overlap=80):
    os.makedirs(level2_folder, exist_ok=True)
    os.makedirs(crop_folder, exist_ok=True)

    exts = ('.svs', '.ndpi')
    wsi_files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]

    if not wsi_files:
        print(f"No .svs/.ndpi files found in {input_folder}")
        return

    print(f"Found {len(wsi_files)} WSI files")
    total_patches = 0

    for filename in tqdm(wsi_files, desc="Processing"):
        level2_name = f"{os.path.splitext(filename)[0]}_level2.png"
        level2_path = os.path.join(level2_folder, level2_name)

        try:
            # Step 1: convert to level 2
            if not os.path.exists(level2_path):
                slide = openslide.OpenSlide(os.path.join(input_folder, filename))
                img = wsi_to_level2(slide, filename)
                img.save(level2_path, "PNG")
                slide.close()
            else:
                img = Image.open(level2_path).convert("RGB")

            # Step 2: crop into patches
            n = crop_image(img, filename, crop_folder, crop_size, overlap)
            total_patches += n
            img.close()

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print(f"\nDone! Generated {total_patches} patches from {len(wsi_files)} WSIs.")


if __name__ == "__main__":
    input_folder = r""
    level2_folder = r""
    crop_folder = r""
    process_wsi(input_folder, level2_folder, crop_folder)
