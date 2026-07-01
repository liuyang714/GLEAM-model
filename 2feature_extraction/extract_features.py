import sys
import os
from functools import partial
from collections import defaultdict
import csv
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# conch lives in a subfolder, add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'conch'))
from open_clip_custom import create_model_from_pretrained

OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def interpolate_pos_embed(pos_embed, input_size, patch_size, num_extra_tokens=1):
    N, L, C = pos_embed.shape
    target_num_patches = (input_size // patch_size) ** 2
    if L == num_extra_tokens + target_num_patches:
        return pos_embed

    cls_pos = pos_embed[:, :num_extra_tokens]
    patch_pos = pos_embed[:, num_extra_tokens:]

    num_patches_orig = L - num_extra_tokens
    h_orig = w_orig = int(num_patches_orig ** 0.5)
    h_new = w_new = input_size // patch_size

    patch_pos = patch_pos.reshape(1, h_orig, w_orig, C).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(h_new, w_new), mode='bicubic', align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h_new * w_new, C)

    return torch.cat((cls_pos, patch_pos), dim=1)


def get_conch_patch_tokens(model, img):
    """Extract patch tokens from CONCH ViT (excluding cls token)."""
    visual_trunk = model.visual.trunk

    x = visual_trunk.patch_embed(img)
    B, H_grid, W_grid, C_embed = x.shape
    x = x.reshape(B, H_grid * W_grid, C_embed)

    num_extra_tokens = 0
    if hasattr(visual_trunk, 'cls_token') and visual_trunk.cls_token is not None:
        num_extra_tokens = 1
        cls_token = visual_trunk.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

    orig_pos_embed = visual_trunk.pos_embed
    patch_size = visual_trunk.patch_embed.proj.kernel_size[0]
    input_image_size = img.shape[-1]
    interpolated_pos_embed = interpolate_pos_embed(
        orig_pos_embed, input_image_size, patch_size, num_extra_tokens
    )
    x = x + interpolated_pos_embed

    for block in visual_trunk.blocks:
        x = block(x)
    x = visual_trunk.norm(x)

    return x[:, num_extra_tokens:]


def load_conch(target_img_size=512):
    ckpt_path = 'conch_kp.bin'  # TODO: change to your conch_kp.bin path,https://huggingface.co/MahmoodLab/CONCH
    model, _ = create_model_from_pretrained("conch_ViT-B-16", ckpt_path)
    model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    model = model.to(device).eval()

    img_transforms = transforms.Compose([
        transforms.Resize((target_img_size, target_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(OPENAI_MEAN, OPENAI_STD),
    ])
    return model, img_transforms


def extract_patient_features(image_paths, model, img_transforms):
    """Extract patch tokens for all images of one patient. Returns [N, num_patches, 768]."""
    patch_tokens_list, image_names = [], []
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc='Extract'):
            img = Image.open(img_path).convert('RGB')
            img = img_transforms(img).unsqueeze(0).to(device, non_blocking=True)
            tokens = get_conch_patch_tokens(model, img)   # [1, num_patches, 768]
            patch_tokens_list.append(tokens[0])
            image_names.append(os.path.basename(img_path))
    return torch.stack(patch_tokens_list), image_names    # [N, num_patches, 768]


def main(args):
    # group images by patient ID (filename format: patientID-number.png)
    patient_groups = defaultdict(list)
    for file_name in os.listdir(args.image_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            patient_id = file_name.split('-')[0]
            patient_groups[patient_id].append(os.path.join(args.image_dir, file_name))

    print(f"Patients: {len(patient_groups)}")

    model, img_transforms = load_conch(args.target_patch_size)
    os.makedirs(args.feat_dir, exist_ok=True)

    for patient_id, image_list in patient_groups.items():
        image_list = sorted(image_list)
        print(f"\n{patient_id}  ({len(image_list)} images)")

        tokens, names = extract_patient_features(image_list, model, img_transforms)
        torch.save({'tokens': tokens, 'names': names},
                   os.path.join(args.feat_dir, f'{patient_id}.pt'))

        with open(os.path.join(args.feat_dir, f'{patient_id}.csv'), 'w', newline='') as f:
            csv.writer(f).writerow(names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--target_patch_size', type=int, default=512)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main(args)
