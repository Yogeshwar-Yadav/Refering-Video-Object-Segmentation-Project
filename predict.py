# import torch
# from tqdm import tqdm
# import os
# from utils import flatten_temporal_batch_dims
# import torch.nn.functional as F
# import torchvision.transforms as transforms 
# import numpy as np
# from PIL import Image
# import cv2 as cv2

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# colors = [[212, 255, 127], [193,182,255], [106,106,255], [255, 206, 135]]

# def to_device(sample, device):
#     if isinstance(sample, torch.Tensor):
#         sample = sample.to(device)
#     elif isinstance(sample, tuple) or isinstance(sample, list):
#         sample = [to_device(s, device) for s in sample]
#     elif isinstance(sample, dict):
#         sample = {k: to_device(v, device) for k, v in sample.items()}
#     return sample

# @torch.no_grad()
# def predict(model, data_loader_val, device, postprocessor, output_dir):
#     model.eval()
#     for batch_dict in tqdm(data_loader_val):
#         print("Batch dictionary structure:")
#         print(f"Type of batch_dict: {type(batch_dict)}")
#         print(f"Shape of batch_dict['image']: {batch_dict['image'].shape}")
#         # Safely inspect batch content
#         if 'image' in batch_dict:
#             print("Image slice:", batch_dict['image'][0, :, :5, :5])  # show first 5x5 pixels per channel
#         if 'text' in batch_dict:
#             print("Text input:", batch_dict['text'])


#         print(batch_dict)  # Print the entire batch dictionary
#         # print(f"Type of batch_dict: {type(batch_dict)}")
#         print("Keys in batch dictionary:", batch_dict.keys())  # Print keys in the dictionary

#         # Check if 'samples' key exists in batch_dict
#         if 'samples' not in batch_dict:
#             raise KeyError("'samples' key is missing in batch_dict. Please check your dataset/dataloader.")

#         # Debug statement to check shape of 'samples'
#         print("Shape of 'samples':", batch_dict['samples'].shape)

#         predictions = []
#         samples = batch_dict['samples'].to(device)
#         targets = to_device(batch_dict['targets'], device)
#         text_queries = batch_dict['text_queries']

#         # Keep only valid targets (frames which are annotated):
#         valid_indices = torch.tensor([i for i, t in enumerate(targets) if None not in t]).to(device)
#         targets = [targets[i] for i in valid_indices.tolist()]
#         outputs = model(samples, valid_indices, text_queries)
#         outputs.pop('aux_outputs', None)

#         outputs, targets = flatten_temporal_batch_dims(outputs, targets)
#         processed_outputs = postprocessor(outputs,
#                                           resized_padded_sample_size=samples.tensors.shape[-2:],
#                                           resized_sample_sizes=[t['size'] for t in targets],
#                                           orig_sample_sizes=[t['orig_size'] for t in targets])
#         image_ids = [t['image_id'] for t in targets]

#         # Derive folder name from image_id – adjust as needed
#         image_id_example = image_ids[0]  # e.g., "video01_frame003"
#         folder_name = image_id_example.split('_')[0]  # --> "video01"
#         save_folder = os.path.join(output_dir, folder_name)
#         os.makedirs(save_folder, exist_ok=True)

#         for p, image_id in zip(processed_outputs, image_ids):
#             value, index = p['scores'].max(dim=0)
#             masks = p['masks'][index]
#             predictions.append({
#                 'image_id': image_id,
#                 'segmentation': masks,
#                 'score': value.item()
#             })

#         # Loop through each valid image and its corresponding prediction
#         images = torch.index_select(samples.tensors, 0, valid_indices)  # [B, C, H, W]

#         for i, (img_tensor, target, pred) in enumerate(zip(images, targets, predictions)):
#             # Resize image to original size
#             target_size = target['orig_size']
#             if isinstance(target_size, list):
#                 target_size = tuple(target_size)

#             img_tensor = img_tensor.unsqueeze(0)
#             img_resized = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)[0]

#             # Convert to image
#             image = img_resized.permute(1, 2, 0)  # [H, W, C]
#             image = (image * torch.tensor(std, device=device) + torch.tensor(mean, device=device)).clamp(0, 1)
#             image = (image.cpu().numpy() * 255).astype(np.uint8)

#             img = image.copy()

#             # Get the mask
#             mask = pred['segmentation'][0].cpu().numpy()  # [H, W]
#             overlay_color = np.full((mask.shape[0], mask.shape[1], 3), colors[i % len(colors)], dtype=np.uint8)
#             overlay_mask = np.zeros_like(img)
#             overlay_mask[mask.astype(bool)] = overlay_color[mask.astype(bool)]

#             img = cv2.addWeighted(img, 1.0, overlay_mask, 0.5, 0)

#             output_name = pred['image_id'] + '.jpg'
#             output_path = os.path.join(save_folder, output_name)
#             print(f"[INFO] Saving visualized frame to {output_path}")
#             Image.fromarray(img).convert('RGB').save(output_path)



import torch
from tqdm import tqdm
import os
from utils import flatten_temporal_batch_dims
import torch.nn.functional as F
import torchvision.transforms as transforms 
import numpy as np
from PIL import Image
import cv2 as cv2

# Mean and std values for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
colors = [[212, 255, 127], [193, 182, 255], [106, 106, 255], [255, 206, 135]]

# Function to move tensors to the correct device
def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        sample = sample.to(device)
    elif isinstance(sample, (tuple, list)):
        sample = [to_device(s, device) for s in sample]
    elif isinstance(sample, dict):
        sample = {k: to_device(v, device) for k, v in sample.items()}
    return sample

# Define the predict function
@torch.no_grad()
@torch.no_grad()
def predict(model, data_loader_val, device, postprocessor, output_dir, tokenizer, text_query='monkey'):
    print("🔍 predict() function called.")
    model.eval()

    all_predictions = []
    all_input_frames = []

    for batch_idx, batch_dict in enumerate(tqdm(data_loader_val)):
        print(f"\n📦 Processing batch {batch_idx+1}")
        print(f"Type of batch_dict: {type(batch_dict)}")
        print(f"Keys in batch_dict: {batch_dict.keys()}")

        if 'image' in batch_dict:
            print(f"✅ Found 'image' with shape: {batch_dict['image'].shape}")
            print("Sample pixel values:", batch_dict['image'][0, :, :5, :5])
        else:
            print("❌ 'image' key missing in batch_dict")
            continue

        if 'text' in batch_dict:
            print(f"📝 Text input (before tokenization): {batch_dict['text']}")
        else:
            print("❌ 'text' key missing in batch_dict")

        if 'targets' not in batch_dict:
            print("⚠️ [WARNING] 'targets' not found in batch_dict. Skipping this batch.")
            continue

        if 'samples' in batch_dict:
            print(f"Shape of 'samples': {batch_dict['samples'].shape}")
        else:
            print("⚠️ 'samples' key missing. Continuing...")

        # --- Preprocess Text ---
        batch_dict = preprocess_text(batch_dict, tokenizer, device)
        if 'text' in batch_dict:
            print(f"🧠 Encoded text shape: {batch_dict['text'].shape}")

        # --- Prepare Samples for Model ---
        samples = {
            "image": batch_dict["image"].to(device),
            "text": batch_dict["text"].to(device),
        }

        # --- Extract Valid Targets ---
        targets = batch_dict.get('targets', None)
        if targets is not None:
            print(f"📌 Targets found: {len(targets)}")
            valid_indices = torch.tensor(
                [i for i, t in enumerate(targets) if isinstance(t, dict) and None not in t.values()]
            ).to(device)
            if valid_indices.numel() == 0:
                print("⚠️ No valid targets found in this batch. Skipping.")
                continue
            targets = [targets[i] for i in valid_indices.tolist()]
        else:
            print("⚠️ No targets found in batch_dict.") 
            valid_indices = None

        # --- Forward Pass ---
        try:
            text_queries = batch_dict['text_queries']
        except KeyError:
            print("❌ Missing 'text_queries' in batch_dict")
            continue

        outputs = model(samples, valid_indices, text_queries)
        if outputs is None:
            print("❌ Model returned None.")
            continue

        outputs.pop('aux_outputs', None)
        outputs, targets = flatten_temporal_batch_dims(outputs, targets)

        print("✅ Model outputs obtained.")
        processed_outputs = postprocessor(
            outputs,
            resized_padded_sample_size=samples["image"].shape[-2:],
            resized_sample_sizes=[t['size'] for t in targets] if targets else [],
            orig_sample_sizes=[t['orig_size'] for t in targets] if targets else [],
        )

        image_ids = [t['image_id'] for t in targets] if targets else []
        if not image_ids:
            print("⚠️ No image IDs found in targets.")
            continue

        folder_name = image_ids[0].split('_')[0]
        save_folder = os.path.join(output_dir, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        predictions = []
        for p, image_id in zip(processed_outputs, image_ids):
            if 'scores' not in p or 'masks' not in p:
                print(f"⚠️ Missing 'scores' or 'masks' in postprocessed output for image {image_id}.")
                continue
            value, index = p['scores'].max(dim=0)
            masks = p['masks'][index]
            predictions.append({
                'image_id': image_id,
                'segmentation': masks,
                'score': value.item()
            })

        images = torch.index_select(samples["image"], 0, valid_indices) if valid_indices is not None else samples["image"]

        for i, (img_tensor, target, pred) in enumerate(zip(images, targets, predictions)):
            if 'orig_size' not in target:
                print(f"⚠️ Missing 'orig_size' in target: {target}")
                continue

            target_size = tuple(target['orig_size']) if isinstance(target['orig_size'], list) else target['orig_size']
            img_tensor = img_tensor.unsqueeze(0)
            img_resized = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)[0]

            # Convert tensor to image
            image = img_resized.permute(1, 2, 0)
            image = (image * torch.tensor(std, device=device) + torch.tensor(mean, device=device)).clamp(0, 1)
            image = (image.cpu().numpy() * 255).astype(np.uint8)
            img = image.copy()

            mask = pred['segmentation'][0].cpu().numpy()
            overlay_color = np.full((mask.shape[0], mask.shape[1], 3), colors[i % len(colors)], dtype=np.uint8)
            overlay_mask = np.zeros_like(img)
            overlay_mask[mask.astype(bool)] = overlay_color[mask.astype(bool)]
            img = cv2.addWeighted(img, 1.0, overlay_mask, 0.5, 0)

            output_name = pred['image_id'] + '.jpg'
            output_path = os.path.join(save_folder, output_name)
            print(f"[📸] Saving visualized frame to {output_path}")
            Image.fromarray(img).convert('RGB').save(output_path)

        all_predictions.extend(predictions)
        all_input_frames.extend([t['image_id'] for t in targets])

    print("✅ predict() finished successfully.")
    return all_predictions, all_input_frames


# Helper function to handle text preprocessing
# def preprocess_text(batch_dict, tokenizer, device):
#     if isinstance(batch_dict["text"], list):
#         # Tokenize text using the tokenizer
#         encoded_text = tokenizer(batch_dict["text"], padding=True, truncation=True, return_tensors="pt")
        
#         # Debugging: Print tokenization result
#         print(f"Encoded Text: {encoded_text}")

#         # Move tokenized text to the appropriate device
#         batch_dict["text"] = encoded_text["input_ids"].to(device)
    
#     return batch_dict


def preprocess_text(batch_dict, tokenizer, device):
    """
    Tokenizes the 'text' entry in the batch_dict using the given tokenizer,
    with 'monkey' as the default if the text is missing or empty.
    Moves tensors to the specified device.
    """
    raw_texts = batch_dict.get('text', 'monkey')

    # Fallback if None or empty string/list
    if not raw_texts or (isinstance(raw_texts, list) and all(t.strip() == '' for t in raw_texts)):
        raw_texts = ['monkey']
    elif isinstance(raw_texts, str):
        raw_texts = [raw_texts]

    # Tokenize text
    encoding = tokenizer(
        raw_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Move tensors to device
    batch_dict['text'] = encoding['input_ids'].to(device)

    if 'attention_mask' in encoding:
        batch_dict['attention_mask'] = encoding['attention_mask'].to(device)

    return batch_dict


            
            


    