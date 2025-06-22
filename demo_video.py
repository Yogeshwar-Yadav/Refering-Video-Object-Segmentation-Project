# import ruamel.yaml
# import torchvision.transforms as T
# # from tools.colormap import colormap
# from datasets.transforms import RandomResize
# from models import build_model
# import os
# from PIL import Image
# from ruamel.yaml import YAML
# import torch
# import misc as utils
# import numpy as np
# import torch.nn.functional as F
# import random
# import argparse
# from torchvision.io import read_video
# import torchvision.transforms.functional as Func
# import shutil

# size_transform = RandomResize(sizes=[360], max_size=640)
# transform = T.Compose([
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# color = np.array([0, 0, 255]).astype('uint8')
# def vis_add_mask(img, mask, color):
#     source_img = np.asarray(img).copy()
#     origin_img = np.asarray(img).copy()
#     color = np.array(color)

#     mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
#     mask = mask > 0.5

#     origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
#     origin_img = Image.fromarray(origin_img)
#     source_img = Image.fromarray(source_img)
#     mask = Image.fromarray(mask)
#     return origin_img, source_img, mask


# def main(config):
#     print(config.backbone_pretrained)
#     model, _, _ = build_model(config) 
#     device = config.device
#     model.to(device)

#     if config.checkpoint_path is not None:
#         checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
#         state_dict = checkpoint["model_state_dict"]
#         missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
#         unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
#         if len(missing_keys) > 0:
#             print('Missing Keys: {}'.format(missing_keys))
#         if len(unexpected_keys) > 0:
#             print('Unexpected Keys: {}'.format(unexpected_keys))
#     else:
#         print("pleas specify the checkpoint")

#     model.eval()
#     video_dir = config.video_dir
#     ver_dir = video_dir.split("/")[-1]
#     ver_dir = ver_dir.split(".")[0]
#     save_dir = os.path.join('/home/nazir/NeurIPS2023_SOC/result', ver_dir)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)

#     output_dir = os.path.join(save_dir, "SOC", "visual")
#     source_dir = os.path.join(save_dir, "SOC", "source")
#     mask_dir = os.path.join(save_dir, "SOC", "mask")
    
#     exp = "a cat sitting on the tree"
#     with open(os.path.join(save_dir, "expression.txt"), 'w') as f:
#         f.write(exp + "\n")
    
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     if not os.path.exists(source_dir):
#         os.makedirs(source_dir)
#     if not os.path.exists(mask_dir):
#         os.makedirs(mask_dir)


#     video_frames, _, _ = read_video(video_dir, pts_unit='sec')  # (T, H, W, C)
#     source_frames= []
#     imgs = []
#     print("length",len(video_frames))
#     num_frame = [12,62,92,132,142,152]
#     for i in range(0,len(video_frames),5):
#         source_frame = Func.to_pil_image(video_frames[i].permute(2, 0, 1))
#         source_frames.append(source_frame) #(C H W)
#     for frame in source_frames:
#         origin_w, origin_h = frame.size
#         img, _ = size_transform(frame)
#         imgs.append(transform(img)) # list[img]
    
#     frame_length = len(imgs)

#     imgs = torch.stack(imgs, dim=0) # [video_len, 3, H, W]
#     samples = utils.nested_tensor_from_videos_list([imgs]).to(config.device)
#     img_h, img_w = imgs.shape[-2:]
#     size = torch.as_tensor([int(img_h), int(img_w)]).to(config.device)
#     targets = [[{"size": size}] for _ in range(frame_length)]
#     valid_indices = None


#     with torch.no_grad():
#         outputs = model(samples, valid_indices, [exp], targets)
    
#     pred_logits = outputs["pred_cls"][:, 0, ...] # [t, q, k]
#     pred_masks = outputs["pred_masks"][:, 0, ...]   # [t, q, h, w] 

#     pred_scores = pred_logits.sigmoid() # [t, q, k]
#     pred_scores = pred_scores.mean(0)   # [q, k]
#     max_scores, _ = pred_scores.max(-1) # [q,]
#     _, max_ind = max_scores.max(-1)     # [1,]

#     max_inds = max_ind.repeat(frame_length)
#     pred_masks = pred_masks[range(frame_length), max_inds, ...] # [t, h, w]
#     pred_masks = pred_masks.unsqueeze(0)

#     pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
#     pred_masks = (pred_masks.sigmoid() > 0.5).squeeze(0).detach().cpu().numpy() 

#     color = [255, 144, 30]

#     for t, img in enumerate(source_frames):
#         origin_img, source_img, mask = vis_add_mask(img, pred_masks[t], color)
#         # save_postfix = img_path.replace(".jpg", ".png")
#         origin_img.save(os.path.join(output_dir, f'{t}.png'))
#         source_img.save(os.path.join(source_dir, f'{t}.png'))
#         mask.save(os.path.join(mask_dir, f'{t}.png'))




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('DEMO script')
#     parser.add_argument('--config_path', '-c',
#                         default='./configs/refer_youtube_vos.yaml',                        help='path to configuration file')
#     parser.add_argument('--running_mode', '-rm', choices=['train', 'test', 'pred', 'resume_train'],
#                         default='test',
#                         help="mode to run, either 'train' or 'eval'")
#     parser.add_argument("--backbone", type=str, required=False,
#                         help="the backbone name")
#     parser.add_argument("--backbone_pretrained_path", "-bpp", type=str, required=False,
#                         help="the backbone_pretrained_path")
#     parser.add_argument('--checkpoint_path', '-ckpt', type=str, default='',
#                             help='the finetune refytbs checkpoint_path')
#     parser.add_argument("--video_dir", type=str, required=False)
#     parser.add_argument("--device", default="cuda")
#     args = parser.parse_args()
    
#     with open(args.config_path) as f:
#         yaml = YAML(typ='safe', pure = True)
#         config = yaml.load(f)
#         # config = ruamel.yaml.safe_load(f)
#     # config = {k: v['value'] for k, v in config.items()}
#     config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
#     config = {**config, **vars(args)}
#     config = argparse.Namespace(**config)
    
#     main(config)


































# import os
# import argparse
# import random
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T
# import torchvision.transforms.functional as Func
# from torchvision.io import read_video
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
# from ruamel.yaml import YAML
# import misc as utils
# import shutil

# # Dataset transforms (keep your original imports)
# from datasets.transforms import RandomResize
# from models import build_model

# # Transforms (keep your original transforms)
# size_transform = RandomResize(sizes=[360], max_size=640)
# transform = T.Compose([
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# def setup_for_distributed(is_master):
#     """Disables printing when not in master process"""
#     import builtins as __builtin__
#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)

#     __builtin__.print = print

# def init_distributed_mode(args):
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True
#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}'.format(
#         args.rank, args.dist_url), flush=True)
#     dist.init_process_group(backend=args.dist_backend, 
#                           init_method=args.dist_url,
#                           world_size=args.world_size, 
#                           rank=args.rank)
#     dist.barrier()
#     setup_for_distributed(args.rank == 0)

# def vis_add_mask(img, mask, color):
#     """Your original visualization function"""
#     source_img = np.asarray(img).copy()
#     origin_img = np.asarray(img).copy()
#     color = np.array(color)

#     mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')
#     mask = mask > 0.5

#     origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
#     origin_img = Image.fromarray(origin_img)
#     source_img = Image.fromarray(source_img)
#     mask = Image.fromarray(mask)
#     return origin_img, source_img, mask

# def main(config):
#     if not hasattr(config, "distributed"):
#         config.distributed = False
#     # Initialize distributed mode if available
#     if torch.cuda.device_count() > 1:
#         init_distributed_mode(config)
    
#     # Build model
#     model, _, _ = build_model(config)
    
#     # Device handling
#     device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
#     # Move model to device and wrap with DDP if multiple GPUs
#     model.to(device)
#     if config.distributed:
#         model = DDP(model, device_ids=[config.gpu])
#     elif torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
#         model = torch.nn.DataParallel(model)
    
#     # Load checkpoint
#     if config.checkpoint_path is not None:
#         checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
#         state_dict = checkpoint["model_state_dict"]
        
#         # Handle DataParallel/DDP state dict
#         if isinstance(model, (torch.nn.DataParallel, DDP)):
#             model.module.load_state_dict(state_dict, strict=False)
#         else:
#             model.load_state_dict(state_dict, strict=False)

#         missing_keys = load_info.missing_keys    
#         unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

#         if missing_keys:
#             print('Missing Keys:', missing_keys)
#         if unexpected_keys:
#             print('Unexpected Keys:', unexpected_keys)
#         if len(missing_keys) > 0:
#             print('Missing Keys: {}'.format(missing_keys))
#         if len(unexpected_keys) > 0:
#             print('Unexpected Keys: {}'.format(unexpected_keys))
#     else:
#         print("Please specify the checkpoint")

#     model.eval()
    
#     # Directory setup
#     video_dir = config.video_dir
#     ver_dir = os.path.splitext(os.path.basename(video_dir))[0]
#     save_dir = os.path.join('/home/nazir/NeurIPS2023_SOC/result', ver_dir)
#     os.makedirs(save_dir, exist_ok=True)

#     output_dir = os.path.join(save_dir, "SOC", "visual")
#     source_dir = os.path.join(save_dir, "SOC", "source")
#     mask_dir = os.path.join(save_dir, "SOC", "mask")
    
#     exp = "a cat sitting on the tree"
#     with open(os.path.join(save_dir, "expression.txt"), 'w') as f:
#         f.write(exp + "\n")
    
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(source_dir, exist_ok=True)
#     os.makedirs(mask_dir, exist_ok=True)

#     # Load and process video
#     video_frames, _, _ = read_video(video_dir, pts_unit='sec')
#     source_frames = []
#     imgs = []
    
#     print("Number of frames:", len(video_frames))
    
#     # Sample frames (adjust sampling as needed)
#     for i in range(0, len(video_frames), 5):
#         source_frame = Func.to_pil_image(video_frames[i].permute(2, 0, 1))
#         source_frames.append(source_frame)
    
#     for frame in source_frames:
#         origin_w, origin_h = frame.size
#         img, _ = size_transform(frame)
#         imgs.append(transform(img))
    
#     frame_length = len(imgs)
#     imgs = torch.stack(imgs, dim=0)
    
#     # Clear cache before processing
#     torch.cuda.empty_cache()
    
#     # Prepare inputs
#     samples = utils.nested_tensor_from_videos_list([imgs]).to(device)
#     img_h, img_w = imgs.shape[-2:]
#     size = torch.as_tensor([int(img_h), int(img_w)]).to(device)
#     targets = [[{"size": size}] for _ in range(frame_length)]
#     valid_indices = None

#     # Inference
#     with torch.no_grad():
#         outputs = model(samples, valid_indices, [exp], targets)
    
#     # Process outputs
#     pred_logits = outputs["pred_cls"][:, 0, ...]
#     pred_masks = outputs["pred_masks"][:, 0, ...]
    
#     pred_scores = pred_logits.sigmoid().mean(0)
#     max_scores, _ = pred_scores.max(-1)
#     _, max_ind = max_scores.max(-1)
    
#     max_inds = max_ind.repeat(frame_length)
#     pred_masks = pred_masks[range(frame_length), max_inds, ...]
#     pred_masks = pred_masks.unsqueeze(0)
    
#     pred_masks = F.interpolate(
#         pred_masks, 
#         size=(origin_h, origin_w), 
#         mode='bilinear', 
#         align_corners=False
#     )
#     pred_masks = (pred_masks.sigmoid() > 0.5).squeeze(0).detach().cpu().numpy()
    
#     # Visualization
#     color = [255, 144, 30]
#     for t, img in enumerate(source_frames):
#         origin_img, source_img, mask = vis_add_mask(img, pred_masks[t], color)
#         origin_img.save(os.path.join(output_dir, f'{t}.png'))
#         source_img.save(os.path.join(source_dir, f'{t}.png'))
#         mask.save(os.path.join(mask_dir, f'{t}.png'))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('DEMO script')
#     parser.add_argument('--config_path', '-c',
#                       default='./configs/a2d_sentences.yaml',
#                       help='path to configuration file')
#     parser.add_argument('--running_mode', '-rm', 
#                       choices=['train', 'test', 'pred', 'resume_train'],
#                       default='test',
#                       help="mode to run")
#     parser.add_argument("--backbone", type=str, required=False,
#                       help="the backbone name")
#     parser.add_argument("--backbone_pretrained_path", "-bpp", 
#                       type=str, required=False,
#                       help="the backbone_pretrained_path")
#     parser.add_argument('--checkpoint_path', '-ckpt', 
#                       type=str, default='',
#                       help='the finetune refytbs checkpoint_path')
#     parser.add_argument("--video_dir", type=str, required=False)
#     parser.add_argument("--device", default="cuda")
    
#     # Distributed training parameters
#     parser.add_argument('--world-size', default=1, type=int,
#                       help='number of nodes for distributed training')
#     parser.add_argument('--rank', default=0, type=int,
#                       help='node rank for distributed training')
#     parser.add_argument('--dist-url', default='env://',
#                       help='url used to set up distributed training')
#     parser.add_argument('--dist-backend', default='nccl',
#                       help='distributed backend')
#     parser.add_argument('--local_rank', default=0, type=int,
#                       help='local rank for distributed training')
    
#     args = parser.parse_args()
    
#     # Load config
#     with open(args.config_path) as f:
#         yaml = YAML(typ='safe', pure=True)
#         config = yaml.load(f)
#         config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v 
#                  for k, v in config.items()}
#         config = {**config, **vars(args)}
#         config = argparse.Namespace(**config)
    
#     main(config)












#######This below one need to be uncommented######








import utils
import os
import argparse
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as Func
from torchvision.io import read_video
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from ruamel.yaml import YAML
import misc as utils
import shutil
import numpy as np
import cv2
from transforms import size_transform, transform

from datasets.transforms import RandomResize
from models import build_model
from PIL import Image
from natsort import natsorted  
# Transforms
size_transform = RandomResize(sizes=[360], max_size=640)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)
















#######This above one need to be uncommented######














# def vis_add_mask(img, mask, color):
#     source_img = np.asarray(img).copy()
#     origin_img = np.asarray(img).copy()
#     color = np.array(color)

#     mask = mask.astype('uint8')
#     mask = mask > 0.5

#     origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
#     origin_img = Image.fromarray(origin_img.astype('uint8'))
#     source_img = Image.fromarray(source_img)
#     mask_img = Image.fromarray((mask * 255).astype('uint8'))
#     return origin_img, source_img, mask_img



























#######This below one need to be uncommented######













def vis_add_mask(img, mask, color):
    source_img = np.asarray(img).copy()
    origin_img = np.asarray(img).copy()
    color = np.array(color)

    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = mask.squeeze()

    # Convert mask to uint8
    mask = mask.astype('uint8')

    # Resize mask to match image dimensions
    resized_mask = cv2.resize(mask, (origin_img.shape[1], origin_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    resized_mask = resized_mask > 0.5  # convert to boolean mask

    # Apply color to masked area
    origin_img[resized_mask] = origin_img[resized_mask] * 0.5 + color * 0.5

    # Convert back to PIL images
    origin_img = Image.fromarray(origin_img.astype('uint8'))
    source_img = Image.fromarray(source_img)
    mask_img = Image.fromarray((resized_mask.astype('uint8') * 255))

    return origin_img, source_img, mask_img



def frames_to_video(frames_dir, output_path, fps=20):
    images = natsorted([img for img in os.listdir(frames_dir) if img.endswith(".jpg") or img.endswith(".png")])
    if not images:
        print(f"[ERROR] No frames found in {frames_dir}")
        return

    # Read the first frame to get width and height
    first_frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec here if needed
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(frames_dir, image))
        video_writer.write(frame)

    video_writer.release()
    print(f"[INFO] Video saved to {output_path}")











#######This above one need to be uncommented######










# def main(config):
#     if not hasattr(config, "distributed"):
#         config.distributed = False

#     # Distributed initialization
#     if torch.cuda.device_count() > 1 and config.distributed:
#         init_distributed_mode(config)

#     # Build and load model
#     model, _, _ = build_model(config)

#     device = torch.device(config.device if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     if config.distributed:
#         model = DDP(model, device_ids=[config.gpu])
#     elif torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
#         model = torch.nn.DataParallel(model)

#     # Load checkpoint
#     if config.checkpoint_path:
#         checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
#         state_dict = checkpoint.get("model_state_dict", checkpoint)

#         missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
#         unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

#         if missing_keys:
#             print(f'Missing Keys: {missing_keys}')
#         if unexpected_keys:
#             print(f'Unexpected Keys: {unexpected_keys}')
#     else:
#         print("❌ Please specify a checkpoint path with --checkpoint_path")
#         return

#     model.eval()

#     # Output directory
#     video_dir = config.video_dir
#     ver_dir = os.path.splitext(os.path.basename(video_dir))[0]
#     save_dir = os.path.join('/home/nazir/NeurIPS2023_SOC/result', ver_dir)
#     os.makedirs(save_dir, exist_ok=True)

#     output_dir = os.path.join(save_dir, "SOC", "visual")
#     source_dir = os.path.join(save_dir, "SOC", "source")
#     mask_dir = os.path.join(save_dir, "SOC", "mask")

#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(source_dir, exist_ok=True)
#     os.makedirs(mask_dir, exist_ok=True)

#     # Write expression
#     exp = "a cat sitting on the tree"
#     with open(os.path.join(save_dir, "expression.txt"), 'w') as f:
#         f.write(exp + "\n")

#     # Load video
#     video_frames, _, _ = read_video(video_dir, pts_unit='sec')
#     print("Number of frames:", len(video_frames))

#     # Limit to first 100 frames (or less if video is shorter)
#     max_frames = 100
#     video_frames = video_frames[:max_frames]

#     source_frames, imgs = [], []
#     for i in range(0, len(video_frames), 5):  # sample every 5th frame
#         pil_img = Func.to_pil_image(video_frames[i].permute(2, 0, 1))
#         source_frames.append(pil_img)
#         img_resized, _ = size_transform(pil_img)
#         imgs.append(transform(img_resized))

#     if not imgs:
#         print("❌ No frames selected from video. Check sampling interval.")
#         return

#     imgs = torch.stack(imgs).to(device)
#     frame_length = imgs.size(0)

#     torch.cuda.empty_cache()

#     # Prepare inputs
#     samples = utils.nested_tensor_from_videos_list([imgs]).to(device)
#     img_h, img_w = imgs.shape[-2:]
#     size_tensor = torch.as_tensor([img_h, img_w], device=device)
#     targets = [[{"size": size_tensor}] for _ in range(frame_length)]
#     valid_indices = None

#     with torch.cuda.amp.autocast():
#     # with torch.no_grad():
#         outputs = model(samples, valid_indices, [exp], targets)

#     pred_logits = outputs["pred_cls"][:, 0, ...]
#     pred_masks = outputs["pred_masks"][:, 0, ...]

#     pred_scores = pred_logits.sigmoid().mean(0)
#     max_scores, _ = pred_scores.max(-1)
#     _, max_ind = max_scores.max(-1)

#     pred_masks = pred_masks[range(frame_length), max_ind, ...]
#     pred_masks = pred_masks.unsqueeze(0)
#     pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode='bilinear', align_corners=False)
#     pred_masks = (pred_masks.sigmoid() > 0.5).squeeze(0).cpu().numpy()

#     color = [255, 144, 30]
#     for t, img in enumerate(source_frames):
#         origin_img, source_img, mask = vis_add_mask(img, pred_masks[t], color)
#         origin_img.save(os.path.join(output_dir, f'{t}.png'))
#         source_img.save(os.path.join(source_dir, f'{t}.png'))
#         mask.save(os.path.join(mask_dir, f'{t}.png'))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('DEMO script')
#     parser.add_argument('--config_path', '-c', default='./configs/a2d_sentences.yaml')
#     parser.add_argument('--running_mode', '-rm', choices=['train', 'test', 'pred', 'resume_train'], default='test')
#     parser.add_argument('--backbone', type=str, required=False)
#     parser.add_argument('--backbone_pretrained_path', '-bpp', type=str, required=False)
#     parser.add_argument('--checkpoint_path', '-ckpt', type=str, default='')
#     parser.add_argument('--video_dir', type=str, required=True)
#     parser.add_argument('--device', default='cuda')

#     # Distributed training parameters
#     parser.add_argument('--world-size', default=1, type=int)
#     parser.add_argument('--rank', default=0, type=int)
#     parser.add_argument('--dist-url', default='env://')
#     parser.add_argument('--dist-backend', default='nccl')
#     parser.add_argument('--local_rank', default=0, type=int)

#     args = parser.parse_args()

#     # Load YAML config
#     with open(args.config_path, 'r') as f:
#         yaml = YAML(typ='safe')
#         base_config = yaml.load(f)

#     config_dict = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in base_config.items()}
#     config_dict.update(vars(args))
#     config = argparse.Namespace(**config_dict)

#     main(config)


















































































#######This below one need to be uncommented######








def main(config):
    if not hasattr(config, "distributed"):
        config.distributed = False

    # Distributed initialization
    if torch.cuda.device_count() > 1 and config.distributed:
        init_distributed_mode(config)

    # Build and load model
    model, _, _ = build_model(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    if config.distributed:
        model = DDP(model, device_ids=[config.gpu])
    elif torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)

    # Load checkpoint
    if config.checkpoint_path:
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

        if missing_keys:
            print(f'Missing Keys: {missing_keys}')
        if unexpected_keys:
            print(f'Unexpected Keys: {unexpected_keys}')
    else:
        print("❌ Please specify a checkpoint path with --checkpoint_path")
        return

    model.eval()

    # Output directory
    video_dir = config.video_dir
    ver_dir = os.path.splitext(os.path.basename(video_dir))[0]
    save_dir = os.path.join('/home/nazir/NeurIPS2023_SOC/result', ver_dir)
    os.makedirs(save_dir, exist_ok=True)

    output_dir = os.path.join(save_dir, "SOC", "visual")
    source_dir = os.path.join(save_dir, "SOC", "source")
    mask_dir = os.path.join(save_dir, "SOC", "mask")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Write expression
    exp = "a cat sitting on the tree"
    with open(os.path.join(save_dir, "expression.txt"), 'w') as f:
        f.write(exp + "\n")

    # Load video
    video_frames, _, _ = read_video(video_dir, pts_unit='sec')
    print("Number of frames:", len(video_frames))

    # max_frames = 100
    # video_frames = video_frames[:max_frames]

    source_frames, imgs = [], []
    for i in range(0, len(video_frames), 10):
        pil_img = Func.to_pil_image(video_frames[i].permute(2, 0, 1))
        source_frames.append(pil_img)
        img_resized, _ = size_transform(pil_img)
        imgs.append(transform(img_resized))

    if not imgs:
        print("❌ No frames selected from video. Check sampling interval.")
        return

    imgs = torch.stack(imgs).to(device)
    frame_length = imgs.size(0)

    # Prepare inputs
    samples = utils.nested_tensor_from_videos_list([imgs]).to(device)
    img_h, img_w = imgs.shape[-2:]
    size_tensor = torch.as_tensor([img_h, img_w], device=device)
    targets = [[{"size": size_tensor}] for _ in range(frame_length)]
    valid_indices = None

    # Run model inference
    with torch.no_grad():
        # with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(samples, valid_indices, [exp], targets)

    pred_logits = outputs["pred_cls"][:, 0, ...]
    pred_masks = outputs["pred_masks"][:, 0, ...]

    pred_scores = pred_logits.sigmoid().mean(0)
    max_scores, _ = pred_scores.max(-1)
    _, max_ind = max_scores.max(-1)

    pred_masks = pred_masks[range(frame_length), max_ind, ...]
    pred_masks = pred_masks.unsqueeze(0)
    pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode='bilinear', align_corners=False)
    pred_masks = (pred_masks.sigmoid() > 0.5).squeeze(0).cpu().numpy()

    torch.cuda.empty_cache()

    color = [255, 144, 30]
    for t, img in enumerate(source_frames):
        origin_img, source_img, mask = vis_add_mask(img, pred_masks[t], color)
        origin_img.save(os.path.join(output_dir, f'{t}.png'))
        source_img.save(os.path.join(source_dir, f'{t}.png'))
        mask.save(os.path.join(mask_dir, f'{t}.png'))

        # ✅ Convert saved frames to video
    input_video_name = os.path.splitext(os.path.basename(config.video_dir))[0]
    frames_dir = output_dir  # This is where the masked frames are saved
    output_video_path = os.path.join(save_dir, "SOC", f"{input_video_name}_output.mp4")

    frames_to_video(frames_dir, output_video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DEMO script')
    parser.add_argument('--config_path', '-c', default='./configs/a2d_sentences.yaml')
    parser.add_argument('--running_mode', '-rm', choices=['train', 'test', 'pred', 'resume_train'], default='test')
    parser.add_argument('--backbone', type=str, required=False)
    parser.add_argument('--backbone_pretrained_path', '-bpp', type=str, required=False)
    parser.add_argument('--checkpoint_path', '-ckpt', type=str, default='')
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--device', default='cuda')

    # Distributed training parameters
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-url', default='env://')
    parser.add_argument('--dist-backend', default='nccl')
    parser.add_argument('--local_rank', default=0, type=int)

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        yaml = YAML(typ='safe')
        base_config = yaml.load(f)

    config_dict = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in base_config.items()}
    config_dict.update(vars(args))
    config = argparse.Namespace(**config_dict)

    main(config)








#######This above one need to be uncommented######































































