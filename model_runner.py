# import yaml
# import subprocess
# import os

# def run_model(video_path, text_path, output_dir):
#     print(f"[INFO] Running model on: {video_path} with text: {text_path}")

#     # Extract the base filename without extension
#     video_filename = os.path.basename(video_path)
#     video_name_no_ext = os.path.splitext(video_filename)[0]

#     # Paths
#     full_video_path = os.path.abspath(video_path)  # Full path for demo_video.py
#     full_text_path = os.path.abspath(text_path)    # Still included in case model uses it

#     # Write YAML config (with extra field for DeformTransformer to avoid crash)
#     config_data = {
#         'DATA': {
#             'VIDEO_PATH': {'value': full_video_path},
#             'TEXT_PATH': {'value': full_text_path}
#         },
#         'OUTPUT': {
#             'DIR': {'value': os.path.abspath(output_dir)}
#         },
#         'DeformTransformer': {
#             'd_model': 256,
#             'nheads': 8,
#             'enc_layers': 6,
#             'dec_layers': 6,
#             'dim_feedforward': 1024,
#             'dropout': 0.1,
#             'num_feature_levels': 4,
#             'num_queries': 300,
#             'dec_n_points': 4,  # Number of decoder points
#             'enc_n_points': 4,  # Number of encoder points
#             'two_stage': True    # Added: Set whether to use a two-stage transformer
#         },
#         'with_box_refine': True,
#         'use_checkpoint': False,
#         'num_classes': 42,
#         'rel_coord': True,  # ✅ Add this line
        
#         # Add the text encoder type
#         'text_encoder_type': 'roberta-base',  # Specify the type of encoder

#         # Add freeze_text_encoder attribute
#         'freeze_text_encoder': True,  # Set whether to freeze the text encoder or not

#         # Add mask_kernels_dim to avoid the error
#         'mask_kernels_dim': 256  # Default value, adjust according to the model's requirement
#     }

#     # Save YAML config to a file
#     config_path = 'temp_config.yaml'
#     with open(config_path, 'w') as f:
#         yaml.dump(config_data, f)

#     # Inference command
#     command = [
#         'python', 'demo_video.py',
#         '-c', config_path,
#         '-rm', 'test',
#         '--backbone', 'video-swin-b',
#         '-bpp', '/home/nazir/NeurIPS2023_SOC/pretrained/video_swin_base.pth',
#         '-ckpt', '/home/nazir/NeurIPS2023_SOC/checkpoint/a2d.pth.tar',
#         '--video_dir', full_video_path
#     ]

#     try:
#         subprocess.run(command, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Inference failed: {e}")
#         raise RuntimeError("Model inference process failed.")

#     # Get expected output video path based on model output directory structure
#     result_path = os.path.join(
#         output_dir, video_name_no_ext, 'SOC', f"{video_name_no_ext}_output.mp4"
#     )

#     return result_path

# import yaml
# import subprocess
# import os

# def run_model(video_path, text_path, output_dir):
#     print(f"[INFO] Running model on: {video_path} with text: {text_path}")

#     # Extract the base filename without extension
#     video_filename = os.path.basename(video_path)
#     video_name_no_ext = os.path.splitext(video_filename)[0]

#     # Paths
#     full_video_path = os.path.abspath(video_path)  # Full path for demo_video.py
#     full_text_path = os.path.abspath(text_path)    # Still included in case model uses it

#     # Write YAML config (with extra field for DeformTransformer to avoid crash)
#     config_data = {
#         'DATA': {
#             'VIDEO_PATH': {'value': full_video_path},
#             'TEXT_PATH': {'value': full_text_path}
#         },
#         'OUTPUT': {
#             'DIR': {'value': os.path.abspath(output_dir)}
#         },
#         'DeformTransformer': {
#             'd_model': 256,
#             'nheads': 8,
#             'enc_layers': 6,
#             'dec_layers': 6,
#             'dim_feedforward': 1024,
#             'dropout': 0.1,
#             'num_feature_levels': 4,
#             'num_queries': 300,
#             'dec_n_points': 4,  # Number of decoder points
#             'enc_n_points': 4,  # Number of encoder points
#             'two_stage': True    # Added: Set whether to use a two-stage transformer
#         },
#         'with_box_refine': True,
#         'use_checkpoint': False,
#         'num_classes': 42,
#         'rel_coord': True,  # ✅ Add this line
        
#         # Add the text encoder type
#         'text_encoder_type': 'roberta-base',  # Specify the type of encoder

#         # Add freeze_text_encoder attribute
#         'freeze_text_encoder': True,  # Set whether to freeze the text encoder or not

#         # Add mask_kernels_dim to avoid the error
#         'mask_kernels_dim': 256,  # Default value, adjust according to the model's requirement
        
#         # Add backbone attribute (required by SOC model)
#         'backbone': 'video-swin-b',  # Specify the backbone you want to use
        
#         # Add pretrained backbone model path (required by SOC model)
#         'backbone_pretrained_path': '/home/nazir/NeurIPS2023_SOC/pretrained/video_swin_base.pth'  # Specify the pretrained model path
#     }

#     # Save YAML config to a file
#     config_path = 'temp_config.yaml'
#     with open(config_path, 'w') as f:
#         yaml.dump(config_data, f)

#     # Inference command
#     command = [
#         'python', 'demo_video.py',
#         '--config_path', config_path,
#         '--checkpoint_path', '/home/nazir/NeurIPS2023_SOC/checkpoint/a2d.pth.tar',
#         '--device', 'cuda'  # Or 'cpu' depending on your hardware
#     ]

#     try:
#         subprocess.run(command, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Inference failed: {e}")
#         raise RuntimeError("Model inference process failed.")

#     # Get expected output video path based on model output directory structure
#     result_path = os.path.join(
#         output_dir, video_name_no_ext, 'SOC', f"{video_name_no_ext}_output.mp4"
#     )

#     return result_path










































# import subprocess
# import os

# def run_model(video_path, text_path, output_dir):
#     print(f"[INFO] Running model on: {video_path} with text: {text_path}")

#     # Extract the base filename without extension
#     video_filename = os.path.basename(video_path)
#     video_name_no_ext = os.path.splitext(video_filename)[0]

#     # Paths
#     full_video_path = os.path.abspath(video_path)  # Full path for demo_video.py
#     full_text_path = os.path.abspath(text_path)    # Still included in case model uses it

#     # Prepare the environment variable and the command
#     command = f"""
#     export CUDA_VISIBLE_DEVICES=5 && python demo_video.py \
#     --config_path /home/nazir/NeurIPS2023_SOC/configs/a2d_sentences.yaml \
#     --checkpoint_path /home/nazir/NeurIPS2023_SOC/checkpoint/a2d.pth.tar \
#     --device cuda \
#     --video_dir /home/nazir/NeurIPS2023_SOC/uploads/testvideo.mp4
#     """

#     try:
#         # Run the command
#         subprocess.run(command, check=True, shell=True)
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Inference failed: {e}")
#         raise RuntimeError("Model inference process failed.")

#     # Get expected output video path based on model output directory structure
#     result_path = os.path.join(
#         output_dir, video_name_no_ext, 'SOC', f"{video_name_no_ext}_output.mp4"
#     )

#     return result_path





































import subprocess
import os

def run_model(video_path, text_path, output_dir, backbone, bpp, ckpt):
    print(f"[INFO] Running model on: {video_path} with text: {text_path}")

    # Extract the base filename without extension
    video_filename = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_filename)[0]

    # Paths
    full_video_path = os.path.abspath(video_path)  # Full path for demo_video.py
    full_text_path = os.path.abspath(text_path)    # Still included in case model uses it

    # Prepare the environment variable and the command
    command = f"""
    export CUDA_VISIBLE_DEVICES=5 && python demo_video.py \
    -c /home/nazir/NeurIPS2023_SOC/configs/a2d_sentences.yaml \
    -rm test \
    --backbone "{backbone}" \
    -bpp "{bpp}" \
    -ckpt "{ckpt}" \
    --video_dir "{full_video_path}"
    """

    try:
        # Run the command
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Inference failed: {e}")
        raise RuntimeError("Model inference process failed.")

    # Get expected output video path based on model output directory structure
    result_path = os.path.join(
        output_dir, video_name_no_ext, 'SOC', f"{video_name_no_ext}_output.mp4"
    )

    return result_path
