export CUDA_VISIBLE_DEVICES=5
python demo_video.py -c /home/nazir/NeurIPS2023_SOC/configs/a2d_sentences.yaml -rm test --backbone "video-swin-b" \
-bpp "/home/nazir/NeurIPS2023_SOC/pretrained/video_swin_base.pth" \
-ckpt "/home/nazir/NeurIPS2023_SOC/checkpoint/a2d.pth.tar" \
--video_dir "/home/nazir/NeurIPS2023_SOC/uploads/testvideo.mp4"