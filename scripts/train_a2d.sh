python3 main.py -c ./configs/a2d_sentences.yaml -rm train -ng 2 --epochs 40 \
--version "a2d" --lr_drop 15 -ws 8 -bs 2 --backbone "video-swin-t" \
-bpp "/home/nazir/NeurIPS2023_SOC/pretrained/video_swin_base.pth"
