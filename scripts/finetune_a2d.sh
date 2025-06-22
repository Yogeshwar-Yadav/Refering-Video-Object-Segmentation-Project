python3 main.py -c ./configs/a2d_sentences.yaml -rm train -ng 8 -epochs 20 \
-pw "/mnt/data_16TB/lzy23/pretrained/pretrained_coco/coco_1/best_pth.tar" --version "finetune_a2d_2" \
--lr_drop 20 -bs 1 -ws 8 --backbone "video-swin-t" \
-bpp "/home/nazir/NeurIPS2023_SOC/pretrained/video_swin_base.pth"

#finetune a2d, NOTE the number gpu lr_drop