# SOC: Semantic-Assisted Object Cluster for Referring Video Object Segmentation
This project implements Semantic-assisted Object Cluster (SOC) for Referring Video Object Segmentation (RVOS), aiming to boost video-level visual-linguistic alignment by aggregating temporal context and semantic guidance. SOC introduces a unified temporal modeling architecture that improves segmentation stability and adaptation to textual variations.

## Environment Setup:
- install pytorch
  `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-   index-url https://download.pytorch.org/whl/cu113`
- install other dependencies
  `pip install h5py opencv-python protobuf av einops ruamel.yaml timm joblib pandas matplotlib cython scipy`
- install transformers numpy
  `pip install transformers==4.24.0`
  `pip install numpy==1.23.5`
- build up MultiScaleDeformableAttention
   ```
   cd ./models/ops
   python setup.py build install
   ```
## Dataset Structure :-
```text
rvosdata/
├── a2d_sentences/
│   ├── Release/CLIPS320/*.mp4
│   └── text_annotations/
│       ├── a2d_annotation.txt
│       └── a2d_annotation_with_instances/*/*.h5
└── refer_youtube_vos/
    ├── train/JPEGImages/*/*.jpg
    ├── train/Annotations/*/*.png
    └── meta_expressions/train/meta_expressions.json
```

## Update paths in:
- ./configs/a2d_sentences.yaml
- ./configs/refer_youtube_vos.yaml
- ./datasets/ scripts (if required)

## Pretrained Model:
```
pretrained/
├── pretrained_swin_transformer/
│   └── [Video-Swin-T or Video-Swin-B]
├── pretrained_roberta/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.json
```
