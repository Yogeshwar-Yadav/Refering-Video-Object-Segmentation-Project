# import argparse
# import torch
# from trainer import Trainer
# from predict import predict
# import ruamel.yaml
# import os
# import wandb
# import warnings
# from ruamel.yaml import YAML

# warnings.filterwarnings("ignore")

# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '12357'


# def run(process_id, args):
#     with open(args.config_path) as f:
#         yaml = YAML(typ='safe', pure=True)
#         config = yaml.load(f)
#     config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
#     config = {**config, **vars(args)}
#     # Override batch_size from YAML only if not provided via CLI
#     if 'batch_size' not in config:
#         with open(args.config_path) as f:
#             yaml = YAML(typ='safe', pure=True)
#             yaml_config = yaml.load(f)
#         config['batch_size'] = yaml_config.get('batch_size', {}).get('value', 1)

#     config = argparse.Namespace(**config)
#     trainer = Trainer(config, process_id, device_id=args.device_ids[process_id], num_processes=args.num_devices)
#     if config.running_mode == 'train':
#         trainer.train()
#     elif config.running_mode == 'resume_train':
#         trainer.load_checkpoint(config.checkpoint_path)
#         trainer.train()
#     # elif config.running_mode == 'test':  # eval mode:
#     #     if os.path.exists(config.checkpoint_path):
#     #         model_state_dict = torch.load(config.checkpoint_path)
#     #         # model.load_state_dict(model_state_dict)
#     #     else:
#     #         print(f"Checkpoint not found at {config.checkpoint_path}, skipping loading.")

#     #     if 'model_state_dict' in model_state_dict.keys():
#     #         model_state_dict = model_state_dict['model_state_dict']
#     #     from torch.nn.parallel import DistributedDataParallel as DDP
#     #     model_without_ddp = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
#     #     model_without_ddp.load_state_dict(model_state_dict, strict=True)
#     #     trainer.evaluate()

#     elif config.running_mode == 'test':  # eval mode:
#         if os.path.exists(config.checkpoint_path):
#             model_state_dict = torch.load(config.checkpoint_path)
#             if 'model_state_dict' in model_state_dict:
#                 model_state_dict = model_state_dict['model_state_dict']
#             from torch.nn.parallel import DistributedDataParallel as DDP
#             model_without_ddp = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
#             model_without_ddp.load_state_dict(model_state_dict, strict=True)
#             trainer.evaluate()
#         else:
#             print(f"Checkpoint not found at {config.checkpoint_path}, skipping evaluation.")

#     # else:
#     #     model_state_dict = torch.load(config.checkpoint_path)
#     #     if 'model_state_dict' in model_state_dict.keys():
#     #         model_state_dict = model_state_dict['model_state_dict']
#     #     from torch.nn.parallel import DistributedDataParallel as DDP
#     #     model_without_ddp = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
#     #     model_without_ddp.load_state_dict(model_state_dict, strict=True)
#     #     predict(model_without_ddp, trainer.data_loader_val, trainer.device, trainer.postprocessor, config.out_dir)
#     elif config.running_mode == 'pred':
#         model_state_dict = torch.load(config.checkpoint_path)
#         if 'model_state_dict' in model_state_dict:
#             model_state_dict = model_state_dict['model_state_dict']

#         from torch.nn.parallel import DistributedDataParallel as DDP
#         model_without_ddp = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
#         model_without_ddp.load_state_dict(model_state_dict, strict=True)

#         predictions, input_frames = predict(
#             model_without_ddp, trainer.data_loader_val, trainer.device, trainer.postprocessor, config.out_dir
#         )

#         # 🔽 ADD THIS BLOCK TO SAVE OUTPUT VIDEO
#         from utils.visualization import generate_video_output  # or your actual function
#         output_path = os.path.join(config.out_dir, 'output_video.mp4')
#         generate_video_output(predictions, input_frames, output_path)

#     wandb.finish()


# if __name__ == '__main__':
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 6'
#     parser = argparse.ArgumentParser('SOC training and evaluation')
#     parser.add_argument('--config_path', '-c', required=True,
#                         help='path to configuration file')
#     parser.add_argument('--running_mode', '-rm', choices=['train', 'test', 'pred', 'resume_train'], required=True,
#                         help="mode to run, either 'train' or 'eval'")
#     parser.add_argument('--epochs', type=int, default=30,
#                         help='the total epochs')
#     parser.add_argument("--pretrained_weights", '-pw', default=None,
#                         help= 'The pretrained weights path'
#                         )
#     parser.add_argument("--version", required=True,
#                         help= "the saved ckpt and output version")
#     parser.add_argument("--lr_drop",default=[20], type=int, nargs='+')
#     parser.add_argument('--window_size', '-ws', type=int, default=8,
#                         help='window length to use during training/evaluation.'
#                              'note - in Refer-YouTube-VOS this parameter is used only during training, as'
#                              ' during evaluation full-length videos (all annotated frames) are used.')
#     parser.add_argument('--batch_size', '-bs', type=int, required=True,
#                         help='training batch size per device')
#     parser.add_argument("--backbone", type=str, required=True,
#                         help="the backbone name")
#     parser.add_argument("--backbone_pretrained_path", "-bpp", type=str, required=True,
#                         help="the backbone_pretrained_path")
    
#     gpu_params_group = parser.add_mutually_exclusive_group(required=True)
#     gpu_params_group.add_argument('--num_gpus', '-ng', type=int, default=argparse.SUPPRESS,
#                                   help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
#     gpu_params_group.add_argument('--gpu_ids', '-gids', type=int, nargs='+', default=argparse.SUPPRESS,
#                                   help='ids of GPUs to run on. mutually exclusive with \'num_gpus\'')
#     gpu_params_group.add_argument('--cpu', '-cpu', action='store_true', default=argparse.SUPPRESS,
#                                   help='run on CPU. Not recommended, but could be helpful for debugging if no GPU is'
#                                        'available.')
#     args = parser.parse_args()

#     # args.num_gpus = 1
#     # if args.eval_batch_size is None:
#     #     args.eval_batch_size = args.batch_size
#     if hasattr(args, 'num_gpus'):
#         args.num_devices = max(min(args.num_gpus, torch.cuda.device_count()), 1)
#         args.device_ids = list(range(args.num_gpus))
#         print(args.device_ids)
#     elif hasattr(args, 'gpu_ids'):
#         for gpu_id in args.gpu_ids:
#             assert 0 <= gpu_id <= torch.cuda.device_count() - 1, \
#                 f'error: gpu ids must be between 0 and {torch.cuda.device_count() - 1}'
#         args.num_devices = len(args.gpu_ids)
#         args.device_ids = args.gpu_ids
#     else:  # cpu
#         args.device_ids = ['cpu']
#         args.num_devices = 1

#     if args.num_devices > 1:
#         torch.multiprocessing.spawn(run, nprocs=args.num_devices, args=(args,))
#     else:  # run on a single GPU or CPU
#         print(f"Device Name : {torch.cuda.get_device_name(torch.cuda.current_device())}")
#         print(torch.cuda.device_count())
#         run(process_id=0, args=args)
import argparse
import torch
from trainer import Trainer
from predict import predict
import ruamel.yaml
import os
import wandb
import warnings
from ruamel.yaml import YAML
from transformers import RobertaTokenizer  # Import the tokenizer class

warnings.filterwarnings("ignore")

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12357'

def run(process_id, args):
    with open(args.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
    config = {**config, **vars(args)}

    # Override batch_size from YAML only if not provided via CLI
    if 'batch_size' not in config:
        with open(args.config_path) as f:
            yaml = YAML(typ='safe', pure=True)
            yaml_config = yaml.load(f)
        config['batch_size'] = yaml_config.get('batch_size', {}).get('value', 1)

    config = argparse.Namespace(**config)
    trainer = Trainer(config, process_id, device_id=args.device_ids[process_id], num_processes=args.num_devices)

    # Load the tokenizer (replace with the correct model's tokenizer if needed)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # Adjust if using a different model

    if config.running_mode == 'train':
        trainer.train()

    elif config.running_mode == 'resume_train':
        trainer.load_checkpoint(config.checkpoint_path)
        trainer.train()

    elif config.running_mode == 'test':
        if os.path.exists(config.checkpoint_path):
            model_state_dict = torch.load(config.checkpoint_path)
            if 'model_state_dict' in model_state_dict:
                model_state_dict = model_state_dict['model_state_dict']
            from torch.nn.parallel import DistributedDataParallel as DDP
            model_without_ddp = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
            model_without_ddp.load_state_dict(model_state_dict, strict=True)
            trainer.evaluate()
        else:
            print(f"Checkpoint not found at {config.checkpoint_path}, skipping evaluation.")

    elif config.running_mode == 'pred':
        model_state_dict = torch.load(config.checkpoint_path)
        if 'model_state_dict' in model_state_dict:
            model_state_dict = model_state_dict['model_state_dict']

        from torch.nn.parallel import DistributedDataParallel as DDP
        model_without_ddp = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
        model_without_ddp.load_state_dict(model_state_dict, strict=True)

        # ✅ Read query text and print debug info
        query_path = getattr(config, "query_path", "uploads/query.txt")
        if os.path.exists(query_path):
            with open(query_path, 'r', encoding='utf-8') as qf:
                query_text = qf.read().strip()
            print(f"\n📝 Loaded Query from '{query_path}': '{query_text}'\n")
        else:
            print(f"\n⚠️ Query file not found at path: {query_path}\n")
            query_text = ""

        # ✅ Pass query text to predict
        predictions, input_frames = predict(
            model_without_ddp,
            trainer.data_loader_val,
            trainer.device,
            trainer.postprocessor,
            config.out_dir,
            tokenizer,
            text_query=query_text  # <== pass query explicitly
        )

        # 🔽 Generate output video
        from utils.visualization import generate_video_output
        output_path = os.path.join(config.out_dir, 'output_video.mp4')
        generate_video_output(predictions, input_frames, output_path)

    wandb.finish()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 6'
    parser = argparse.ArgumentParser('SOC training and evaluation')
    parser.add_argument('--config_path', '-c', required=True, help='path to configuration file')
    parser.add_argument('--running_mode', '-rm', choices=['train', 'test', 'pred', 'resume_train'], required=True,
                        help="mode to run, either 'train' or 'eval'")
    parser.add_argument('--epochs', type=int, default=30, help='the total epochs')
    parser.add_argument("--pretrained_weights", '-pw', default=None, help='The pretrained weights path')
    parser.add_argument("--version", required=True, help="the saved ckpt and output version")
    parser.add_argument("--lr_drop", default=[20], type=int, nargs='+')
    parser.add_argument('--window_size', '-ws', type=int, default=8, help='window length to use during training/evaluation.')
    parser.add_argument('--batch_size', '-bs', type=int, required=True, help='training batch size per device')
    parser.add_argument("--backbone", type=str, required=True, help="the backbone name")
    parser.add_argument("--backbone_pretrained_path", "-bpp", type=str, required=True, help="the backbone_pretrained_path")
    parser.add_argument("--query_path", "-q", default="uploads/query.txt", help="path to query.txt file")  # ✅ added arg

    gpu_params_group = parser.add_mutually_exclusive_group(required=True)
    gpu_params_group.add_argument('--num_gpus', '-ng', type=int, default=argparse.SUPPRESS,
                                  help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    gpu_params_group.add_argument('--gpu_ids', '-gids', type=int, nargs='+', default=argparse.SUPPRESS,
                                  help='ids of GPUs to run on. mutually exclusive with \'num_gpus\'')
    gpu_params_group.add_argument('--cpu', '-cpu', action='store_true', default=argparse.SUPPRESS,
                                  help='run on CPU. Not recommended, but could be helpful for debugging if no GPU is available.')

    args = parser.parse_args()

    if hasattr(args, 'num_gpus'):
        args.num_devices = max(min(args.num_gpus, torch.cuda.device_count()), 1)
        args.device_ids = list(range(args.num_gpus))
        print(args.device_ids)
    elif hasattr(args, 'gpu_ids'):
        for gpu_id in args.gpu_ids:
            assert 0 <= gpu_id <= torch.cuda.device_count() - 1, \
                f'error: gpu ids must be between 0 and {torch.cuda.device_count() - 1}'
        args.num_devices = len(args.gpu_ids)
        args.device_ids = args.gpu_ids
    else:  # cpu
        args.device_ids = ['cpu']
        args.num_devices = 1

    if args.num_devices > 1:
        torch.multiprocessing.spawn(run, nprocs=args.num_devices, args=(args,))
    else:  # run on a single GPU or CPU
        print(f"Device Name : {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(torch.cuda.device_count())
        run(process_id=0, args=args)

