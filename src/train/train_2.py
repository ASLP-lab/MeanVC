# training script.

from importlib.resources import files

from src.model import  DiT 
from src.model import TrainerDis as Trainer

from prefigure.prefigure import get_all_args
import json
import os

import time

os.environ['OMP_NUM_THREADS']="1"
os.environ['MKL_NUM_THREADS']="1"


def main():
    args = get_all_args()
    print("Parsed arguments:", args)
    if isinstance(args.feature_list, str):
        args.feature_list = args.feature_list.split()
    if isinstance(args.additional_feature_list, str):
        args.additional_feature_list = args.additional_feature_list.split()
    if isinstance(args.feature_pad_values, str):
        args.feature_pad_values = args.feature_pad_values.split() 

    with open(args.model_config) as f:
        model_config = json.load(f)

    if model_config["model_type"] == "DiT":
        wandb_resume_id = None
        model_cls = DiT

    
    model = DiT(**model_config["model"])

    total_params = sum(p.numel() for p in model.parameters())/ 1000000
    print("Total parameters: {:.6f} M".format(total_params))

    trainer = Trainer(
        model,
        args,
        model_config,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=str(files("src").joinpath(f"../../ckpts/{args.exp_name}")),
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project="meanvc",
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=args.last_per_steps,
        bnb_optimizer=False,
        reset_lr=args.reset_lr,
        batch_size=args.batch_size,
        grad_ckpt=args.grad_ckpt
    )

    trainer.train(
        resumable_with_seed=args.resumable_with_seed,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
