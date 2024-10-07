#!/usr/bin/env python
# coding=utf-8
import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    EMAModel,
    compute_dream_and_update_latents,
    compute_snr,
)
from diffusers.utils import (
    check_min_version,
    deprecate,
    is_wandb_available,
    make_image_grid,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# Importa gli adattatori
from dataset_adapters import get_dataset_adapter

if is_wandb_available():
    import wandb

# Verifica versione minima di diffusers
check_min_version("0.31.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script di training con supporto per diversi adattatori di dataset."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Percorso del modello pre-addestrato o identificatore del modello da huggingface.co/models.",
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        default="huggingface_dataset",
        required=False,
        help="Tipo di adattatore da utilizzare per il dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Il nome del Dataset dalla Hugging Face Hub da utilizzare per il training.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Percorso della cartella di dati di training locale.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default=None,
        help="La colonna del dataset contenente le immagini.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="La colonna del dataset contenente le didascalie.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="La directory dove verranno salvati i modelli e i dataset scaricati.",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Se specificato, utilizza il token di autenticazione per accedere ai dataset privati su Hugging Face Hub.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Se specificato, utilizza l'Exponential Moving Average (EMA) per lo smoothing dei pesi.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Utilizza l'ottimizzatore 8-bit Adam per ridurre l'utilizzo di memoria.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Risoluzione delle immagini di input; le immagini saranno ridimensionate a questa risoluzione.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Se specificato, applica un crop centrato alle immagini.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Se specificato, applica un flip orizzontale casuale alle immagini durante il training.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Dimensione del batch di training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Numero di passi di accumulo dei gradienti prima di un aggiornamento.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Abilita il gradient checkpointing per ridurre l'utilizzo di memoria.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Precisione mista da utilizzare. Opzioni: 'no', 'fp16', 'bf16'.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Numero massimo di passi di training.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Numero di epoche di training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Tasso di apprendimento.",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Valore massimo per il clipping del gradiente.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="Tipo di scheduler per il tasso di apprendimento.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Numero di passi per il warmup del tasso di apprendimento.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="La directory di output dove salvare il modello.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="La directory dove salvare i log di TensorBoard.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed per la riproducibilità.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Se specificato, il modello sarà caricato su Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Nome del repository su Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Token di autenticazione per Hugging Face Hub.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Salva un checkpoint ogni X passi.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Percorso del checkpoint da cui riprendere il training.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Rango locale per il training distribuito.",
    )
    args = parser.parse_args()

    # Gestione del local_rank per il training distribuito
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    # Configurazione dell'acceleratore
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=args.logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Imposta il seed
    if args.seed is not None:
        set_seed(args.seed)

    # Carica scheduler, tokenizer e modelli
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # Carica i modelli pre-addestrati
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    # Congela vae e text_encoder e imposta unet in modalità training
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Abilita il gradient checkpointing se specificato
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Inizializza l'ottimizzatore
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Per utilizzare l'ottimizzatore 8-bit Adam, installa bitsandbytes."
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
    )

    # Carica l'adattatore del dataset
    adapter = get_dataset_adapter(args.adapter_type)

    if args.dataset_name:
        train_dataset = adapter.load_dataset_from_name(
            args.dataset_name, tokenizer, args.resolution, args
        )
    elif args.train_data_dir:
        train_dataset = adapter.load_dataset(
            args.train_data_dir, tokenizer, args.resolution, args
        )
    else:
        raise ValueError("È necessario specificare --dataset_name o --train_data_dir.")

    # DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Calcolo dei passi di training
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps
            / (len(train_dataloader) / args.gradient_accumulation_steps)
        )

    # Scheduler del tasso di apprendimento
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepara modelli, dataloader e ottimizzatore con l'acceleratore
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Imposta il tipo di dato dei pesi
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Sposta text_encoder e vae sul dispositivo corretto
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Inizializza il progress bar
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Passi di training")

    # Loop di training
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Converti le immagini nello spazio latente
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Campiona il rumore e i timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Aggiungi rumore ai latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Ottieni gli hidden states del testo
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Prevedi il rumore residuo
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Calcola la perdita
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        # Salva il modello
                        unet.eval()
                        pipeline = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=accelerator.unwrap_model(unet),
                            scheduler=noise_scheduler,
                        )
                        pipeline.save_pretrained(args.output_dir)
                        unet.train()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Salva il modello finale
    if accelerator.is_main_process:
        unet.eval()
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            scheduler=noise_scheduler,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = Path(args.output_dir).name
            else:
                repo_name = args.hub_model_id
            create_repo(repo_id=repo_name, exist_ok=True, token=args.hub_token)
            pipeline.push_to_hub(repo_id=repo_name, use_auth_token=args.hub_token)

    accelerator.end_training()


if __name__ == "__main__":
    main()
