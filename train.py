import jittor as jt
import jittor.nn as nn
import argparse
import copy
import logging
import math
import os
import warnings
import itertools
import json
from pathlib import Path
from safetensors.numpy import save_file
import transformers
from PIL import Image
from PIL.ImageOps import exif_transpose
from jittor import transform
from jittor.compatibility.optim import AdamW
from jittor.compatibility.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DiffusionPipeline
import diffusers
from JDiffusion import (AutoencoderKL, UNet2DConditionModel,)
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

jt.flags.use_cuda = 1
diffusers.training_utils.set_seed(1234)
jt.misc.set_global_seed(1234, different_seed_for_mpi=False)

def test_in_train(pipeline, prompt_json_path, img_save_dir,  instance_prompt, num_inference_steps):
    with open(prompt_json_path, "r") as file:
        prompts = json.load(file)
    for id, prompt in prompts.items():
        prompt_new = f"A painting of a {prompt} in the style of {instance_prompt}"
        print(f"测试用的提示词：{prompt_new}")
        image = pipeline(prompt_new, 
                        num_inference_steps=num_inference_steps,
                        height=args.resolution, 
                        width=args.resolution, 
                        guidance_scale=args.guidance_scale).images[0]
        os.makedirs(img_save_dir, exist_ok=True)
        image.save(f"{img_save_dir}/{prompt}.png")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--num_process",type=int,default=1)
    parser.add_argument("--pretrained_model_name_or_path",type=str,default='models/stable-diffusion-2-1')
    parser.add_argument("--revision",type=str,default=None,required=False)
    parser.add_argument("--variant",type=str,default=None)
    parser.add_argument("--tokenizer_name",type=str,default=None)
    parser.add_argument("--instance_data_dir",type=str,default="dataset/TrainA/00/images")
    parser.add_argument("--class_data_dir",type=str,default=None,required=False)
    parser.add_argument("--instance_prompt",type=str,default="TOK")
    parser.add_argument("--class_prompt",type=str,default=None)
    parser.add_argument("--num_class_images",type=int,default=100)
    parser.add_argument("--output_dir",type=str,default="checkpoints/test/style_00")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution",type=int,default=512)
    parser.add_argument("--center_crop",default=False,action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument("--sample_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps",type=int,default=10)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--gradient_checkpointing",action="store_true")
    parser.add_argument("--learning_rate",type=float,default=5e-4)
    parser.add_argument("--scale_lr",action="store_true",default=False)
    parser.add_argument("--lr_scheduler",type=str,default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles",type=int,default=1)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--dataloader_num_workers",type=int,default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--tokenizer_max_length",type=int,default=None,required=False,)
    parser.add_argument("--text_encoder_use_attention_mask",action="store_true",required=False,)
    parser.add_argument("--validation_images",required=False,default=None,nargs="+")
    parser.add_argument("--class_labels_conditioning",required=False,default=None)

    # prior preservation loss
    parser.add_argument("--with_prior_preservation",default=False,action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    # for test
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--save_checkpoint", action="store_true")
    # train text encoder
    parser.add_argument("--train_text_encoder", action="store_true",help="Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    parser.add_argument("--text_encoder_lr",type=float,default=5e-6,help="Text encoder learning rate to use.")
    parser.add_argument("--train_text_encoder_ti",action="store_true",help=("Whether to use textual inversion"))
    parser.add_argument("--train_text_encoder_ti_frac",type=float,default=0.5,help=("The percentage of epochs to perform textual inversion"))
    parser.add_argument("--train_text_encoder_frac",type=float,default=1.0,help=("The percentage of epochs to perform text encoder tuning"))
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder")
    parser.add_argument("--token_abstraction",type=str,default="TOK")
    parser.add_argument("--num_new_tokens_per_abstraction",type=int,default=2,help="每个token_abstraction标识符插入到标记器的新标记数")
    # Sparse Updating
    parser.add_argument("--use_sparse_update", action="store_true")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.train_ids = None
        self.inserting_toks  = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, inserting_toks):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(inserting_toks, list), "inserting_toks should be a list of strings."
            assert all(isinstance(tok, str) for tok in inserting_toks), "All elements in inserting_toks should be strings."

            self.inserting_toks = inserting_toks
            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            text_encoder.resize_token_embeddings(len(tokenizer))

            self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            std_token_embedding = text_encoder.text_model.embeddings.token_embedding.weight.data.std()
            print(f"{idx} text encodedr's std_token_embedding: {std_token_embedding}")
            text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids] = (
                jt.randn(len(self.train_ids), text_encoder.text_model.config.hidden_size).to(device=self.device).to(dtype=self.dtype) * std_token_embedding
            )
            self.embeddings_settings[f"original_embeddings_{idx}"] = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            inu = jt.ones((len(tokenizer),), dtype=jt.bool)
            inu[self.train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = inu
            print(self.embeddings_settings[f"index_no_updates_{idx}"].shape)

            idx += 1

    # copied from train_dreambooth_lora_sdxl_advanced.py
    def save_embeddings(self, file_path: str):
        assert self.train_ids is not None, "Initialize new tokens before saving embeddings."
        tensors = {}
        # text_encoder_0 - CLIP ViT-L/14, text_encoder_1 -  CLIP ViT-G/14 - TODO - change for sd
        idx_to_text_encoder_name = {0: "clip_l", 1: "clip_g"}
        for idx, text_encoder in enumerate(self.text_encoders):
            assert text_encoder.text_model.embeddings.token_embedding.weight.data.shape[0] == len(self.tokenizers[0]), "Tokenizers should be the same."
            new_token_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids]
            tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings.numpy()
        print(f"保存的位置：{file_path}")
        save_file(tensors, file_path)

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    @jt.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we need to normalize them
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = jt.logical_not(index_no_updates)
            new_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates]
            off_ratio = std_token_embedding / new_embeddings.std()

            new_embeddings = new_embeddings * (off_ratio**0.1)
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates] = new_embeddings

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        train_text_encoder_ti, 
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        repeats=1,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        token_abstraction_dict=None,  # token mapping for textual inversion
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        instance_images_path = list(Path(instance_data_root).iterdir())
        self.instance_images_path = []
        for img in instance_images_path:
            self.instance_images_path.extend(itertools.repeat(img, repeats))
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = class_data_root
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transform.Compose(
            [
                transform.Resize(size),
                transform.CenterCrop(size) if center_crop else transform.RandomCrop(size),
                transform.RandomHorizontalFlip(0.5),
                transform.ToTensor(),
                transform.ImageNormalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)
        instance_image_name = os.path.basename(self.instance_images_path[index % self.num_instance_images])[:-4]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        prompt = f"A painting of a {instance_image_name} in the style of {self.instance_prompt}"
        example["instance_tokens"] = tokenize_prompt(self.tokenizer, prompt, args.train_text_encoder_ti)

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)
            class_image_name = os.path.basename(self.class_images_path[index % self.num_class_images])[:-4]

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            class_prompt = f"A painting of a {class_image_name}" # 构造提示词
            example["class_tokens"] = tokenize_prompt(self.tokenizer, class_prompt)
        return example

def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_tokens"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if with_prior_preservation:
        input_ids += [example["class_tokens"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = jt.stack(pixel_values)
    pixel_values = pixel_values.float()
    input_ids = jt.cat(input_ids, dim=0)

    batch = { "input_ids": input_ids, "pixel_values": pixel_values}
    return batch

def tokenize_prompt(tokenizer, prompt,  add_special_tokens=False):
    if add_special_tokens:
        text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                add_special_tokens=add_special_tokens,
                return_tensors="pt",
            )
    else:
        text_inputs = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoder, tokenizers, prompt, text_input_ids_list=None):
    assert text_input_ids_list is not None
    text_input_ids = text_input_ids_list
    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device),output_hidden_states=True,)
    return prompt_embeds[0]


def main(args):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # Handle the repository creation
    if args.output_dir is not None and args.save_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder="tokenizer",revision=args.revision,use_fast=False,)
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    
    if args.train_text_encoder_ti:
        # we parse the provided token identifier (or identifiers) into a list. s.t. - "TOK" -> ["TOK"], "TOK,
        # TOK2" -> ["TOK", "TOK2"] etc.
        token_abstraction_list = "".join(args.token_abstraction.split()).split(",")
        print(f"list of token identifiers: {token_abstraction_list}")

        token_abstraction_dict = {}
        token_idx = 0
        for i, token in enumerate(token_abstraction_list):
            token_abstraction_dict[token] = [f"<s{token_idx + i + j}>" for j in range(args.num_new_tokens_per_abstraction)]
            token_idx += args.num_new_tokens_per_abstraction - 1

        # replace instances of --token_abstraction in --instance_prompt with the new tokens: "<si><si+1>" etc.
        for token_abs, token_replacement in token_abstraction_dict.items():
            args.instance_prompt = args.instance_prompt.replace(token_abs, "".join(token_replacement))
            # if args.with_prior_preservation:
            #     args.class_prompt = args.class_prompt.replace(token_abs, "".join(token_replacement))
        print(f"更改之后的token：{args.instance_prompt}")
        # initialize the new tokens for textual inversion
        embedding_handler = TokenEmbeddingsHandler([text_encoder], [tokenizer])
        inserting_toks = []
        for new_tok in token_abstraction_dict.values():
            inserting_toks.extend(new_tok)
        embedding_handler.initialize_new_tokens(inserting_toks=inserting_toks)
    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = jt.float32
    for name, param in unet.named_parameters():
        param.requires_grad = True
        
    for name, param in unet.named_parameters():
        assert param.requires_grad == True, name

    # 处理需要训练的text encoder的参数
    if args.train_text_encoder:
        for name, param in text_encoder.named_parameters():
            param.requires_grad = True
        text_parameters = text_encoder.parameters()
    elif args.train_text_encoder_ti:
        text_parameters = []
        for name, param in text_encoder.named_parameters():
            if "token_embedding" in name:
                param.requires_grad = True
                text_parameters.append(param)
            else:
                param.requires_grad = False

    optimizer_unet = AdamW(
        list(unet.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.train_text_encoder or args.train_text_encoder_ti:
        optimizer_text = AdamW(
            list(text_parameters),
            lr=args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay_text_encoder if args.adam_weight_decay_text_encoder else args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    pre_computed_encoder_hidden_states = None
    pre_computed_class_prompt_encoder_hidden_states = None

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        repeats=args.repeats,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        train_text_encoder_ti=args.train_text_encoder_ti,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_unet = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_unet,
        num_warmup_steps=args.lr_warmup_steps * args.num_process,
        num_training_steps=args.max_train_steps * args.num_process,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    if args.train_text_encoder or args.train_text_encoder_ti:
        lr_scheduler_text = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_text,
            num_warmup_steps=args.lr_warmup_steps * args.num_process,
            num_training_steps=args.max_train_steps * args.num_process,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * args.num_process * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Style = {args.instance_data_dir}")
    print(f"  Class = {args.class_data_dir}")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=False,
    )
    if args.train_text_encoder:
        num_train_epochs_text_encoder = int(args.train_text_encoder_frac * args.num_train_epochs)
    elif args.train_text_encoder_ti:  # args.train_text_encoder_ti
        num_train_epochs_text_encoder = int(args.train_text_encoder_ti_frac * args.num_train_epochs)

    for epoch in range(first_epoch, args.num_train_epochs):
        if args.train_text_encoder or args.train_text_encoder_ti:
            if epoch == num_train_epochs_text_encoder: # 停止优化text encoder参数
                print("PIVOT HALFWAY", epoch)
                optimizer_text.lr = 0.0
            else:
                text_encoder.train()

        unet.train()
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor
            # Sample noise that we'll add to the latents
            noise = jt.randn_like(model_input)
            bsz = model_input.shape[0]

            if args.use_sparse_update:
                timesteps = jt.randint(0, 50, (bsz,)).long()
                timesteps = timesteps * (1000 // args.num_inference_steps)
            else:
                timesteps = jt.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), ).to(device=model_input.device)
                timesteps = timesteps.long()

            # 5. Add noise to the model input according to the noise magnitude at each timestep
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # Get the text embedding for conditioning
            input_ids = batch["input_ids"]
            prompt_embeds_input = encode_prompt(
                    text_encoder = text_encoder,
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=input_ids
                )

            if args.class_labels_conditioning == "timesteps":
                class_labels = timesteps
            else:
                class_labels = None

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds_input,
                class_labels=class_labels,
                return_dict=False,
            )[0]

            if model_pred.shape[1] == 6:
                model_pred, _ = jt.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.with_prior_preservation:
                model_pred, model_pred_prior = jt.chunk(model_pred, 2, dim=0)
                target, target_prior = jt.chunk(target, 2, dim=0)
                loss = nn.mse_loss(model_pred.float(), target.float(), reduction="mean")
                prior_loss = nn.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                loss = nn.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
            loss.backward()

            optimizer_unet.step()
            lr_scheduler_unet.step()

            if args.train_text_encoder or args.train_text_encoder_ti:
                optimizer_text.step()
                lr_scheduler_text.step()

            optimizer_unet.zero_grad()
            if args.train_text_encoder or args.train_text_encoder_ti:
                optimizer_text.zero_grad()

            if args.train_text_encoder_ti:
                embedding_handler.retract_embeddings()

            progress_bar.update(1)
            global_step += 1
            if args.train_text_encoder or args.train_text_encoder_ti:
                logs = {"loss": loss.detach().item(), "lru": lr_scheduler_unet.get_last_lr()[0], "lrt": lr_scheduler_text.get_last_lr()[0], "bsz": bsz}
            else:
                logs = {"loss": loss.detach().item(), "lru": lr_scheduler_unet.get_last_lr()[0], "bsz": bsz,}
            progress_bar.set_postfix(**logs)

            # Test the model 
            if args.test and (global_step in [300] or global_step >= args.max_train_steps):
                prompt_json_path=f'{args.instance_data_dir}/../prompt.json'
                img_save_dir=f'{args.output_dir}'.replace('checkpoints','results').replace('style_','').replace('_e_',f'_e{global_step}_')
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae = vae,
                    unet = unet,
                    tokenizer = tokenizer,
                    text_encoder = text_encoder,
                    variant=args.variant
                )
                # test_in_train(pipeline, prompt_json_path, img_save_dir, args.instance_prompt, args.num_inference_steps)    
                if args.save_checkpoint:
                    pipeline.save_pretrained(args.output_dir,safe_serialization=False)
            if global_step >= args.max_train_steps:
                break
    
if __name__ == "__main__":
    args = parse_args()
    main(args)