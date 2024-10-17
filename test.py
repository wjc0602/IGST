import argparse
import diffusers
import jittor
import json
import os
import tqdm

from diffusers import DiffusionPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument("--num_styles", type=int, default=28)
    parser.add_argument("--model_path", type=str, default='models')
    parser.add_argument("--save_dir", type=str, default='results')
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # cuda_flag
    jittor.flags.use_cuda = 1
    # setting random seed
    diffusers.training_utils.set_seed(args.seed)
    jittor.misc.set_global_seed(args.seed, different_seed_for_mpi=False)
    # Test in a no_grad environment
    with jittor.no_grad():
        for taskid in tqdm.tqdm(range(0, args.num_styles)):
            # Load Model
            taskid = '{:02d}'.format(taskid)
            # Create a folder to save the results
            os.makedirs(f"./{args.save_dir}/{taskid}", exist_ok=True)
            # Create diffusion pipeline and move to cuda
            pipe = DiffusionPipeline.from_pretrained(f"{args.model_path}/style_{taskid}").to("cuda")
            # Load Prompts
            with open(f"dataset/TrainB/{taskid}/prompt.json", "r") as file:
                prompts = json.load(file)
            # Go through each Prompt and generate a picture
            for _, prompt in prompts.items():
                # Construct Prompt
                prompt_new = f"A painting of a {prompt} in the style of <s0>"
                image = pipe(prompt_new, num_inference_steps=args.num_inference_steps, height=512, width=512).images[0]
                image.save(f"./{args.save_dir}/{taskid}/{prompt}.png")
