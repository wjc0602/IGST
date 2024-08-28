import json, os, tqdm, jittor
from diffusers import DiffusionPipeline
import argparse
import diffusers

jittor.flags.use_cuda = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple example of a testing script.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    diffusers.training_utils.set_seed(args.seed)
    jittor.misc.set_global_seed(args.seed, different_seed_for_mpi=False)
    save_dir = 'results'

    with jittor.no_grad():
        for taskid in tqdm.tqdm(range(0, 28)):
            print(f"1. 加载模型！")
            taskid = '{:02d}'.format(taskid)
            os.makedirs(f"./{save_dir}/{taskid}", exist_ok=True)
            model_path = f"checkpoints/style_{taskid}"
            pipe = DiffusionPipeline.from_pretrained(model_path).to("cuda")
            print(f"2. 加载Prompt！")
            with open(f"dataset/TrainB/{taskid}/prompt.json", "r") as file:
                prompts = json.load(file)
                
            print(f"3. 遍历每一个Prompt，开始风格{taskid}的生成！")
            for id, prompt in prompts.items():
                prompt_new = f"A painting of a {prompt} in the style of <s0>"
                image = pipe(prompt_new, num_inference_steps=args.num_inference_steps, height=512, width = 512).images[0]
                image.save(f"./{save_dir}/{taskid}/{prompt}.png")
