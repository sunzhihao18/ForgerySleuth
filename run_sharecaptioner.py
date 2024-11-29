# https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/tools/share-cap_batch_infer.py

import argparse
import random
import os
import sys

import torch

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt.real_analysis_text import REAL_ANALYSIS_TEXT_LIST


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="Lin-Chen/ShareCaptioner")

    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)

    parser.add_argument("--num_gpus", default=1, type=int)
    
    return parser.parse_args(args)


def auto_configure_device_map(num_gpus):
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus

    device_map = {
        'visual_encoder': 0,
        'ln_vision': 0,
        'Qformer': 0,
        'internlm_model.model.embed_tokens': 0,
        'internlm_model.model.norm': 0,
        'internlm_model.lm_head': 0,
        'query_tokens': 0,
        'flag_image_start': 0,
        'flag_image_end': 0,
        'internlm_proj': 0,
    }

    used = 6
    gpu_target = 0

    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'internlm_model.model.layers.{i}'] = gpu_target
        used += 1

    return device_map


def main(args):
    args = parse_args(args)

    # You can download ShareCaptioner in advance, 
    # and use `local_files_only=True` to force the use of local weights, 
    # avoiding potential network issues.
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True).eval().half()

    if args.num_gpus > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(args.num_gpus)
        model = dispatch_model(model, device_map=device_map)
    else:
        model.cuda()

    model.tokenizer = tokenizer

    if args.image_path and os.path.exists(args.image_path):
        image_path = args.image_path
    else:
        image_path = input("Please enter the path to the image file: ")

    image = Image.open(image_path).convert('RGB')

    prompt_seg1 = '<|User|>:'
    prompt_seg2 = f'Analyze the image in a comprehensive and detailed manner.{model.eoh}\n<|Bot|>:'

    with torch.no_grad():
        image = model.vis_processor(image).unsqueeze(0)
        image = model.encode_img(image.to(torch.float16))

        prompt_emb1 = model.encode_text(prompt_seg1, add_special_tokens=True).unsqueeze(0)
        prompt_emb2 = model.encode_text(prompt_seg2, add_special_tokens=False).unsqueeze(0)
        
        input = torch.cat([prompt_emb1, image, prompt_emb2], dim=1)

        out_embeds = model.internlm_model.generate(
            inputs_embeds=input,
            max_length=512,
            num_beams=3,
            min_length=1,
            do_sample=True,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1.,
            eos_token_id=model.tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        caption = model.decode_text(out_embeds)
        caption = caption.replace('\n', '')

    analysis = random.choice(REAL_ANALYSIS_TEXT_LIST)
    analysis = analysis.replace('[DETAILED_CAPTION]', caption)
    
    if args.output_path:
        if os.path.exists(args.output_path):
            print(f"File {args.output_path} already exists.")
        else:
            os.mkdir(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, 'w') as f:
                f.write(analysis)

    print(analysis)


if __name__ == "__main__":
    main(sys.argv[1:])