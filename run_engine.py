import argparse
import os
import sys

import cv2
import numpy as np

import torch

from PIL import Image

from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

from model.forgery_analyst.llava.conversation import conv_templates
from model.forgery_analyst.llava.utils import disable_torch_init
from model.forgery_analyst.llava.model.builder import load_pretrained_model
from model.forgery_analyst.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="Zhihao18/ForgeryAnalyst-llava-13B")

    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--mask-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)

    parser.add_argument("--manipulation-type", type=str, default='photoshop', 
                        choices=['photoshop', 'copy-move', 'remove', 'AI-generate'])

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    return parser.parse_args(args)


def highlight_forgery_boundary(image_path, mask_path, thickness=5):
    image = cv2.imread(image_path)

    (B, G, R) = cv2.split(image)
    sum_B, sum_G, sum_R = np.sum(B), np.sum(G), np.sum(R)

    min_channel = min(('R', sum_R), ('G', sum_G), ('B', sum_B), key=lambda x: x[1])
    color_dict = {'B': [255, 0, 0], 'G': [0, 255, 0], 'R': [0, 0, 255]}
    color = color_dict[min_channel[0]]

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    _, mask = cv2.threshold(mask, 32, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Create a new mask to mark the outer boundary touching areas
    outer_boundary_touching_mask = np.zeros_like(mask)
    
    # Mark pixels at the outer boundary in the mask
    outer_boundary_touching_mask[0, :] = mask[0, :]     # Top row
    outer_boundary_touching_mask[-1, :] = mask[-1, :]   # Bottom row
    outer_boundary_touching_mask[:, 0] = mask[:, 0]     # Left column
    outer_boundary_touching_mask[:, -1] = mask[:, -1]   # Right column

    outer_boundary = cv2.Canny(outer_boundary_touching_mask, threshold1=100, threshold2=200)
    outer_boundary_contours, _ = cv2.findContours(outer_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, outer_boundary_contours, -1, color, thickness)

    boundary = cv2.Canny(mask, threshold1=100, threshold2=200)
    boundary_contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, boundary_contours, -1, color, thickness)

    image_hb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return image_hb


def prepare_data(image_path, mask_path, manipulation_type='photoshop'):
    image = [highlight_forgery_boundary(image_path, mask_path)]

    default_question = "You are a rigorous and responsible image tampering (altering) detection expert. " \
        "You can localize the exact tampered region and analyze your detection decision according to tampering clues at different levels. " \
        "Assuming that you have detected this is a <FAKE> image and the manipulation type is [MANIPULATION_TYPE], " \
        "the exact tampered region boundary is highlighted with color in this image (and your detection IS correct).\n" \
        "Please provide the chain-of-clues supporting your detection decision in the following style: " \
        "# high-level semantic anomalies (such as content contrary to common sense, inciting and misleading content), " \
        "# middle-level visual defects (such as traces of tampered region or boundary, lighting inconsistency, perspective relationships, and physical constraints) and " \
        "# low-level pixel statistics (such as noise, color, textural, sharpness, and AI-generation fingerprint), " \
        "where the high-level anomalies are significant doubts worth attention, and the middle-level and low-level findings are reliable evidence." 
    
    question = default_question.replace('[MANIPULATION_TYPE]', manipulation_type)
    question = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    return image, prompt


def main(args):
    args = parse_args(args)

    disable_torch_init()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path)
    )

    if args.image_path and os.path.exists(args.image_path):
        image_path = args.image_path
    else:
        input("Please enter the path to the image file: ")
    
    if args.mask_path and os.path.exists(args.mask_path):
        mask_path = args.mask_path
    else:
        input("Please enter the path to the forgery mask file: ")
    
    image, prompt = prepare_data(image_path, mask_path, args.manipulation_type)

    image_size = [x.size for x in image]
    image_tensor = process_images(image, image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=image_size,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    if args.output_path:
        if os.path.exists(args.output_path):
            print(f"File {args.output_path} already exists.")
        else:
            os.mkdir(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, 'w') as f:
                f.write(outputs)

    print(outputs)


if __name__ == "__main__":
    main(sys.argv[1:])