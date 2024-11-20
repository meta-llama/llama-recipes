import os
import argparse
import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from tqdm.auto import tqdm
import csv
from PIL import Image
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import shutil
import time

USER_TEXT = """
You are an expert fashion captioner, we are writing descriptions of clothes, look at the image closely and write a caption for it.

Write the following Title, Size, Category, Gender, Type, Description in JSON FORMAT, PLEASE DO NOT FORGET JSON, 

ALSO START WITH THE JSON AND NOT ANY THING ELSE, FIRST CHAR IN YOUR RESPONSE IS ITS OPENING BRACE

FOLLOW THESE STEPS CLOSELY WHEN WRITING THE CAPTION: 
1. Only start your response with a dictionary like the example below, nothing else, I NEED TO PARSE IT LATER, SO DONT ADD ANYTHING ELSE-IT WILL BREAK MY CODE
Remember-DO NOT SAY ANYTHING ELSE ABOUT WHAT IS GOING ON, just the opening brace is the first thing in your response nothing else ok?
2. REMEMBER TO CLOSE THE DICTIONARY WITH '}'BRACE, IT GOES AFTER THE END OF DESCRIPTION-YOU ALWAYS FORGET IT, THIS WILL CAUSE A LOT OF ISSUES
3. If you cant tell the size from image, guess it! its okay but dont literally write that you guessed it
4. Do not make the caption very literal, all of these are product photos, DO NOT CAPTION HOW OR WHERE THEY ARE PLACED, FOCUS ON WRITING ABOUT THE PIECE OF CLOTHING
5. BE CREATIVE WITH THE DESCRIPTION BUT FOLLOW EVERYTHING CLOSELY FOR STRUCTURE
6. Return your answer in dictionary format, see the example below

{"Title": "Title of item of clothing", "Size": {'S', 'M', 'L', 'XL'}, #select one randomly if you cant tell from the image. DO NOT TELL ME YOU ESTIMATE OR GUESSED IT ONLY THE LETTER IS ENOUGH", Category":  {T-Shirt, Shoes, Tops, Pants, Jeans, Shorts, Skirts, Shoes, Footwear}, "Gender": {M, F, U}, "Type": {Casual, Formal, Work Casual, Lounge}, "Description": "Write it here"}

Example: ALWAYS RETURN ANSWERS IN THE DICTIONARY FORMAT BELOW OK?

{"Title": "Casual White pant with logo on it", "size": "L", "Category": "Jeans", "Gender": "U", "Type": "Work Casual", "Description": "Write it here, this is where your stuff goes"} 
"""

def is_image_corrupt(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError, Image.UnidentifiedImageError):
        return True

def find_and_move_corrupt_images(folder_path, corrupt_folder):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    num_cores = mp.cpu_count()
    with tqdm(total=len(image_files), desc="Checking for corrupt images", unit="file", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(is_image_corrupt, image_files))
            pbar.update(len(image_files))
    
    corrupt_images = [img for img, is_corrupt in zip(image_files, results) if is_corrupt]
    
    os.makedirs(corrupt_folder, exist_ok=True)
    for img in tqdm(corrupt_images, desc="Moving corrupt images", unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
        shutil.move(img, os.path.join(corrupt_folder, os.path.basename(img)))
    
    print(f"Moved {len(corrupt_images)} corrupt images to {corrupt_folder}")

def get_image(image_path):
    return Image.open(image_path).convert('RGB')

def llama_progress_bar(total, desc, position=0):
    """Custom progress bar with llama emojis."""
    bar_format = "{desc}: |{bar}| {percentage:3.0f}% [{n_fmt}/{total_fmt}, {rate_fmt}{postfix}]"
    return tqdm(total=total, desc=desc, position=position, bar_format=bar_format, ascii="🦙·")

def process_images(rank, world_size, args, model_name, input_files, output_csv):
    model = MllamaForConditionalGeneration.from_pretrained(model_name, device_map=f"cuda:{rank}", torch_dtype=torch.bfloat16, token=args.hf_token)
    processor = MllamaProcessor.from_pretrained(model_name, token=args.hf_token)

    chunk_size = len(input_files) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(input_files)
    
    results = []
    
    pbar = llama_progress_bar(total=end_idx - start_idx, desc=f"GPU {rank}", position=rank)
    
    for filename in input_files[start_idx:end_idx]:
        image_path = os.path.join(args.input_path, filename)
        image = get_image(image_path)

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT}]}
        ]

        prompt = processor.apply_chat_template(conversation, add_special_tokens=False, add_generation_prompt=True, tokenize=False)
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, temperature=1, top_p=0.9, max_new_tokens=512)
        decoded_output = processor.decode(output[0])[len(prompt):]

        results.append((filename, decoded_output))
        
        pbar.update(1)
        pbar.set_postfix({"Last File": filename})

    pbar.close()

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Caption'])
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Image Captioning")
    parser.add_argument("--hf_token", required=True, help="Hugging Face API token")
    parser.add_argument("--input_path", required=True, help="Path to input image folder")
    parser.add_argument("--output_path", required=True, help="Path to output CSV folder")
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--corrupt_folder", default="corrupt_images", help="Folder to move corrupt images")
    args = parser.parse_args()

    model_name = "meta-llama/Llama-3.2-11b-Vision-Instruct"

    print("🦙 Starting image processing pipeline...")
    start_time = time.time()

    # Find and move corrupt images
    corrupt_folder = os.path.join(args.input_path, args.corrupt_folder)
    find_and_move_corrupt_images(args.input_path, corrupt_folder)

    # Get list of remaining (non-corrupt) image files
    input_files = [f for f in os.listdir(args.input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"\n🦙 Processing {len(input_files)} images using {args.num_gpus} GPUs...")

    mp.set_start_method('spawn', force=True)
    processes = []

    for rank in range(args.num_gpus):
        output_csv = os.path.join(args.output_path, f"captions_gpu_{rank}.csv")
        p = mp.Process(target=process_images, args=(rank, args.num_gpus, args, model_name, input_files, output_csv))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n🦙 Total processing time: {total_time:.2f} seconds")
    print("🦙 Image captioning completed successfully!")

if __name__ == "__main__":
    main()
