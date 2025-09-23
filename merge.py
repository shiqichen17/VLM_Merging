import torch
import os
import argparse
import pdb
from transformers import (
    LlavaNextForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForVision2Seq,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from huggingface_hub import login
import re
import pdb
import merge_utils

def extract_layer_number(key):
    """Extract layer number from the key using regex."""
    match = re.search(r'\d+', key)  # Find the first occurrence of one or more digits
    return int(match.group()) if match else None  # Return the matched number or None

def merge_models(model1_path, model2_path, output_dir, alpha, mode='base', base_layer_num=-1, basemodel_path='base', density=0.2, alpha2=0.2):
    """
    Merges two models based on task-specific weight vectors relative to a base model.
    
    Args:
        model1_path (str): Path to the first model.
        model2_path (str): Path to the second model.
        output_dir (str): Directory to save the merged model.
        alpha (float): Weighting factor for combining the models.
        mode (str): Merging mode: base, layerswap, ties
        base_layer_num (int): Base layer number, required for layerswap
        basemodel_path (str): Base model, required for ties mode
        density (float): Density required for ties mode
        alpha2 (float): Alpha2 might need for ties mode
    """
    

    if 'llava' in model1_path:
        model1 = LlavaNextForConditionalGeneration.from_pretrained(
            model1_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=False,
            trust_remote_code=True,
            # device_map=create_other_model_device_map(model1_path)
        ).language_model
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=False,
            # device_map=create_other_model_device_map(model2_path)
        )
        # pdb.set_trace()
        excluded_keys = {'model.embed_tokens.weight', 'lm_head.weight'}

    elif 'idefics' in model1_path:
        model1 = AutoModelForVision2Seq.from_pretrained(
            model1_path,
            torch_dtype=torch.float16,    
        ).model.text_model
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=False,
            # device_map=create_other_model_device_map(model2_path)
        ).model
        excluded_keys = {'embed_tokens.weight', 'lm_head.weight'}

    elif 'Qwen' in model1_path:
        # Check if it's Qwen2.5-VL model
        if 'Qwen2.5-VL' in model1_path or 'Qwen2_5-VL' in model1_path:
            model1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model1_path, torch_dtype=torch.bfloat16
            ).model
        else:
            model1 = Qwen2VLForConditionalGeneration.from_pretrained(
                model1_path, torch_dtype="auto"
            ).model
        model2 = AutoModelForCausalLM.from_pretrained(model2_path, torch_dtype="auto").model
        excluded_keys = {'embed_tokens.weight'}

    else:
        model1 = AutoModelForCausalLM.from_pretrained(
            model1_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=False,
            trust_remote_code=True,
            # device_map=create_other_model_device_map(model1_path)
        ).language_model
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=False,
            device_map=create_other_model_device_map(model2_path)
        )
        excluded_keys = {'model.embed_tokens.weight', 'lm_head.weight'}

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    del model1
    del model2

    if mode in ['ties', 'dareties','darelinear']:
        basemodel=AutoModelForCausalLM.from_pretrained(
            basemodel_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=False,
        )
        if 'llava' not in model1_path:
            basemodel=basemodel.model
        state_dict_base = basemodel.state_dict()
        del basemodel
        taskvec1 = {
            k: state_dict1[k] - state_dict_base[k]
            for k in state_dict1.keys()
            if k not in excluded_keys
        }
        taskvec2 = {
            k: state_dict2[k] - state_dict_base[k]
            for k in state_dict2.keys()
            if k not in excluded_keys
        }
        del state_dict2
        if alpha2 is not None:
            weights=torch.tensor([alpha,alpha2])
        else:
            weights=torch.tensor([alpha,1-alpha])
        # merge by modules
        if mode=='ties':
            mixvec={k: merge_utils.ties([taskvec1[k],taskvec2[k]],weights,density) for k in taskvec1.keys()} 
        elif mode=='dareties':
            mixvec={k: merge_utils.dare_ties([taskvec1[k],taskvec2[k]],weights,density) for k in taskvec1.keys()}
        elif mode=='darelinear':
            mixvec={k: merge_utils.dare_linear([taskvec1[k],taskvec2[k]],weights,density) for k in taskvec1.keys()}

        state_dict1 = {
            k: state_dict_base[k] + mixvec[k] if k not in excluded_keys else state_dict1[k]
            for k in state_dict_base.keys()
        }
    else:
        from tqdm import tqdm
        for layer in tqdm(list(state_dict2.keys())):
            layer_number = extract_layer_number(layer)
            if layer not in excluded_keys:
                if mode == 'layerswap':
                    if layer_number is not None and layer_number <= base_layer_num:
                        state_dict1[layer].copy_(state_dict1[layer])
                    else:
                        state_dict1[layer].copy_(alpha * state_dict1[layer] + (1 - alpha) * state_dict2[layer])
                elif mode == 'base':
                    state_dict1[layer].copy_(alpha * state_dict1[layer] + (1 - alpha) * state_dict2[layer])
            else:
                print(layer)

    # Save the merged model state dict
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"merged_model_{alpha}.pth")
    torch.save(state_dict1, save_path)

    print(f"Merged model weights saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge two models based on task-specific deltas from a base model.")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to the first model.")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to the second model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model.")
    parser.add_argument("--alpha", type=float, required=True, help="Weighting factor for the merge.")
    parser.add_argument("--mode", type=str, default='base', help="Merging mode: base, layerswap, ties, dareties, darelinear")
    parser.add_argument("--base_layer_num", type=int, default=-1, help="Base layer number, required for layerswap")
    parser.add_argument("--basemodel_path", type=str, default='base', help="Base model, required for ties mode")
    parser.add_argument("--density", type=float, default=0.2, help="Density required for ties mode")
    parser.add_argument("--alpha2", type=float, default=0.2, help="Alpha2 might need for ties mode")

    args = parser.parse_args()

    # Perform the merge operation
    merge_models(args.model1_path, args.model2_path, args.output_dir, args.alpha, args.mode, args.base_layer_num, args.basemodel_path, args.density, args.alpha2)

if __name__ == "__main__":
    main()
