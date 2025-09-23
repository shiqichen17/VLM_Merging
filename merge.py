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
            # Load Qwen2.5-VL model and extract language model part
            full_model1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model1_path, torch_dtype=torch.bfloat16
            )
            # Try different possible language model attributes
            if hasattr(full_model1, 'language_model'):
                model1 = full_model1.language_model
            elif hasattr(full_model1, 'model') and hasattr(full_model1.model, 'language_model'):
                model1 = full_model1.model.language_model
            elif hasattr(full_model1, 'model'):
                model1 = full_model1.model
            else:
                model1 = full_model1
            print(f"Qwen2.5-VL model1 structure: {type(model1)}")
        else:
            model1 = Qwen2VLForConditionalGeneration.from_pretrained(
                model1_path, torch_dtype="auto"
            ).model

        # Load model2 - should be the base language model
        if 'Qwen' in model2_path:
            model2_full = AutoModelForCausalLM.from_pretrained(model2_path, torch_dtype="auto")
            model2 = model2_full.model if hasattr(model2_full, 'model') else model2_full
            print(f"Qwen2.5 model2 structure: {type(model2)}")
        else:
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
        basemodel_full=AutoModelForCausalLM.from_pretrained(
            basemodel_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=False,
        )
        if 'llava' not in model1_path and 'Qwen' in basemodel_path:
            basemodel=basemodel_full.model if hasattr(basemodel_full, 'model') else basemodel_full
        elif 'llava' not in model1_path:
            basemodel=basemodel_full.model
        else:
            basemodel=basemodel_full
        print(f"Base model structure: {type(basemodel)}")
        state_dict_base = basemodel.state_dict()
        del basemodel
        # Get common keys between models and base model for ties mode
        common_keys_with_base = set(state_dict1.keys()) & set(state_dict2.keys()) & set(state_dict_base.keys())

        print(f"\n=== TIES MODE ANALYSIS ===")
        print(f"Model1 keys: {len(state_dict1)}")
        print(f"Model2 keys: {len(state_dict2)}")
        print(f"Base model keys: {len(state_dict_base)}")
        print(f"Common keys across all 3 models: {len(common_keys_with_base)}")
        print(f"Excluded keys: {excluded_keys}")

        # Keys that will be used for task vectors
        taskvec_keys = common_keys_with_base - excluded_keys
        excluded_common_keys = common_keys_with_base & excluded_keys

        print(f"\n=== TIES MERGE PLAN ===")
        print(f"Keys for task vectors: {len(taskvec_keys)}")
        print(f"Keys excluded from task vectors: {len(excluded_common_keys)}")

        if excluded_common_keys:
            print(f"Excluded common keys: {sorted(list(excluded_common_keys))}")

        # Show sample of keys for task vectors
        sample_taskvec_keys = sorted(list(taskvec_keys))[:10]
        print(f"Sample task vector keys: {sample_taskvec_keys}")

        taskvec1 = {
            k: state_dict1[k] - state_dict_base[k]
            for k in taskvec_keys
        }
        taskvec2 = {
            k: state_dict2[k] - state_dict_base[k]
            for k in taskvec_keys
        }

        print(f"\n=== TASK VECTORS CREATED ===")
        print(f"TaskVec1 keys: {len(taskvec1)}")
        print(f"TaskVec2 keys: {len(taskvec2)}")

        # Check for keys not common to all three models
        only_in_base = set(state_dict_base.keys()) - common_keys_with_base
        only_in_model1_or_2 = (set(state_dict1.keys()) | set(state_dict2.keys())) - common_keys_with_base

        if only_in_base:
            print(f"\n=== KEYS ONLY IN BASE MODEL ({len(only_in_base)}) ===")
            if len(only_in_base) <= 20:
                for key in sorted(only_in_base):
                    print(f"  - {key}")
            else:
                print(f"First 10: {sorted(list(only_in_base))[:10]}")

        if only_in_model1_or_2:
            print(f"\n=== KEYS NOT IN ALL 3 MODELS ({len(only_in_model1_or_2)}) ===")
            if len(only_in_model1_or_2) <= 20:
                for key in sorted(only_in_model1_or_2):
                    print(f"  - {key}")
            else:
                print(f"First 10: {sorted(list(only_in_model1_or_2))[:10]}")
        del state_dict2
        if alpha2 is not None:
            weights=torch.tensor([alpha,alpha2])
        else:
            weights=torch.tensor([alpha,1-alpha])
        # merge by modules
        print(f"\n=== APPLYING {mode.upper()} MERGE ===")
        print(f"Weights: {weights}")
        print(f"Density: {density}")

        if mode=='ties':
            mixvec={k: merge_utils.ties([taskvec1[k],taskvec2[k]],weights,density) for k in taskvec1.keys()}
        elif mode=='dareties':
            mixvec={k: merge_utils.dare_ties([taskvec1[k],taskvec2[k]],weights,density) for k in taskvec1.keys()}
        elif mode=='darelinear':
            mixvec={k: merge_utils.dare_linear([taskvec1[k],taskvec2[k]],weights,density) for k in taskvec1.keys()}

        print(f"Mixed vectors created: {len(mixvec)} keys")

        # Count what gets merged vs preserved
        merged_from_mixvec = 0
        preserved_from_original = 0

        state_dict1_new = {}
        for k in state_dict_base.keys():
            if k not in excluded_keys and k in mixvec:
                state_dict1_new[k] = state_dict_base[k] + mixvec[k]
                merged_from_mixvec += 1
            else:
                state_dict1_new[k] = state_dict1.get(k, state_dict_base[k])
                preserved_from_original += 1

        state_dict1 = state_dict1_new

        print(f"\n=== TIES MERGE RESULTS ===")
        print(f"Keys merged from mixvec: {merged_from_mixvec}")
        print(f"Keys preserved as original: {preserved_from_original}")
        print(f"Total keys in final model: {len(state_dict1)}")

        # Validation for TIES mode
        if len(taskvec_keys) == merged_from_mixvec:
            print(f"✅ PERFECT TIES MERGE: All task vector keys were merged!")
        else:
            print(f"⚠️  PARTIAL TIES MERGE: {len(taskvec_keys)} task keys vs {merged_from_mixvec} merged")
    else:
        from tqdm import tqdm
        # Get common keys between both models to avoid KeyError
        common_keys = set(state_dict1.keys()) & set(state_dict2.keys())
        print(f"\n=== MODEL MERGE ANALYSIS ===")
        print(f"Model1 keys: {len(state_dict1)}")
        print(f"Model2 keys: {len(state_dict2)}")
        print(f"Common keys: {len(common_keys)}")
        print(f"Excluded keys: {excluded_keys}")

        # Keys that will be processed
        mergeable_keys = common_keys - excluded_keys
        excluded_common_keys = common_keys & excluded_keys

        print(f"\n=== MERGE PLAN ===")
        print(f"Keys to merge: {len(mergeable_keys)}")
        print(f"Keys to exclude (but common): {len(excluded_common_keys)}")

        if excluded_common_keys:
            print(f"Excluded common keys: {sorted(list(excluded_common_keys))}")

        # Show sample of keys that will be merged
        sample_mergeable = sorted(list(mergeable_keys))[:10]
        print(f"Sample mergeable keys: {sample_mergeable}")

        merged_count = 0
        skipped_count = 0

        for layer in tqdm(list(common_keys), desc="Processing layers"):
            layer_number = extract_layer_number(layer)
            if layer not in excluded_keys:
                if mode == 'layerswap':
                    if layer_number is not None and layer_number <= base_layer_num:
                        state_dict1[layer].copy_(state_dict1[layer])  # Keep model1
                    else:
                        state_dict1[layer].copy_(alpha * state_dict1[layer] + (1 - alpha) * state_dict2[layer])
                elif mode == 'base':
                    state_dict1[layer].copy_(alpha * state_dict1[layer] + (1 - alpha) * state_dict2[layer])
                merged_count += 1
            else:
                skipped_count += 1

        print(f"\n=== MERGE RESULTS ===")
        print(f"Successfully merged: {merged_count} layers")
        print(f"Skipped (excluded): {skipped_count} layers")

        # Print detailed analysis of non-matching keys
        only_in_model1 = set(state_dict1.keys()) - set(state_dict2.keys())
        only_in_model2 = set(state_dict2.keys()) - set(state_dict1.keys())

        if only_in_model1:
            print(f"\n=== KEYS ONLY IN MODEL1 ({len(only_in_model1)}) ===")
            if len(only_in_model1) <= 20:
                for key in sorted(only_in_model1):
                    print(f"  - {key}")
            else:
                print(f"First 10: {sorted(list(only_in_model1))[:10]}")
                print(f"Last 10:  {sorted(list(only_in_model1))[-10:]}")

        if only_in_model2:
            print(f"\n=== KEYS ONLY IN MODEL2 ({len(only_in_model2)}) ===")
            if len(only_in_model2) <= 20:
                for key in sorted(only_in_model2):
                    print(f"  - {key}")
            else:
                print(f"First 10: {sorted(list(only_in_model2))[:10]}")
                print(f"Last 10:  {sorted(list(only_in_model2))[-10:]}")

        # Validation check
        if len(common_keys) == len(state_dict1) == len(state_dict2):
            print(f"\n✅ PERFECT MATCH: All keys align between models!")
        else:
            print(f"\n⚠️  PARTIAL MATCH: Only common keys were merged.")

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
