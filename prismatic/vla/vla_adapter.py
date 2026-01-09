import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor


def load_vla_adapter(
    vlm_path,
    config_file_path,
    device: torch.device,
    num_images_in_input: int,
    use_flash_attention_2,
):
    processor = AutoProcessor.from_pretrained(config_file_path, trust_remote_code=False)
    from prismatic.models import load

    vlm = load(
        vlm_path,
        hf_token="",
        load_for_training=True,
        use_flash_attention_2=use_flash_attention_2,
    )
    config = AutoConfig.from_pretrained(config_file_path, trust_remote_code=False)
    vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device)
    replace_map = [
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
        ("llm_backbone.llm", "language_model"),
        ("projector.projector.0", "projector.fc1"),
        ("projector.projector.2", "projector.fc2"),
        ("projector.projector.4", "projector.fc3"),
        ("gamma", "scale_factor"),
    ]

    def rename_state_dict_keys(state_dict, replace_map):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for old, new in replace_map:
                if old in new_k:
                    new_k = new_k.replace(old, new)
            new_state_dict[new_k] = v
        return new_state_dict

    old_state_dict = vlm.state_dict()
    RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
    vla.load_state_dict(RAW_STATE_DICT, strict=False)
    del old_state_dict
    vla.vision_backbone.set_num_images_in_input(num_images_in_input)
    return processor, vla, RAW_STATE_DICT
