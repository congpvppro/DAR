"""Native VILA 1.5 adapter; source revision and runtime are pinned separately."""
from pathlib import Path

from common import require

VILA_REVISION = '6b941da19e31ddfdfaa60160908ccf0978d96615'


class Vila:
    def __init__(self, model_path, device='cuda'):
        import torch
        import transformers
        from llava.model.builder import load_pretrained_model
        require(transformers.__version__ == '4.36.2', 'VILA requires its isolated Transformers 4.36.2 runtime')
        require(Path(model_path).is_dir(), 'Attach a complete local VILA1.5-3b snapshot')
        self.torch = torch
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            str(model_path), 'VILA1.5-3b', device=device, device_map=device,
            attn_implementation='eager', model_max_length=4096)
        self.model.eval()
        require(not self.model.config.s2, 'This adapter is for VILA1.5-3b, not 3b-s2')
        self.context = min(4096, self.model.llm.config.max_position_embeddings)

    def tensors(self, pictures):
        from llava.mm_utils import process_images
        return process_images(pictures, self.processor, self.model.config).to(
            self.model.device, dtype=self.model.dtype)

    def embed(self, pictures):
        # The same penultimate SigLIP patch features used by native VILA.
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(pictures), 4):
                features = self.model.get_vision_tower()(self.tensors(pictures[start:start + 4]))
                outputs.append(features.float().mean(dim=1).cpu())
        return self.torch.cat(outputs).numpy()

    def generate(self, pictures, prompt, max_tokens=900):
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        require(0 < len(pictures) <= 4, 'VILA calls use at most four keyframes')
        # Explicit --conv-mode from the official VILA1.5-3B README example.
        conversation = conv_templates['vicuna_v1'].copy()
        conversation.append_message(conversation.roles[0], (DEFAULT_IMAGE_TOKEN + '\n') * len(pictures) + prompt)
        conversation.append_message(conversation.roles[1], None)
        text = conversation.get_prompt()
        ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors='pt').unsqueeze(0).to(self.model.device)
        pixels = self.tensors(pictures)
        with self.torch.inference_mode():
            # Check exact image-token expansion before native code can truncate it.
            visual_tokens = self.model.encode_images(pixels).shape[1]
        expanded = ids.shape[1] + len(pictures) * (visual_tokens - 1)
        require(expanded + max_tokens <= self.context,
                f'Teacher context exceeded: {expanded}+{max_tokens}>{self.context}')
        stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
        with self.torch.inference_mode():
            result = self.model.generate(ids, images=[pixels], do_sample=False,
                num_beams=1, max_new_tokens=max_tokens, use_cache=True,
                stopping_criteria=[KeywordsStoppingCriteria([stop], self.tokenizer, ids)],
                pad_token_id=self.tokenizer.eos_token_id)
        # Native VILA uses inputs_embeds. Transformers 4.36.2 prepends its dummy
        # BOS (not the text prompt) to the generated IDs; exclude it from counts.
        require(int(result[0, 0]) == self.tokenizer.bos_token_id, 'Unexpected native generation prefix')
        completion = result[:, 1:]
        raw = self.tokenizer.batch_decode(completion, skip_special_tokens=True)[0].strip()
        if raw.endswith(stop):
            raw = raw[:-len(stop)].strip()
        return dict(raw=raw, generated_tokens=int(completion.shape[1]), input_tokens=int(expanded),
                    hit_token_limit=bool(completion.shape[1] >= max_tokens))
