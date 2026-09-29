"""CPU smoke of native VILA with tiny RANDOM weights; no accuracy claim/download.

Run inside the isolated teacher runtime (PYTHONPATH includes pinned native source).
This exercises SigLIP -> native projector -> Llama generation and the adapter's
multi-image token accounting, without requiring the actual 3B checkpoint.
"""
import os
from pathlib import Path
import tempfile


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        os.environ['HF_HOME'] = str(root / 'hf-cache')
        os.environ['HF_HUB_OFFLINE'] = '1'
        import torch
        import sentencepiece as spm
        from PIL import Image
        from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
        from llava.model.language_model.llava_llama import LlavaLlamaConfig, LlavaLlamaModel
        from llava.model.multimodal_encoder.siglip import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor
        from llava.model.multimodal_projector.base_projector import MultimodalProjectorConfig, MultimodalProjector
        from teacher_vila import Vila

        torch.manual_seed(7)
        torch.set_num_threads(2)
        corpus = root / 'corpus.txt'
        corpus.write_text('A person stands near a red wall. Describe only visible objects. Return JSON.\n' * 50)
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root / 'tokenizer'),
                                      vocab_size=48, hard_vocab_limit=False, minloglevel=2)
        tokenizer = LlamaTokenizer(str(root / 'tokenizer.model'))
        tokenizer.save_pretrained(root / 'llm')
        llm_config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                                max_position_embeddings=4096)
        LlamaForCausalLM(llm_config).save_pretrained(root / 'llm')
        vision_config = SiglipVisionConfig(hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=4, image_size=42, patch_size=14)
        SiglipVisionModel(vision_config).save_pretrained(root / 'vision_tower')
        SiglipImageProcessor(size={'height': 42, 'width': 42}).save_pretrained(root / 'vision_tower')
        projector_config = MultimodalProjectorConfig('mlp_downsample')
        config = LlavaLlamaConfig(llm_cfg=llm_config.to_dict(), vision_tower_cfg=vision_config.to_dict(),
                    mm_projector_cfg=projector_config.to_dict(), hidden_size=32, mm_hidden_size=16,
                    model_dtype='torch.float32', resume_path=str(root), image_aspect_ratio='resize',
                    mm_vision_select_layer=-2, mm_vision_select_feature='cls_patch',
                    mm_use_im_start_end=False, mm_use_im_patch_token=False, s2=False)
        config._name_or_path = str(root)
        config.model_dtype = 'torch.float32'
        MultimodalProjector(projector_config, config).save_pretrained(root / 'mm_projector')
        model = LlavaLlamaModel(config=config, device_map='cpu', attn_implementation='eager',
                               model_max_length=4096).eval()
        teacher = Vila.__new__(Vila)
        teacher.torch, teacher.model, teacher.tokenizer = torch, model, model.tokenizer
        teacher.processor, teacher.context = model.get_vision_tower().image_processor, 4096
        pictures = [Image.new('RGB', (40, 32), 'red'), Image.new('RGB', (40, 32), 'blue')]
        embeddings = teacher.embed(pictures)
        assert embeddings.shape == (2, 16)
        result = teacher.generate(pictures, 'Describe these images.', max_tokens=4)
        assert result['input_tokens'] > 0 and 0 < result['generated_tokens'] <= 4
        print('PASS: native tiny VILA CPU, two-image embed/generate; RANDOM weights, not graph quality')
        print({k: v for k, v in result.items() if k != 'raw'})


if __name__ == '__main__':
    main()
