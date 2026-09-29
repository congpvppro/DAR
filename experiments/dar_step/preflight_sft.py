"""Encode EVERY row with the actual ms-swift template before expensive SFT."""
import argparse
import copy
import os

from common import digest, dump, read_jsonl, require

# These settings must match train.sh. Irregular keyframes are multi-image inputs.
VISION_ENV = dict(FPS_MIN_FRAMES='16', FPS_MAX_FRAMES='16', VIDEO_MIN_PIXELS='100352',
                  VIDEO_MAX_PIXELS='100352', VIDEO_MIN_TOKEN_NUM='128', VIDEO_MAX_TOKEN_NUM='128',
                  MIN_PIXELS='100352', MAX_PIXELS='100352', IMAGE_MIN_TOKEN_NUM='128', IMAGE_MAX_TOKEN_NUM='128')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'dataset', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--max-length', type=int, default=16384)
    args = parser.parse_args()
    os.environ.update(VISION_ENV)
    import swift
    from swift.llm import get_model_tokenizer, get_template
    require(swift.__version__ == '3.12.5', 'Use the pinned student runtime')
    _, processor = get_model_tokenizer(args.model, model_type='qwen2_5_vl',
                                      load_model=False, download_model=False)
    template = get_template('qwen2_5_vl', processor, max_length=None)
    template.set_mode('train')
    rows, lengths = read_jsonl(args.dataset), []
    for row in rows:
        encoded = template.encode(copy.deepcopy(row))
        length = len(encoded['input_ids'])
        require(any(x != -100 for x in encoded['labels']), 'Target missing from loss')
        if row.get('images'):
            require(encoded['image_grid_thw'].shape[0] == len(row['images']), 'Image count differs')
        lengths.append(dict(id=row['id'], tokens=length))
        print(row['id'], length, flush=True)
    overflow = [x for x in lengths if x['tokens'] > args.max_length]
    dump(args.output, dict(dataset_sha256=digest(rows), max_length=args.max_length,
                          vision_env=VISION_ENV, lengths=lengths, overflow=overflow))
    require(not overflow, f'{len(overflow)} rows exceed context. Adjust graph/frame budget explicitly; no silent truncation.')


if __name__ == '__main__':
    main()
