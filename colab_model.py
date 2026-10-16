"""Download the pinned Qwen snapshot once; all subsequent model loads are local."""
import json
import re
import time
from pathlib import Path


def retry_download(operation, attempts=5):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            chain, transient = error, False
            seen = set()
            while chain is not None and id(chain) not in seen:
                seen.add(id(chain))
                status = getattr(getattr(chain, 'response', None), 'status_code', None)
                transient |= status in (408, 429, 500, 502, 503, 504)
                transient |= isinstance(chain, (TimeoutError, ConnectionError))
                transient |= type(chain).__name__ in (
                    'ConnectError', 'ReadError', 'TimeoutException', 'ConnectTimeout',
                    'ReadTimeout', 'WriteTimeout', 'RemoteProtocolError')
                chain = chain.__cause__ or chain.__context__
            if not transient or attempt + 1 == attempts:
                raise
            delay = min(2 ** (attempt + 1), 30)
            print(f'Hugging Face tạm thời không truy cập được; thử lại sau {delay}s.')
            time.sleep(delay)


def prepare_model_snapshot(model_id, revision, cache_dir, token=None):
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    def fetch(filename, pinned_revision):
        if re.fullmatch(r'[0-9a-f]{40}', pinned_revision):
            cached = try_to_load_from_cache(model_id, filename, revision=pinned_revision, cache_dir=cache_dir)
            if isinstance(cached, str) and Path(cached).is_file() and Path(cached).stat().st_size > 0:
                return Path(cached)
        return Path(retry_download(lambda: hf_hub_download(
            model_id, filename, revision=pinned_revision, cache_dir=cache_dir, token=token)))

    config_path = fetch('config.json', revision)
    commit = config_path.parent.name
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise RuntimeError(f'Không xác định được model commit từ snapshot: {config_path.parent}')
    files = ('generation_config.json', 'preprocessor_config.json', 'video_preprocessor_config.json',
             'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt',
             'chat_template.json', 'model.safetensors.index.json')
    for filename in files:
        fetch(filename, commit)
    index = json.loads((config_path.parent / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    shards = sorted(set(index['weight_map'].values()))
    if not shards:
        raise RuntimeError('Weight index không có shard.')
    for shard in shards:
        if Path(shard).name != shard or not shard.endswith('.safetensors'):
            raise ValueError(f'Tên model shard không hợp lệ: {shard}')
        fetch(shard, commit)
    print('Model snapshot sẵn sàng để nạp offline:', commit)
    return config_path.parent, commit
