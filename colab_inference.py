@torch.inference_mode()
def predict_jsonl(model, entries, out_path, manifest):
    """Resume a matching JSONL prefix and back up progress at batch boundaries."""
    import gc
    model.eval()
    model.gradient_checkpointing_disable()
    processor.tokenizer.padding_side = 'left'
    out_path = Path(out_path)
    progress = out_path.with_suffix('.progress.jsonl')
    manifest_path = out_path.with_suffix('.manifest.json')
    progress_manifest_path = out_path.with_suffix('.progress.manifest.json')
    previous_manifest = json.loads(progress_manifest_path.read_text()) if progress_manifest_path.exists() else None
    if (progress.exists() or previous_manifest is not None) and previous_manifest != manifest:
        raise RuntimeError(f'{progress}: inference config đã đổi. Chọn RUN_LABEL mới.')
    expected_ids = [e['image_id'] for e in entries]
    completed = []
    if progress.exists():
        lines = progress.read_bytes().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                if index == len(lines) - 1:
                    break  # A disconnect can leave the last write incomplete.
                raise
            if index >= len(entries) or row.get('image_id') != expected_ids[index] or not isinstance(row.get('raw'), str):
                raise RuntimeError(f'{progress}: IDs không khớp thứ tự inference.')
            completed.append(row)
    progress.parent.mkdir(parents=True, exist_ok=True)
    temporary = progress.with_name(progress.name + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in completed), encoding='utf-8')
    temporary.replace(progress)
    atomic_json(progress_manifest_path, manifest)
    backup_path(progress_manifest_path)
    last_backup = time.perf_counter()
    batch_size = INFER_BATCH_SIZE

    def generate_batch(rows):
        messages = [[user_message(load_image(e))] for e in rows]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt', padding=True)
        inputs.pop('token_type_ids', None)
        inputs = to_device(inputs)
        with torch.autocast('cuda', dtype=torch.float16):
            generated = model.generate(**inputs, do_sample=False, use_cache=True,
                                       max_new_tokens=MAX_NEW_TOKENS,
                                       pad_token_id=processor.tokenizer.pad_token_id)
        decoded = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:],
                                         skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if len(decoded) != len(rows):
            raise RuntimeError('Số output generation không khớp số ảnh.')
        return decoded

    with progress.open('a', encoding='utf-8') as stream:
        try:
            start = len(completed)
            while start < len(entries):
                rows = entries[start:start + batch_size]
                retry = False
                try:
                    raw = generate_batch(rows)
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    retry = True
                if retry:
                    # Leave the exception scope before freeing tensors retained by its traceback.
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f'Inference thiếu VRAM: giảm batch xuống {batch_size}, thử lại cùng ID.')
                    continue
                for entry, text in zip(rows, raw):
                    stream.write(json.dumps({'image_id': entry['image_id'], 'raw': text}, ensure_ascii=False) + '\n')
                stream.flush()
                now = time.perf_counter()
                if start == len(completed) or now - last_backup >= PREDICTION_BACKUP_SECONDS:
                    backup_path(progress)
                    last_backup = time.perf_counter()
                start += len(rows)
                print(f'Inference {manifest["split"]}: {start}/{len(entries)}')
        finally:
            stream.flush()
            backup_path(progress)
    progress.replace(out_path)
    atomic_json(manifest_path, manifest)
    assert valid_prediction_file(out_path, entries, manifest)
    backup_path(out_path)
    backup_path(manifest_path)
