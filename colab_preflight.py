"""GPU preflight cell: verify the largest training example before the long E0 run."""


def preflight_training():
    # Check every input image before any long GPU job, including images in later splits.
    for entries in splits.values():
        for entry in entries:
            with Image.open(DATA_ROOT / entry['image_path']) as image:
                image.load()
                assert image.size == (entry['width'], entry['height']), entry['image_id']
    setups = [name for name in ('E1', 'E2', 'E3') if name in RUNS]
    if 'E4' in RUNS:
        setups += ['E3'] if E4_BASE_SETUP == 'auto' else [E4_BASE_SETUP]
    if not setups:
        print('Preflight: dữ liệu ảnh hợp lệ; không có setup train được chọn.')
        return
    setup = max(setups, key=lambda name: int(name[1:]))
    config = load_config(setup)
    # Detect incompatible old outputs BEFORE doing E0 or allocating the model.
    for name in set(setups):
        candidate = load_config(name)
        load_completed_run(run_directory(name) / 'final', candidate)
        latest_training_checkpoint(run_directory(name), candidate)
    longest = max(train_samples, key=lambda entry: len(processor.tokenizer.encode(
        objects_to_answer(entry['objects']), add_special_tokens=False)))
    model = optimizer = scaler = batch = result = loss = params = None
    try:
        model = apply_setup(load_base_model(), setup, config)
        model.train()
        params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=float(config['lr']), weight_decay=0, foreach=False)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        sample = RoadDataset([longest])[0]
        batch = to_device(collate_fn([sample] * config['batch_size']))
        torch.cuda.reset_peak_memory_stats()
        succeeded = False
        # Permit normal GradScaler backoff, as in the real training loop.
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                result = model(**batch)
                loss = result.loss
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Preflight: loss không hữu hạn.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if torch.isfinite(norm).item():
                scaler.step(optimizer)
                scaler.update()
                succeeded = True
                break
            scaler.update(new_scale=scaler.get_scale() / 2)
            result = loss = None
        if not succeeded:
            raise FloatingPointError('Preflight: gradient overflow sau 8 lần thử.')
        torch.cuda.synchronize()
        status = dict(setup=setup, image_id=longest['image_id'],
                      peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3,
                      model_revision=MODEL_COMMIT, implementation=COLAB_SOURCE_SHA256)
        atomic_json(OUTPUT_ROOT / 'preflight.json', status)
        backup_path(OUTPUT_ROOT / 'preflight.json')
        print('Preflight GPU PASS:', status)
    finally:
        model = optimizer = scaler = batch = result = loss = params = None
        gc.collect()
        torch.cuda.empty_cache()


preflight_training()
E0_RESULT = execute_e0() if 'E0' in RUNS else None
