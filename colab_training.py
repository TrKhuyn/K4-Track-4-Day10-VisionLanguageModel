"""Training cell source embedded by prepare_colab.py into lab_day10.ipynb.

Execute in the notebook namespace after data/model/LoRA cells. This file keeps
the standalone notebook's training implementation reviewable without outputs.
"""
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict


def select_training_entries(entries, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('train_fraction phải trong (0,1].')
    indices = random.Random(seed).sample(range(len(entries)), len(entries))
    return [entries[i] for i in indices[:max(1, round(len(entries) * fraction))]]


def load_completed_run(directory, config):
    directory = Path(directory)
    record_path = directory / 'run.json'
    if not record_path.exists():
        return None
    try:
        run = json.loads(record_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(run, dict):
        return None
    if run.get('checkpoint_format', 1) >= 2 and not valid_checkpoint(directory):
        return None
    if not run.get('fingerprint'):
        return None
    if run.get('fingerprint') != config_fingerprint(config):
        raise RuntimeError(f'{directory}: config/model/dataset đã thay đổi. Dùng RUN_LABEL mới.')
    if not run.get('complete') or run['completed_steps'] != config['max_steps']:
        return None
    if not (directory / 'adapter/adapter_config.json').exists():
        raise RuntimeError(f'{directory}: thiếu adapter_config.json.')
    if not any((directory / 'adapter').glob('*.safetensors')):
        raise RuntimeError(f'{directory}: thiếu trọng số adapter.')
    if config['train_projector'] and not (directory / 'extra_trainable.pt').exists():
        raise RuntimeError(f'{directory}: thiếu projector.')
    return run


def save_checkpoint(model, directory, run, history, training_state):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CHECKPOINT_MARKER).unlink(missing_ok=True)
    model.save_pretrained(directory / 'adapter', save_embedding_layers=False)
    processor.save_pretrained(directory / 'processor')
    if run['config']['train_projector']:
        extra = {n: p.detach().float().cpu().contiguous()
                 for n, p in model.get_base_model().named_parameters() if is_projector(n)}
        assert extra
        temporary = directory / 'extra_trainable.pt.tmp'
        torch.save(extra, temporary)
        temporary.replace(directory / 'extra_trainable.pt')
    temporary = directory / 'training_state.pt.tmp'
    torch.save(training_state, temporary)
    temporary.replace(directory / 'training_state.pt')
    (directory / 'loss.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history), encoding='utf-8')
    shutil.copy2(directory.parent / 'train_image_ids.json', directory / 'train_image_ids.json')
    atomic_json(directory / 'run.json', run)
    seal_checkpoint(directory, run)
    backup_path(directory)
    STORE.prune_snapshots(directory.parent, keep=KEEP_STEP_CHECKPOINTS)
    print('Checkpoint đã lưu và kiểm tra:', directory)


def latest_training_checkpoint(directory, config):
    expected = config_fingerprint(config)
    candidates = sorted(Path(directory).glob('step-*'), reverse=True)
    for path in candidates:
        if not valid_checkpoint(path):
            continue
        run = json.loads((path / 'run.json').read_text(encoding='utf-8'))
        if run['fingerprint'] != expected:
            raise RuntimeError('Checkpoint khác cấu hình. Chọn RUN_LABEL mới.')
        return path, run
    return None, None


def train_experiment(config):
    config = copy.deepcopy(config)
    directory = run_directory(config['name'])
    final = directory / 'final'
    completed = load_completed_run(final, config)
    if completed:
        print('Dùng lại checkpoint hoàn tất:', final)
        backup_path(final)
        return final
    checkpoint, previous_run = latest_training_checkpoint(directory, config) if RESUME_TRAINING else (None, None)
    if checkpoint is None and directory.exists() and any(directory.iterdir()):
        if not RESTART_INCOMPLETE:
            raise RuntimeError(f'{directory}: không có checkpoint resume hợp lệ. Bật RESTART_INCOMPLETE để train lại.')
        # Do not overwrite an old run with a different identity.
        for path in directory.glob('step-*/run.json'):
            try:
                saved_run = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if not isinstance(saved_run, dict):
                continue
            if saved_run.get('fingerprint') != config_fingerprint(config):
                raise RuntimeError('Run cũ khác cấu hình. Chọn RUN_LABEL mới.')
        print('Không có trạng thái resume hợp lệ: bắt đầu từ bước 0.')
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(config['seed'])
    rows = select_training_entries(train_samples, config['train_fraction'], config['seed'])
    atomic_json(directory / 'train_image_ids.json', [e['image_id'] for e in rows])
    model = optimizer = scheduler = scaler = params = iterator = saved = batch = output = loss = None
    history = []
    step = skipped = consecutive_skips = batches_in_epoch = 0
    run_id = uuid.uuid4().hex
    previous_minutes = previous_peak = 0
    try:
        model = apply_setup(load_base_model(), config['setup'], config)
        if checkpoint:
            result = set_peft_model_state_dict(model, load_peft_weights(str(checkpoint / 'adapter'), device='cpu'))
            assert not result.unexpected_keys, result.unexpected_keys
            if config['train_projector']:
                extra = torch.load(checkpoint / 'extra_trainable.pt', map_location='cpu', weights_only=True)
                parameters = dict(model.get_base_model().named_parameters())
                assert set(extra) == {n for n in parameters if is_projector(n)}
                with torch.no_grad():
                    for name, value in extra.items():
                        assert parameters[name].shape == value.shape
                        parameters[name].copy_(value.to(parameters[name].device))
                del extra, parameters
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=float(config['lr']), weight_decay=0, foreach=False)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, int(config['max_steps'] * 0.05)),
            num_training_steps=config['max_steps'])
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        generator = torch.Generator().manual_seed(config['seed'])
        loader = DataLoader(RoadDataset(rows), batch_size=config['batch_size'], shuffle=True,
                            collate_fn=collate_fn, num_workers=0, generator=generator)
        if checkpoint:
            # Only load state from this run's own trusted checkpoint directory.
            saved = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            step, skipped = previous_run['completed_steps'], previous_run['skipped_steps']
            run_id = previous_run['run_id']
            previous_minutes, previous_peak = previous_run['train_minutes'], previous_run['peak_vram_gb']
            history = read_jsonl(checkpoint / 'loss.jsonl')
            batches_in_epoch = saved['batches_in_epoch']
            generator.set_state(saved['epoch_generator_state'])
        # Replay the current epoch's deterministic sampler, then restore dropout RNG.
        epoch_generator_state = generator.get_state()
        iterator = iter(loader)
        if saved:
            for _ in range(batches_in_epoch):
                next(iterator)
            random.setstate(saved['rng_python'])
            np.random.set_state(saved['rng_numpy'])
            torch.set_rng_state(saved['rng_torch'])
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
            saved = None
            print('Resume', config['name'], 'từ bước', step)
        (directory / 'loss.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history), encoding='utf-8')
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = last_checkpoint = last_log_backup = time.perf_counter()

        def record(complete):
            torch.cuda.synchronize()
            return {'config': config, 'fingerprint': config_fingerprint(config), 'checkpoint_format': 2,
                    'run_id': run_id, 'complete': complete, 'completed_steps': step,
                    'skipped_steps': skipped, 'train_images': len(rows),
                    'trainable_params': sum(p.numel() for p in params),
                    'peak_vram_gb': max(previous_peak, torch.cuda.max_memory_allocated() / 1024**3),
                    'train_minutes': previous_minutes + (time.perf_counter() - started) / 60,
                    'dataset_version': DATASET_VERSION, 'model_id': MODEL_ID,
                    'model_revision': MODEL_COMMIT, 'prompt': PROMPT}

        def snapshot(path, complete=False):
            save_checkpoint(model, path, record(complete), history, {
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'rng_python': random.getstate(),
                'rng_numpy': np.random.get_state(), 'rng_torch': torch.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state_all(),
                'epoch_generator_state': epoch_generator_state,
                'batches_in_epoch': batches_in_epoch,
            })

        while step < config['max_steps']:
            loss_sum = 0.0
            for _ in range(config['grad_accum']):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch_generator_state = generator.get_state()
                    iterator = iter(loader)
                    batches_in_epoch = 0
                    batch = next(iterator)
                batches_in_epoch += 1
                batch = to_device(batch)
                with torch.autocast('cuda', dtype=torch.float16):
                    output = model(**batch)
                    loss = output.loss / config['grad_accum']
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Loss nan/inf: kiểm tra mask/dtype và giảm lr.')
                loss_sum += loss.detach().item()
                scaler.scale(loss).backward()
                batch = output = loss = None
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            used_lr = optimizer.param_groups[0]['lr']
            # A nonfinite norm can result from reduction overflow even if individual gradients are finite.
            if not torch.isfinite(norm).item():
                scaler.update(new_scale=scaler.get_scale() / 2)
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                consecutive_skips += 1
                if consecutive_skips >= 8:
                    raise FloatingPointError('8 bước gradient overflow liên tiếp.')
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            consecutive_skips = 0
            scheduler.step()
            step += 1
            entry = {'step': step, 'loss': loss_sum, 'lr': used_lr,
                     'grad_norm': float(norm),
                     'minutes': previous_minutes + (time.perf_counter() - started) / 60}
            history.append(entry)
            with (directory / 'loss.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry) + '\n')
            now = time.perf_counter()
            if step == 1 or step % 10 == 0:
                print(f'{config["name"]}: {step}/{config["max_steps"]}, loss={loss_sum:.4f}')
            if step == 1 or step % 10 == 0 or now - last_log_backup >= LOG_BACKUP_SECONDS:
                backup_path(directory / 'loss.jsonl')
                atomic_json(directory / 'progress.json', record(False))
                backup_path(directory / 'progress.json')
                last_log_backup = time.perf_counter()
            if step < config['max_steps'] and (
                    step % CHECKPOINT_EVERY_STEPS == 0 or now - last_checkpoint >= CHECKPOINT_EVERY_SECONDS):
                snapshot(directory / f'step-{step:04d}')
                last_checkpoint = time.perf_counter()
        snapshot(final, complete=True)
        return final
    finally:
        model = optimizer = scheduler = scaler = params = iterator = saved = batch = output = loss = None
        gc.collect()
        torch.cuda.empty_cache()
