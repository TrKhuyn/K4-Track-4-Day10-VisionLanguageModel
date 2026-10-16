"""Sync embedded support cells, validate notebook source, and package Colab files.

Usage: python prepare_colab.py [--check] [--bundle work/colab_source.zip]
Instructor-owned scripts/, configs/ and tests/ are never edited.
"""
import argparse
import ast
import copy
import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Chỉ hash mã ảnh hưởng trực tiếp model/train/inference. Cell recovery chỉ sao
# chép artifact đã hoàn tất, nên không được làm checkpoint E1--E3 cũ mất hiệu lực.
SUPPORT_FILES = ('colab_storage.py', 'colab_training.py', 'colab_inference.py',
                 'colab_model.py', 'colab_preflight.py')


def synced_notebook(notebook, root=ROOT):
    digest = hashlib.sha256(''.join((root / p).read_text(encoding='utf-8') for p in SUPPORT_FILES).encode()).hexdigest()
    for index, cell in enumerate(notebook['cells']):
        cell.setdefault('id', f'day10-cell-{index:02d}')
        source = ''.join(cell['source'])
        embedded = cell['metadata'].get('colab_source')
        if embedded:
            source = (root / embedded).read_text(encoding='utf-8')
        replacements = cell['metadata'].get('colab_function_sources', {})
        if replacements:
            nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for name, filename in sorted(replacements.items(), key=lambda item: nodes[item[0]].lineno, reverse=True):
                node = nodes[name]
                start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                lines = source.splitlines(keepends=True)
                lines[start:node.end_lineno] = [(root / filename).read_text(encoding='utf-8').rstrip() + '\n']
                source = ''.join(lines)
        if cell['metadata'].get('colab_config'):
            source = re.sub(r"COLAB_SOURCE_SHA256 = '[^']*'", f"COLAB_SOURCE_SHA256 = '{digest}'", source)
        if source != ''.join(cell['source']):
            cell['source'] = source.splitlines(keepends=True)
            if cell['cell_type'] == 'code':
                cell['execution_count'], cell['outputs'] = None, []
        if cell['cell_type'] == 'code':
            ast.parse(source, filename=f'lab_day10.ipynb:cell-{index}')
    return notebook


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if embedded sources are stale; do not write.')
    parser.add_argument('--bundle', type=Path, help='ZIP of source files (no data, weights, tokens or outputs).')
    parser.add_argument('--clean-notebook', type=Path, help='Export a small Colab notebook without stale outputs/widgets.')
    args = parser.parse_args(argv)
    path = ROOT / 'lab_day10.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    before = json.dumps(notebook, ensure_ascii=False)
    synced_notebook(notebook)
    if args.check and json.dumps(notebook, ensure_ascii=False) != before:
        parser.error('Notebook support cells are stale. Run python prepare_colab.py first.')
    if not args.check:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if args.bundle or args.clean_notebook:
        source_notebook = copy.deepcopy(notebook)
        source_notebook['metadata'].pop('widgets', None)
        for cell in source_notebook['cells']:
            if cell['cell_type'] == 'code':
                cell['execution_count'], cell['outputs'] = None, []
        clean_source = json.dumps(source_notebook, ensure_ascii=False, indent=1)
        if args.clean_notebook:
            args.clean_notebook.parent.mkdir(parents=True, exist_ok=True)
            args.clean_notebook.write_text(clean_source + '\n', encoding='utf-8')
            print('Clean notebook:', args.clean_notebook)
    if args.bundle:
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py', '.txt', '.md')]
        for folder in ('scripts', 'configs', 'tests', 'student_checks'):
            files.extend(p for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.yaml') and '__pycache__' not in p.parts)
        files.append(ROOT / 'pytest.ini')
        with zipfile.ZipFile(args.bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(set(files)):
                archive.write(file, file.relative_to(ROOT))
            # The instructor tests read this file. Keep the bundle small and omit old outputs.
            archive.writestr('lab_day10.ipynb', clean_source)
            archive.write(ROOT / 'lab_day10_e4_recovery.ipynb', 'lab_day10_e4_recovery.ipynb')
        print('Source bundle:', args.bundle)
    print('Notebook source validated; support cells are synchronized.')


if __name__ == '__main__':
    main()
