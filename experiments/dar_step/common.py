"""Small, dependency-free contracts shared by STEP-inspired data stages."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Callers use fresh run directories. A failed write never becomes a valid cache.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def parse(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    require(isinstance(value, dict), 'Expected JSON object')
    return value


def bounded_text(value, limit=160):
    require(isinstance(value, str) and 0 < len(value.strip()) <= limit, 'Invalid/too long text')
    require('<image>' not in value and '<video>' not in value, 'Reserved media token in graph text')
    return value


def number(value):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value), 'Expected finite number')
    return float(value)
