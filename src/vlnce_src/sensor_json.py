import json
import os
from pathlib import Path
import tempfile

import numpy as np


def json_compatible(value):
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode('utf-8', errors='backslashreplace')
    if isinstance(value, dict):
        return {json_compatible(k): json_compatible(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    return value


def write_sensor_json(path, info):
    
    payload = json.dumps(json_compatible(info))
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
