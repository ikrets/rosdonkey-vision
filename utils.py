import json
import subprocess
from copy import copy
import sys

from pathlib import Path
from typing import Dict, Any


def save_run_parameters(path: Path, parameters: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)

    parameters = copy(parameters)
    with (path / 'parameters.json').open('w') as fp:
        parameters['commit'] = (
            subprocess.run('git rev-parse HEAD', shell=True, stdout=subprocess.PIPE)
            .stdout.decode('ascii')
            .strip()
        )
        parameters['script'] = sys.argv[0]
        json.dump(parameters, fp, indent=4)
