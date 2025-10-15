import os
import re
from typing import Any, Dict

import yaml

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate(value: Any, context: Dict[str, Any]) -> Any:
	if isinstance(value, str):
		def repl(match: re.Match[str]) -> str:
			path = match.group(1).split(".")
			ref = context
			for key in path:
				if not isinstance(ref, dict) or key not in ref:
					raise KeyError(f"Missing config reference: {match.group(1)}")
				ref = ref[key]
			return str(ref)
		return _VAR_PATTERN.sub(repl, value)
	elif isinstance(value, dict):
		return {k: _interpolate(v, context) for k, v in value.items()}
	elif isinstance(value, list):
		return [_interpolate(v, context) for v in value]
	return value


def load_config(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		cfg: Dict[str, Any] = yaml.safe_load(f)
	# First pass interpolation using the full dict context (supports forward refs inside same section)
	interpolated = _interpolate(cfg, cfg)
	# Expand user and env vars in paths
	paths = interpolated.get("paths", {})
	for k, v in list(paths.items()):
		if isinstance(v, str):
			v = os.path.expandvars(os.path.expanduser(v))
			paths[k] = v
	interpolated["paths"] = paths
	return interpolated
