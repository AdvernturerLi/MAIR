"""Find the shared reproduction workspace without embedding a personal path."""
import os
from pathlib import Path

def workspace_root():
 override=os.environ.get('IMAGE_RECONSTRUCT_ROOT')
 if override:return Path(override).expanduser().resolve()
 for parent in Path(__file__).resolve().parents:
  if (parent/'MAIR').is_dir() and (parent/'AgenticIR').is_dir():return parent
 raise RuntimeError('Set IMAGE_RECONSTRUCT_ROOT to the workspace containing MAIR, AgenticIR and experiments.')
