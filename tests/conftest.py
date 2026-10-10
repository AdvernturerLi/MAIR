import sys
from pathlib import Path

# 保证 MAIR 根目录在 import 路径上
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
