import json
from pyiqa.default_model_configs import DEFAULT_CONFIGS
import timm
from pyiqa.utils.download_util import DEFAULT_CACHE_DIR
print(json.dumps({x:DEFAULT_CONFIGS[x] for x in ('lpips','maniqa','musiq','clipiqa')},indent=2))
print('CACHE',DEFAULT_CACHE_DIR)
print('TIMM',json.dumps(timm.models.get_pretrained_cfg('vit_base_patch8_224').to_dict(),default=str))
