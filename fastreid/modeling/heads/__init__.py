

from .build import REID_HEADS_REGISTRY, build_reid_heads

# import all the meta_arch, so they will be registered
from .linear_head import LinearHead
from .bnneck_head import BNneckHead
