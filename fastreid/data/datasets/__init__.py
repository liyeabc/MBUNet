# encoding: utf-8


from fastreid.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""


from MBU_reid.celeb_reid_light import celeb_reid_light
from MBU_reid.celeb import celeb_reid
from MBU_reid.prcc import PRCC
