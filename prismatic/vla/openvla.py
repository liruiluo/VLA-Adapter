from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from ..extern.hf.configuration_prismatic import OpenVLAConfig
from ..extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from ..extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

def register_openvla():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)