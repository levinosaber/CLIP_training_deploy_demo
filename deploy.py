from PIL import Image
import torch
import numpy as np
import tensorrt as trt
import os
import onnx

from clip_inference import CLIP
from utils.utils import get_configs
from model.simple_tokenizer import tokenize, SimpleTokenizer


def onnx2trt(onnx_file_path, trt_file_path = None):

    if trt_file_path is None:
        trt_file_path = onnx_file_path.replace(".onnx", ".trt")

    onnx.checker.check_model(onnx_file_path)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as f_model:
        if not parser.parse(f_model.read()):
            print("ERROR: Failed to parse the ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
        else:
            config = builder.create_builder_config()

            profile = builder.create_optimization_profile()

            input_name = "images"
            min_shape = (1, 3, 224, 224)
            opt_shape = (16, 3, 224, 224)
            max_shape = (32, 3, 224, 224)
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            serialized_engine = builder.build_serialized_network(network, config)

            with open(trt_file_path, "wb") as f_engine:
                f_engine.write(serialized_engine)
    return

if __name__ == "__main__":

    clip = CLIP()
    image_path = "4604111022.jpg"
    captions   = [
        "a picture of dog",
        "a picture of human eating food",
        "a picture of giraffe",
        "a picture of chimpanzee"
    ]
    _tokenizer = SimpleTokenizer()
    captions = tokenize(_tokenizer, captions, truncate=True)
    clip.export_onnx(image_path, captions)

    os.environ['CUDA_MODULE_LOADING'] = "LAZY"
    onnx2trt("best_epoch_weights.onnx")
    
