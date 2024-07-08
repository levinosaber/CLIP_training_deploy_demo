from PIL import Image
import torch
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os 

from utils.utils import cvtColor, letterbox_image, get_configs, preprocess_input
from model.simple_tokenizer import SimpleTokenizer, tokenize

def inference_trt(engine_path, image_path, captions, phi="openai/VIT-B-32", letterbox_image_flag = False):
    os.environ['CUDA_MODULE_LOADING'] = "LAZY"

    image = cvtColor(Image.open(image_path))
    config = get_configs(phi)
    image_data = letterbox_image(image, [config["input_resolution"], config["input_resolution"]], letterbox_image_flag)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    image_data = np.ascontiguousarray(image_data)
    captions = np.array(captions, dtype=np.int64)

    imgs_number = image_data.shape[0]
    captions_number = len(captions)

    # trt infer
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as engine_f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_f.read())
        context = engine.create_execution_context()

        context.set_input_shape("images", (imgs_number, 3, 224, 224))
        assert context.all_binding_shapes_specified

        # input 
        trt_input_image_data = cuda.mem_alloc(image_data.nbytes)
        trt_input_text_data = cuda.mem_alloc(captions.nbytes)
        cuda.memcpy_htod(trt_input_image_data, image_data)
        cuda.memcpy_htod(trt_input_text_data, captions)



        output_1 = np.empty((imgs_number, captions_number), dtype=np.float32)  # output named logits_per_image
        output_2 = np.empty((captions_number, imgs_number), dtype=np.float32)  # output named logits_per_text
        trt_output_1 = cuda.mem_alloc(output_1.nbytes)
        trt_output_2 = cuda.mem_alloc(output_2.nbytes)

        bindings = [int(trt_input_image_data), int(trt_input_text_data), int(trt_output_1), int(trt_output_2)]
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(output_1, trt_output_1)
        cuda.memcpy_dtoh(output_2, trt_output_2)

        return output_1, output_2
    

if __name__ == "__main__":
    engine_path = "best_epoch_weights.trt"
    image_path = "4604111022.jpg"
    captions   = [
        "a picture of dog",
        "a picture of human eating food",
        "a picture of giraffe",
        "a picture of chimpanzee"
    ]
    phi = "openai/VIT-B-32"

    captions = tokenize(SimpleTokenizer(), captions, truncate=True)

    logits_per_image, logits_per_text = inference_trt(engine_path, image_path, captions, phi)

    print(logits_per_image)
    print(logits_per_text)

    probs = torch.tensor(logits_per_image).softmax(dim=-1).cpu().numpy()

    print(f"final probs: {probs}")