import numpy as np
import torch
from PIL import Image

from model.clip import CLIP as CLIP_Model
from utils.utils import (cvtColor, get_configs, letterbox_image,
                         preprocess_input)


class CLIP(object):
    _defaults = {
        #--------------------------------------------------------------------#
        #   指向logs文件夹下的权值文件 reference to the weights file under the logs folder
        #--------------------------------------------------------------------#
        "model_path"        : 'best_epoch_weights.pth',
        #----------------------------------------------------------------------------------------------------------------------------------------#
        #   模型的种类  Model Type
        #   openai/VIT-B-16为openai公司开源的CLIP模型中，VIT-B-16规模的CLIP模型，英文文本与图片匹配，有公开预训练权重可用。 openal/VIT-B-16 is the openai company's open source CLIP model, VIT-B-16 scale CLIP model, English text and image matching, with public pre-training weights available.
        #   openai/VIT-B-32为openai公司开源的CLIP模型中，VIT-B-32规模的CLIP模型，英文文本与图片匹配，有公开预训练权重可用。 openai/VIT-B-32 is the openai company's open source CLIP model, VIT-B-32 scale CLIP model, English text and image matching, with public pre-training weights available.
        #   self-cn/VIT-B-32为自实现的模型，VIT-B-32规模的CLIP模型，英文文本与图片匹配，中文文本与图片匹配，无公开预训练权重可用  self-cn/VIT-B-32 is a self-implemented model, VIT-B-32 scale CLIP model, English text and image matching, Chinese text and image matching, no public pre-training weights available
        #-----------------------------------------------------------------------------------------------------------------------------------------#
        "phi"               : "openai/VIT-B-32",
        #--------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize  This variable is used to control whether to use letterbox_image to resize the input image without distortion
        #   否则对图像进行CenterCrop                                        Otherwise, the image is CenterCrop
        #--------------------------------------------------------------------#
        "letterbox_image"   : False,
        #--------------------------------------------------------------------#
        #   是否使用Cuda                                      Whether to use Cuda
        #   没有GPU可以设置成False                            No GPU can be set to False
        #--------------------------------------------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化CLIP  Initialize CLIP
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()

    #---------------------------------------------------#
    #   生成模型   Generate model
    #---------------------------------------------------#
    def generate(self):
        self.config = get_configs(self.phi)

        self.net    = CLIP_Model(**self.config)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            # self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
            
    #---------------------------------------------------#
    #   检测图片   Detect image
    #---------------------------------------------------#
    def detect_image(self, image, texts):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。  Here the image is converted to an RGB image to prevent an error when predicting a grayscale image.
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB   The code only supports the prediction of RGB images, so all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   CenterCrop
        #---------------------------------------------------------#
        image_data  = letterbox_image(image, [self.config['input_resolution'], self.config['input_resolution']], self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度   Add the batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！  Input the image into the network for prediction!
            #---------------------------------------------------------#
            logits_per_image, logits_per_text = self.net(images, texts)
            
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        return probs
    
    #---------------------------------------------------#
    #   检测图片   Detect image
    #---------------------------------------------------#
    def detect_image_for_eval(self, images=None, texts=None):
        with torch.no_grad():
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！   Input the image into the network for prediction!
            #---------------------------------------------------------#
            if images is not None:
                images_feature = self.net.encode_image(images)
            else:
                images_feature = None
                
            if texts is not None:
                texts_feature = self.net.encode_text(texts)
            else:
                texts_feature = None
        
        return images_feature, texts_feature
    

    def export_onnx(self, example_image_path, example_text, onnx_path=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if onnx_path is None:
            onnx_path = self.model_path.split(".")[0] + ".onnx"
        with torch.no_grad():
            image = cvtColor(Image.open(example_image_path))
            image_data  = letterbox_image(image, [self.config['input_resolution'], self.config['input_resolution']], self.letterbox_image)
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
            images = torch.from_numpy(image_data).to(device)
            torch.onnx.export(
                self.net,
                (images, example_text),
                onnx_path,  
                opset_version=14,  # version 13 can not work
                input_names=['images', 'texts'],
                output_names=['logits_per_image', 'logits_per_text'],
                dynamic_axes={
                    "images": {0: "batch_size"},
                    # "texts": {0: "batch_size"},
                    "logits_per_image": {0: "number_of_images"},
                    "logits_per_text": { 1:"number_of_images"},
                }                
            )