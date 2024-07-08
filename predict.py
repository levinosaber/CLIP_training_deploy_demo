from PIL import Image
import torch
import numpy as np

from clip_inference import CLIP
from utils.utils import get_configs
from model.simple_tokenizer import tokenize, SimpleTokenizer

if __name__ == "__main__":

    clip = CLIP()

    # image path
    # image_path = "12dfb53aac8cbb0d907e964d4553c26f.png"
    # find corresponding caption
    # captions   = [
    #     "a picture of dog",
    #     "a picture of cat",
    #     "a picture of bird",
    #     "a picture of chimpanzee"
    # ]
    # image_path = "smaug.jpg"
    # captions   = [
    #     "a picture of dog",
    #     "a picture of devil dragon",
    #     "a picture of bird",
    #     "a picture of chimpanzee"
    # ]

    # image_path = "dvalin.png"
    # captions   = [
    #     "a picture of dog",
    #     "a picture of dragon",
    #     "a picture of bird",
    #     "a picture of chimpanzee"
    # ]

    # image_path = "904404695.jpg"
    # captions   = [
    #     "a picture of dog",
    #     "a picture of human",
    #     "a picture of crocodile",
    #     "a picture of chimpanzee"
    # ]

    # image_path = "904404695.jpg"
    # captions   = [
    #     "a picture of dog",
    #     "a picture of human lying in the water",
    #     "a picture of crocodile",
    #     "a picture of chimpanzee"
    # ]

    image_path = "4604111022.jpg"
    captions   = [
        "a picture of dog",
        "a picture of human",
        "a picture of giraffe",
        "a picture of chimpanzee"
    ]

    # image_path = "4604111022.jpg"
    # captions   = [
    #     "a picture of dog",
    #     "a picture of human eating food",
    #     "a picture of giraffe",
    #     "a picture of chimpanzee"
    # ]

    _tokenizer = SimpleTokenizer()

    captions = tokenize(_tokenizer, captions, truncate=True)

    image = Image.open(image_path)
    probs = clip.detect_image(image, captions)
    print("Label probs:", probs)