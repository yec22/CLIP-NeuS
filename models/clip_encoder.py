import torch
import clip
from PIL import Image
from torchvision import transforms

class CLIP_Encoder():
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.device = device
        self.preprocess = transforms.Compose([
                transforms.Resize(224, Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
        for param in self.model.parameters():
            param.requires_grad = False

    def CLIP_Encode(self, image):
        image = torch.flip(image, dims=[0]) # BGR2RGB
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_feature = self.model.encode_image(image)

        return image_feature