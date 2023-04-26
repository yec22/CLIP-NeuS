import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("/data/yesheng/data/scannet/scans/scene0050_00/color/0.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a piano", "a dog", "a cat", "a sofa"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    print(abs(image_features - text_features[0]).mean())
    print(abs(image_features - text_features[1]).mean())
    print(abs(image_features - text_features[2]).mean())
    print(abs(image_features - text_features[3]).mean())

    logits_per_image, logits_per_text = model(image, text)
    print(logits_per_image)
    print(logits_per_text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image1 = preprocess(Image.open("novel_view.png")).unsqueeze(0).to(device)
# image2 = preprocess(Image.open("mask_000000.png")).unsqueeze(0).to(device)
# image3 = preprocess(Image.open("mask_000002.png")).unsqueeze(0).to(device)

# with torch.no_grad():
#     image_features1 = model.encode_image(image2)
#     image_features2 = model.encode_image(image3)

#     diff = abs(image_features1 - image_features2).mean()
#     print(diff)
