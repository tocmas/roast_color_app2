from torchvision import transforms
from torchvision.models import mobilenet_v2
import torch
import cv2
import numpy as np


# 前処理
class CustomPreprocessing:
    def __init__(self, blur_size=(7, 7)):
        self.blur_size = blur_size  # ガウシアンブラーのサイズ

    def __call__(self, image):
        image = np.array(image)

        # ホワイトバランス調整
        r, g, b = cv2.split(image)
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]

        # RGBの平均値を計算
        k = (r_avg + g_avg + b_avg) / 3

        # RGB各チャンネルのゲインを計算
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg

        r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

        img_balanced = cv2.merge([r, g, b])

        # 明るさの調整：最大値を基準にする
        max_val = np.max(img_balanced)
        img_max_adjusted = img_balanced * (255.0 / max_val)

        # ガウシアンブラーを追加
        img_blurred = cv2.GaussianBlur(img_max_adjusted, self.blur_size, 5)

        # convert to uint8
        img_blurred = img_blurred.astype(np.uint8)

        return img_blurred

preprocessing = transforms.Compose([
    transforms.Resize((240, 180)),
    transforms.CenterCrop(180), 
    CustomPreprocessing(),
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the correct device
model = mobilenet_v2(pretrained=True)
model = model.to(device)

def predict(input, model):
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        inputs = input.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
    return predicted
