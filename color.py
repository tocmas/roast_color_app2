from torchvision import transforms
from torchvision.models import mobilenet_v2
import torch
import cv2
import numpy as np


# 前処理
# カスタム前処理関数
class CustomPreprocessing:
    def __init__(self, gamma=0.9, blur_size=(5, 5)):
        self.gamma = gamma
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

        # ガンマ補正
        lookup_table = np.zeros((256, 1), dtype='uint8')
        for i in range(256):
            lookup_table[i][0] = 255 * pow(float(i) / 255, 1.0 / self.gamma)
        img_gamma_corrected = cv2.LUT(np.uint8(img_max_adjusted), lookup_table)

        # ガウシアンブラーを追加
        img_blurred = cv2.GaussianBlur(img_gamma_corrected, self.blur_size, 10)

        return img_blurred

preprocessing = transforms.Compose([
    CustomPreprocessing(),
    transforms.ToPILImage(),
    transforms.Resize((120, 90)),
    transforms.CenterCrop(90), 
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
