import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.models import MLP_MNIST
from src.utils import get_device

def create_test_digit(digit, size=(28, 28)):
    """テスト用の数字画像を作成"""
    # 白い背景の画像を作成
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # フォントサイズを調整（数字が見えるように）
    font_size = min(size) // 2
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 数字を中央に描画
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    draw.text((x, y), text, fill='black', font=font)
    
    return img

def test_inference():
    """推論をテスト"""
    device = get_device()
    
    # モデルを読み込み
    checkpoint = torch.load("ckpts/mnist-mlp_best.pth", map_location='cpu')
    model = MLP_MNIST(num_classes=10)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # テスト用の数字画像を作成（0-9）
    for digit in range(10):
        # 画像を作成
        img = create_test_digit(digit)
        
        # 前処理
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 推論
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print(f"数字 {digit} → 予測: {predicted_class} (信頼度: {confidence:.4f})")
        
        # 画像を保存（確認用）
        img.save(f"test_digit_{digit}.png")

if __name__ == "__main__":
    test_inference() 