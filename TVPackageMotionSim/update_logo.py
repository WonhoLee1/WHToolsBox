from PIL import Image, ImageDraw, ImageFont
import os

# Load the original image
img_path = r'C:\Users\GOODMAN\WHToolsBox\logo-R1.png'
save_path = r'c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\sidebar_logo.png'

if os.path.exists(img_path):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # 1. 기존 텍스트 영역 (하단 약 10%) 지우기
    # 배경색 (이미지의 최하단 구석 색상 추출)
    bg_color = img.getpixel((10, h-10))
    # 텍스트가 있는 아래쪽 영역 (h*0.80 ~ h) - 좀 더 넉넉하게
    draw.rectangle([0, int(h*0.88), w, h], fill=bg_color)
    
    # 2. 새로운 텍스트 쓰기
    new_text = "ADVANCED TV PACKAGE MOTION PHYSICS SIMULATION"
    
    # 폰트 로드 (Windows Malgun Gothic / Arial / Calibri 시도)
    try:
        font = ImageFont.truetype("arial.ttf", 20) # 20pt로 축소
    except:
        font = ImageFont.load_default()
        
    # 텍스트 레이아웃 (정확한 중앙 배치)
    bbox = draw.textbbox((0, 0), new_text, font=font)
    t_width = bbox[2] - bbox[0]
    t_height = bbox[3] - bbox[1]
    
    # 수평 중앙, 수직 하단영역 중앙 (0.94h 근처)
    tx = (w - t_width) // 2
    ty = int(h * 0.90) # 0.9 위치로 위아래 여백을 보며 수동 조절 가능
    
    # 원본과 유사한 Cyan 계열 (R: 120, G: 210, B: 255)
    text_color = (120, 210, 255) 
    
    draw.text((tx, ty), new_text, fill=text_color, font=font)
    
    # 저장
    img.save(save_path)
    print(f"Logo Refined and saved to {save_path}")
else:
    print("Source image NOT found.")
