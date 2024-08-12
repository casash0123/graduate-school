import cv2
import numpy as np
import os, sys

import tkinter as tk
from tkinter import *  # __all__
from tkinter import filedialog
from PIL import ImageTk, Image

image_load_State = False
image_full_path = []
select_image = ""
current_image = ""
current_image_pre = ""
select_image_name = ""

# 전처리 관련 변수 초기값 설정
Blurring_k_size = 0
Sharpening_str = 0
binary_th = 150
Adaptive_G_Block_size = 3
Morph_Kernel = 0
dilation_iter = 0
erosion_iter = 0
Morph_Type = "CLOSE"
Morph_TOPHAT = False
Morph_BLACKHAT = False
Binary_INV_Type = cv2.THRESH_BINARY
Adaptive_G_Type = None
Equalize = False
Otsu = False
binary_th_R = 128
binary_th_G = 128
binary_th_B = 128
binary_th_R_Type = False
binary_th_G_Type = False
binary_th_B_Type = False

BRIEF_Flag = False
BRIEF_result = ""
SURF_Flag = False
SURF_result = ""
ORB_Flag = False
ORB_result = ""
feature_detector = None

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.scale_factor = 1.0  # 초기 확대/축소 배율
        self.prev_x = None
        self.prev_y = None

    # sharpening 커널생성
    def sharpening(self, image, strength):
        b = (1 - strength) / 8
        sharpening_kernel = np.array([[b, b, b],
                                      [b, strength, b],
                                      [b, b, b]])
        output = cv2.filter2D(image, -1, sharpening_kernel)
        return output

    # 이미지 전처리
    def preprocessing_img(self, src_img):
        global current_image_pre

        # 그레이스케일 변환
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        if Equalize : # Equalize, 명암 평활화
            gray = cv2.equalizeHist(gray)

        if Blurring_k_size != 0: # 가우시안 블러 적용 : 잡음 제거
            Gaussian_Kernel = (Blurring_k_size * 2) - 1
            gray = cv2.GaussianBlur(gray, (Gaussian_Kernel, Gaussian_Kernel), 0)

        if Sharpening_str != 0: # Sharpening 적용 : 엣지 강조
            k_val = (1 - Sharpening_str) / 8
            sharpening_kernel = np.array([[k_val, k_val, k_val],
                                        [k_val, Sharpening_str, k_val],
                                        [k_val, k_val, k_val]])
            gray = cv2.filter2D(gray, -1, sharpening_kernel)

        if Otsu: # Otsu 이진화(이진화 임계값 자동계산)
            _, thresh = cv2.threshold(gray, binary_th, 255, Binary_INV_Type | cv2.THRESH_OTSU)
        elif Adaptive_G_Type: # Adaptive Gaussian 이진화
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, Adaptive_G_Block_size, 1)
        elif binary_th_R_Type: # R 채널 이진화
            r, g, b = cv2.split(src_img) # RGB 채널로 분리
            _, thresh = cv2.threshold(r, binary_th_R, 255, Binary_INV_Type)
        elif binary_th_G_Type: # G 채널 이진화
            r, g, b = cv2.split(src_img) # RGB 채널로 분리
            _, thresh = cv2.threshold(g, binary_th_G, 255, Binary_INV_Type)
        elif binary_th_B_Type: # B 채널 이진화
            r, g, b = cv2.split(src_img) # RGB 채널로 분리
            _, thresh = cv2.threshold(b, binary_th_B, 255, Binary_INV_Type)
        else : # 일반 이진화
            _, thresh = cv2.threshold(gray, binary_th, 255, Binary_INV_Type)

        kernel = np.ones((Morph_Kernel, Morph_Kernel), np.uint8) # 모폴로지 커널 생성
        if Morph_Type == "CLOSE": # 모폴로지 닫힘 연산 : 팽창 -> 침식
            dilation = cv2.dilate(thresh, kernel, iterations=dilation_iter)  # 팽창(Dilation)
            erosion = cv2.erode(dilation, kernel, iterations=erosion_iter)   # 침식(Erosion)
            result_img = erosion
        elif Morph_Type == "OPEN": # 모폴로지 열림 연산 : 침식 -> 팽창
            erosion = cv2.erode(thresh, kernel, iterations=erosion_iter)     # 침식(Erosion)
            dilation = cv2.dilate(erosion, kernel, iterations=dilation_iter) # 팽창(Dilation)
            result_img = dilation

        if Morph_TOPHAT and not Morph_BLACKHAT:  # 모폴로지 탑햇 연산
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            result_img = tophat
        elif Morph_BLACKHAT and not Morph_TOPHAT: # 모폴로지 블랙햇 연산
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            result_img = blackhat

        current_image_pre = result_img
        return result_img

    '''
    # SURF, BRIEF, ORB 특징점 검출 및 설명
    def detect_and_describe_features(self, img):
        if feature_detector is None:
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = feature_detector.detectAndCompute(gray, None)

        # 특징점을 이미지에 그리기
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
        return img_with_keypoints
    '''
    # 화면에 이미지 출력
    def display_image(self):
        if image_load_State != True:
            return

        img = current_image
        if (BRIEF_Flag == False) and (SURF_Flag == False) and (ORB_Flag == False):
            pre_img = self.preprocessing_img(current_image) # 이미지 전처리
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        elif (SURF_Flag == True) and (BRIEF_Flag == False) and (ORB_Flag == False) :
            pre_img = SURF_result
        elif (SURF_Flag == False) and (BRIEF_Flag == True) and (ORB_Flag == False) :
            pre_img = BRIEF_result
        elif (SURF_Flag == False) and (BRIEF_Flag == False) and (ORB_Flag == True) :
            pre_img = ORB_result

        # 이미지타입 변환 : opencv 이미지 타입 -> PIL 이미지 타입
        img = Image.fromarray(img)
        pre_img = Image.fromarray(pre_img)

        # 이미지 크기 가져오기
        img_width, img_height = img.size

        # OriginImageFrame 크기 가져오기
        frame_width = OriginImageFrame.winfo_width()
        frame_height = OriginImageFrame.winfo_height()

        # 이미지가 프레임보다 큰 경우에만 이미지를 축소하여 표시
        if img_width > frame_width or img_height > frame_height:
            ratio = min(frame_width / img_width, frame_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            # 이미지 확대/축소 : 마우스 휠 이벤트
            img = img.resize((int(new_width * self.scale_factor), int(new_height * self.scale_factor)), Image.LANCZOS)
            pre_img = pre_img.resize((int(new_width * self.scale_factor), int(new_height * self.scale_factor)), Image.LANCZOS)

        # 이미지를 프레임 중앙에 표시
        x_offset = (frame_width - img.width) // 2
        y_offset = (frame_height - img.height) // 2
        background_ori = Image.new('RGBA', (frame_width, frame_height), (255, 255, 255, 0))
        background_pre = Image.new('RGBA', (frame_width, frame_height), (255, 255, 255, 0))
        background_ori.paste(img, (x_offset, y_offset))
        background_pre.paste(pre_img, (x_offset, y_offset))
        resize_ori_image = ImageTk.PhotoImage(image=background_ori)
        resize_pre_image = ImageTk.PhotoImage(image=background_pre)

        # 이미지 표시
        self.Ori_image_on_canvas = OriginImageCanvas.create_image(0, 0, anchor=tk.NW, image=resize_ori_image)
        OriginImageCanvas.config(scrollregion=OriginImageCanvas.bbox(tk.ALL))
        OriginImageCanvas.image = resize_ori_image

        self.Pre_image_on_canvas = PreprocessingImageCanvas.create_image(0, 0, anchor=tk.NW, image=resize_pre_image)
        PreprocessingImageCanvas.config(scrollregion=PreprocessingImageCanvas.bbox(tk.ALL))
        PreprocessingImageCanvas.image = resize_pre_image

# 이미지 폴더 선택
def select_image_folder():
    global image_load_State
    # 이미지 폴더 선택 다이얼로그 열기
    folder_path = filedialog.askdirectory()
    if folder_path:
        # 선택한 이미지 폴더 내의 이미지 파일 표시
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif'))]
        if image_files:
            # 이미지 파일들을 리스트박스에 추가
            for image_file in image_files:
                global image_full_path
                image_folder, filename = os.path.split(image_file)
                image_listbox.insert(tk.END, filename)
                image_full_path.append(image_file)
        else:
            print("선택한 폴더에 이미지 파일이 없습니다.")
    image_load_State = True

# 리스트 박스에서 이미지 선택
def display_selected_image(event):
    global current_image, select_image_name
    # 선택된 이미지 파일 경로 가져오기
    selected_index = image_listbox.curselection()
    if selected_index:
        select_image_name = image_listbox.get(selected_index)
        selected_image_path = image_full_path[selected_index[0]]
        selected_image_path = selected_image_path.replace("\\", "/")
        image_path_array = np.fromfile(selected_image_path, np.uint8)
        img_ori = cv2.imdecode(image_path_array, cv2.IMREAD_COLOR)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        img = img_ori.copy()

        current_image = img
        image_viewer.display_image()

# Equalize 버튼 클릭시 호출 : Equalize, 명암 평활화
def button_click_Equalize():
    global Equalize
    if Equalize == False:
        Equalize = True
        button_Equalize.config(text="Equalize ON", bg="gray")  # 눌려진 상태
    else:
        Equalize = False
        button_Equalize.config(text="Equalize OFF", bg="SystemButtonFace")  # 안눌려진 상태
    image_viewer.display_image()

# Otsu 버튼 클릭시 호출 : Otsu 이진화(이진화 임계값 자동계산)
def button_click_Otsu():
    global Otsu
    if Otsu == False:
        Otsu = True
        button_Otsu.config(text="Otsu ON", bg="gray")  # 눌려진 상태로 만듬
    else:
        Otsu = False
        button_Otsu.config(text="Otsu OFF", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

# Binary_INV 버튼 클릭시 호출 : 이진영상 반전 (흰->검, 검->흰)
def button_Click_Binary_INV():
    global Binary_INV_Type
    if Binary_INV_Type == cv2.THRESH_BINARY:
        Binary_INV_Type = cv2.THRESH_BINARY_INV
        button_binary_Inv.config(text="Binary Inv ON", bg="gray")  # 눌려진 상태로 만듬
    else:
        Binary_INV_Type = cv2.THRESH_BINARY
        button_binary_Inv.config(text="Binary Inv OFF", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

# Adaptive_G 버튼 클릭시 호출 : Adaptive Gaussian
def button_Click_Adaptive_G():
    global Adaptive_G_Type
    if Adaptive_G_Type == None:
        Adaptive_G_Type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        button_Adaptive_G.config(text="Adaptive G ON", bg="gray")  # 눌려진 상태로 만듬
    else:
        Adaptive_G_Type = None
        button_Adaptive_G.config(text="Adaptive G OFF", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

# R 버튼 클릭시 호출 : R 채널 이진화
def button_Click_R_binary_th():
    global binary_th_R_Type
    if binary_th_R_Type == False:
        binary_th_R_Type = True
        button_R_binary_th.config(text="R", bg="gray")  # 눌려진 상태로 만듬
    else:
        binary_th_R_Type = False
        button_R_binary_th.config(text="R", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

# G 버튼 클릭시 호출 : G 채널 이진화
def button_Click_G_binary_th():
    global binary_th_G_Type
    if binary_th_G_Type == False:
        binary_th_G_Type = True
        button_G_binary_th.config(text="G", bg="gray")  # 눌려진 상태로 만듬
    else:
        binary_th_G_Type = False
        button_G_binary_th.config(text="G", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

def button_Click_B_binary_th():
    global binary_th_B_Type
    if binary_th_B_Type == False:
        binary_th_B_Type = True
        button_B_binary_th.config(text="B", bg="gray")  # 눌려진 상태로 만듬
    else:
        binary_th_B_Type = False
        button_B_binary_th.config(text="B", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

def button_click_Morph_CLOSE():
    global Morph_Type
    if Morph_Type == "CLOSE":
        Morph_Type = "OPEN"
        button_Mor_CLOSE.config(text="Morphology OPEN", bg="gray")  # 눌려진 상태로 만듬
    else:
        Morph_Type = "CLOSE"
        button_Mor_CLOSE.config(text="Morphology CLOSE", bg="SystemButtonFace")  # 안눌려진 상태
    image_viewer.display_image()

def button_click_Morph_TOPHAT():
    global Morph_TOPHAT
    if Morph_TOPHAT == False:
        Morph_TOPHAT = True
        button_Mor_TOPHAT.config(text="TOPHAT ON", bg="gray")  # 눌려진 상태로 만듬
    else:
        Morph_TOPHAT = False
        button_Mor_TOPHAT.config(text="TOPHAT OFF", bg="SystemButtonFace")  # 안눌려진 상태
    image_viewer.display_image()

def button_click_Morph_BLACKHAT():
    global Morph_BLACKHAT
    if Morph_BLACKHAT == False:
        Morph_BLACKHAT = True
        button_Mor_BLACKHAT.config(text="BLACKHAT ON", bg="gray")  # 눌려진 상태로 만듬
    else:
        Morph_BLACKHAT = False
        button_Mor_BLACKHAT.config(text="BLACKHAT OFF", bg="SystemButtonFace")  # 안눌려진 상태
    image_viewer.display_image()

def bar_changed_Blurring(value):
    global Blurring_k_size
    Blurring_k_size = int(value)
    image_viewer.display_image()

def bar_changed_Sharpening(value):
    global Sharpening_str
    Sharpening_str = int(value)
    image_viewer.display_image()

def bar_changed_bar_binary_th(value):
    global binary_th
    binary_th = int(value)
    image_viewer.display_image()

def bar_changed_bar_binary_th_R(value):
    global binary_th_R
    binary_th_R = int(value)
    image_viewer.display_image()

def bar_changed_bar_binary_th_G(value):
    global binary_th_G
    binary_th_G = int(value)
    image_viewer.display_image()

def bar_changed_bar_binary_th_B(value):
    global binary_th_B
    binary_th_B = int(value)
    image_viewer.display_image()

def bar_changed_bar_Adaptive_G(value):
    global Adaptive_G_Block_size
    if int(value) < 3:
        Adaptive_G_Block_size = 3
    if int(value) % 2 == 0:
        Adaptive_G_Block_size = int(value) + 1
    image_viewer.display_image()

def bar_changed_Moph_Kernel(value):
    global Morph_Kernel
    # 모폴로지 연산을 위한 구조 요소 생성
    Morph_Kernel = (int(value)) * 2 + 1
    image_viewer.display_image()

def bar_changed_dilation_iter(value):
    global dilation_iter
    dilation_iter = int(value)
    image_viewer.display_image()

def bar_changed_erosion_iter(value):
    global erosion_iter
    erosion_iter = int(value)
    image_viewer.display_image()

def button_click_SURF():
    global BRIEF_Flag, SURF_Flag, ORB_Flag, SURF_result
    try:
        if SURF_Flag == False:
            SURF_Flag = True
            BRIEF_Flag = False
            ORB_Flag = False

            img = current_image
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print("SURF 연산 Error check2")
            surf = cv2.xfeatures2d_SURF.create(10000)
            print("SURF 연산 Error check3")
            kp, des = surf.detectAndCompute(img, None)
            print("SURF 연산 Error check3")

            SURF_result = cv2.drawKeypoints(img, kp, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            print("SURF 연산 Error check4")

            button_SURF.config(bg="gray")
            button_BRIEF.config(bg="SystemButtonFace")
            button_ORB.config(bg="SystemButtonFace")
        else:
            SURF_Flag = False
            button_SURF.config(bg="SystemButtonFace")
    except:
        print("SURF 연산 Error")

    image_viewer.display_image()

def button_click_BRIEF():
    global BRIEF_Flag, SURF_Flag, ORB_Flag, BRIEF_result
    try:
        if BRIEF_Flag == False:
            BRIEF_Flag = True
            SURF_Flag = False
            ORB_Flag = False
            img = current_image
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img2 = None

            # STAR 탐지기 먼저 개시
            star = cv2.xfeatures2d.StarDetector_create()
            # BRIEF 추출기 개시
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            # STAR로 키포인트를 검출하고 BRIEF로 디스크립터 계산
            kp = star.detect(img, None)
            kp, des = brief.compute(img, kp)
            BRIEF_result = cv2.drawKeypoints(img, kp, img2, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            button_SURF.config(bg="SystemButtonFace")
            button_BRIEF.config(bg="gray")
            button_ORB.config(bg="SystemButtonFace")
        else :
            BRIEF_Flag = False
            button_BRIEF.config(bg="SystemButtonFace")
    except:
        print("BRIEF 연산 Error")

    image_viewer.display_image()

def button_click_ORB():
    global BRIEF_Flag, SURF_Flag, ORB_Flag, ORB_result
    try:
        if ORB_Flag == False:
            BRIEF_Flag = False
            SURF_Flag = False
            ORB_Flag = True

            img = current_image
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img2 = None

            orb = cv2.ORB_create()
            orb.setMaxFeatures(200)
            kp = orb.detect(img, None)
            kp, des = orb.compute(img, kp)

            ORB_result = cv2.drawKeypoints(img, kp, img2, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            button_SURF.config(bg="SystemButtonFace")
            button_BRIEF.config(bg="SystemButtonFace")
            button_ORB.config(bg="gray")
        else:
            ORB_Flag = False
            button_ORB.config(bg="SystemButtonFace")
    except:
        print("ORB 연산 Error")

    image_viewer.display_image()


# UI 생성 및 title 설정
root = tk.Tk()
root.geometry("1920x705")
root.title("Image Preprocessing Test Program")

# 이미지 폴더 선택 버튼 생성
select_button = tk.Button(root, text="Select Image Folder", command=select_image_folder)
select_button.place(x=10, y=10)

# 이미지 파일 경로를 표시할 리스트박스 생성
image_listbox = tk.Listbox(root)
image_listbox.place(x=10, y=40, height=260, width=480)

# Preprocessing Parameter 프레임
PreprocessingParamFrame = LabelFrame(root, text="Preprocessing Param")
PreprocessingParamFrame.place(x=10, y=310, height=385, width=480)
PreprocessingParam = Label(PreprocessingParamFrame)

# Original Image 프레임 : in root
OriginImageFrame = LabelFrame(root, text="Original Image")
OriginImageFrame.place(x=500, y=10, height=685, width=700)
OriginImageCanvas = tk.Canvas(OriginImageFrame, bg="#F0F0F0")
OriginImageCanvas.pack(fill="both", expand=True)

# Preprocessing Image 프레임 : in root
PreprocessingImageFrame = LabelFrame(root, text="Preprocessing Image")
PreprocessingImageFrame.place(x=1210, y=10, height=685, width=700)
PreprocessingImageCanvas = tk.Canvas(PreprocessingImageFrame, bg="#F0F0F0")
PreprocessingImageCanvas.pack(fill="both", expand=True)

# Blurring 드래그 바 : in Preprocessing Parameter 프레임
label_Blurring = Label(PreprocessingParamFrame, text="Blurring : ", anchor='w', font=("Consolas", 10))
label_Blurring.place(x=10, y=0, height=40, width=120)
bar_Blurring = tk.Scale(PreprocessingParamFrame, from_= 0, to_=20, orient=tk.HORIZONTAL, command=bar_changed_Blurring)
bar_Blurring.set(Blurring_k_size)
bar_Blurring.place(x=130, y=0, height=40, width=100)

# Sharpening 드래그 바 : in Preprocessing Parameter 프레임
label_Sharpening = Label(PreprocessingParamFrame, text="Sharpening : ", anchor='w', font=("Consolas", 10))
label_Sharpening.place(x=240, y=0, height=40, width=120)
bar_Sharpening = tk.Scale(PreprocessingParamFrame, from_= 0, to_=20, orient=tk.HORIZONTAL, command=bar_changed_Sharpening)
bar_Sharpening.set(Sharpening_str)
bar_Sharpening.place(x=370, y=0, height=40, width=100)

# Equalize 버튼 : in Preprocessing Parameter 프레임
button_Equalize = tk.Button(PreprocessingParamFrame, text="Equalize OFF", width=100, anchor='center', font=("Consolas", 10), command=button_click_Equalize)
button_Equalize.place(x=10, y=50, height=25, width=110)
# Otsu 버튼 : in Preprocessing Parameter 프레임
button_Otsu = tk.Button(PreprocessingParamFrame, text="Otsu OFF", width=100, anchor='center', font=("Consolas", 10), command=button_click_Otsu)
button_Otsu.place(x=125, y=50, height=25, width=110)
# binary Inverse버튼 : in Preprocessing Parameter 프레임
button_binary_Inv = tk.Button(PreprocessingParamFrame, text="Binary Inv OFF", width=100, anchor='center', font=("Consolas", 10), command=button_Click_Binary_INV)
button_binary_Inv.place(x=240, y=50, height=25, width=110)
# Adaptive Gaussian 버튼 : in Preprocessing Parameter 프레임
button_Adaptive_G = tk.Button(PreprocessingParamFrame, text="Adaptive G OFF", width=100, anchor='center', font=("Consolas", 10), command=button_Click_Adaptive_G)
button_Adaptive_G.place(x=355, y=50, height=25, width=110)

# binary_th 드래그 바 : in Preprocessing Parameter 프레임
label_binary_th = Label(PreprocessingParamFrame, text=" Binary threshold : ", anchor='w', font=("Consolas", 10))
label_binary_th.place(x=5, y=80, height=40, width=150)
bar_binary_th = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 255, orient=tk.HORIZONTAL, command=bar_changed_bar_binary_th)
bar_binary_th.set(binary_th)
bar_binary_th.place(x=200, y=80, height=40, width=270)

# R_binary_th 버튼 : in Preprocessing Parameter 프레임
button_R_binary_th = tk.Button(PreprocessingParamFrame, text="R", anchor='center', font=("Consolas", 10), command=button_Click_R_binary_th)
button_R_binary_th.place(x=10, y=125, height=30, width=30)
# R_binary_th 드래그 바 : in Preprocessing Parameter 프레임
bar_R_binary_th = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 255, orient=tk.HORIZONTAL, command=bar_changed_bar_binary_th_R)
bar_R_binary_th.set(binary_th_R)
bar_R_binary_th.place(x=50, y=120, height=40, width=105)

# G_binary_th 버튼 : in Preprocessing Parameter 프레임
button_G_binary_th = tk.Button(PreprocessingParamFrame, text="G", anchor='center', font=("Consolas", 10), command=button_Click_G_binary_th)
button_G_binary_th.place(x=170, y=125, height=30, width=30)
# G_binary_th 드래그 바 : in Preprocessing Parameter 프레임
bar_G_binary_th = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 255, orient=tk.HORIZONTAL, command=bar_changed_bar_binary_th_G)
bar_G_binary_th.set(binary_th_R)
bar_G_binary_th.place(x=210, y=120, height=40, width=105)

# B_binary_th 버튼 : in Preprocessing Parameter 프레임
button_B_binary_th = tk.Button(PreprocessingParamFrame, text="B", anchor='center', font=("Consolas", 10), command=button_Click_B_binary_th)
button_B_binary_th.place(x=325, y=125, height=30, width=30)
# B_binary_th 드래그 바 : in Preprocessing Parameter 프레임
bar_B_binary_th = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 255, orient=tk.HORIZONTAL, command=bar_changed_bar_binary_th_B)
bar_B_binary_th.set(binary_th_R)
bar_B_binary_th.place(x=365, y=120, height=40, width=105)

# Adaptive Gaussian 드래그 바 : in Preprocessing Parameter 프레임
label_Adaptive_G = Label(PreprocessingParamFrame, text=" Adaptive Gaussian Block : ", anchor='w', font=("Consolas", 10))
label_Adaptive_G.place(x=5, y=160, height=40, width=190)
bar_Adaptive_G = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 200, orient=tk.HORIZONTAL, command=bar_changed_bar_Adaptive_G)
bar_Adaptive_G.set(Adaptive_G_Block_size)
bar_Adaptive_G.place(x=200, y=160, height=40, width=270)

# Mor_CLOSE 버튼 : in Preprocessing Parameter 프레임
button_Mor_CLOSE = tk.Button(PreprocessingParamFrame, text="Morphology CLOSE", anchor='center', font=("Consolas", 10), command=button_click_Morph_CLOSE)
button_Mor_CLOSE.place(x=10, y=210, height=25, width=120)
# Mor_TOPHAT 버튼 : in Preprocessing Parameter 프레임
button_Mor_TOPHAT = tk.Button(PreprocessingParamFrame, text="TOPHAT OFF", anchor='center', font=("Consolas", 10), command=button_click_Morph_TOPHAT)
button_Mor_TOPHAT.place(x=140, y=210, height=25, width=120)
# Mor_BLACKHAT 버튼 : in Preprocessing Parameter 프레임
button_Mor_BLACKHAT = tk.Button(PreprocessingParamFrame, text="BLACKHAT OFF", anchor='center', font=("Consolas", 10), command=button_click_Morph_BLACKHAT)
button_Mor_BLACKHAT.place(x=270, y=210, height=25, width=120)

# Moph_Kernel 드래그 바 : in Preprocessing Parameter 프레임
label_Moph_Kernel = Label(PreprocessingParamFrame, text=" Morphology Kernel : ", anchor='w', font=("Consolas", 10))
label_Moph_Kernel.place(x=5, y=240, height=40, width=150)
bar_Morph_Kernel = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 30, orient=tk.HORIZONTAL, command=bar_changed_Moph_Kernel)
bar_Morph_Kernel.set(Morph_Kernel)
bar_Morph_Kernel.place(x=200, y=240, height=40, width=270)

# dilation_iter 드래그 바 : in Preprocessing Parameter 프레임
label_dilation_iter = Label(PreprocessingParamFrame, text="Dilation iter : ", anchor='w', font=("Consolas", 10))
label_dilation_iter.place(x=10, y=280, height=40, width=120)
bar_dilation_iter = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 5, orient=tk.HORIZONTAL, command=bar_changed_dilation_iter)
bar_dilation_iter.set(dilation_iter)
bar_dilation_iter.place(x=130, y=280, height=40, width=100)

# erosion_iter 드래그 바 : in Preprocessing Parameter 프레임
label_erosion_iter = Label(PreprocessingParamFrame, text=" Erosion iter : ", anchor='w', font=("Consolas", 10))
label_erosion_iter.place(x=240, y=280, height=40, width=120)
bar_erosion_iter = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 5, orient=tk.HORIZONTAL, command=bar_changed_erosion_iter)
bar_erosion_iter.set(erosion_iter)
bar_erosion_iter.place(x=370, y=280, height=40, width=100)

# SURF, BRIEF, ORB 버튼 추가
button_SURF = tk.Button(PreprocessingParamFrame, text="SURF", anchor='center', font=("Consolas", 10), command=button_click_SURF)
button_SURF.place(x=10, y=330, height=25, width=120)
button_BRIEF = tk.Button(PreprocessingParamFrame, text="BRIEF", anchor='center', font=("Consolas", 10), command=button_click_BRIEF)
button_BRIEF.place(x=140, y=330, height=25, width=120)
button_ORB = tk.Button(PreprocessingParamFrame, text="ORB", anchor='center', font=("Consolas", 10), command=button_click_ORB)
button_ORB.place(x=270, y=330, height=25, width=120)

# 이미지 뷰어 생성
image_viewer = ImageViewer(root)
# 이미지를 표시할 때 리스트박스에서 이미지 선택하도록 설정
image_listbox.bind("<Double-Button-1>", display_selected_image)

root.mainloop()
