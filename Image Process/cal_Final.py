import cv2
import numpy as np
import os, sys

import tkinter as tk
from tkinter import *  # __all__
from tkinter import filedialog
from PIL import ImageTk, Image
from scipy.signal import find_peaks

image_full_path = []
Save_BBoxTxt_path = "./BBoxTxt/"
Save_PreImgae_path = './BinaryResult/'
select_image = ""
current_image = ""
current_image_pre = ""
current_image_blob = ""
current_image_pre_blob = ""
select_image_name = ""
blobs = []
blobs_Bbox = []
ROI_devide_Per = 3
blob_size_MIN = 3
blob_size_MAX = 200
pixel_limit = 163840000
CalBlobFlag = False

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

CANNY_Flag = False
CANNY_result = ""
BRIEF_Flag = False
BRIEF_result = ""
ORB_Flag = False
ORB_result = ""
feature_detector = None

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.scale_factor = 1.0  # 초기 확대/축소 배율
        self.prev_x = None
        self.prev_y = None

    def sharpening(self, image, strength):
        b = (1 - strength) / 8
        sharpening_kernel = np.array([[b, b, b],
                                      [b, strength, b],
                                      [b, b, b]])
        output = cv2.filter2D(image, -1, sharpening_kernel)
        return output

    def preprocessing_img(self, src_img):
        #global binary_th, Morph_Kernel, dilation_iter, erosion_iter, Morph_Type, Equalize, Otsu, current_image_pre
        global current_image_pre

        # 그레이스케일 변환
        gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)

        if Equalize :
            gray = cv2.equalizeHist(gray)

        if Blurring_k_size != 0:
            # 가우시안 블러 적용 : 잡음 제거
            Gaussian_Kernel = (Blurring_k_size * 2) - 1
            gray = cv2.GaussianBlur(gray, (Gaussian_Kernel, Gaussian_Kernel), 0)

        if Sharpening_str != 0:
            k_val = (1 - Sharpening_str) / 8
            sharpening_kernel = np.array([[k_val, k_val, k_val],
                                        [k_val, Sharpening_str, k_val],
                                        [k_val, k_val, k_val]])
            gray = cv2.filter2D(gray, -1, sharpening_kernel)

        if Otsu:
            _, thresh = cv2.threshold(gray, binary_th, 255, Binary_INV_Type | cv2.THRESH_OTSU)
        elif Adaptive_G_Type:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, Binary_INV_Type, Adaptive_G_Block_size, 1)
        elif binary_th_R_Type:
            # RGB 채널로 분리
            r, g, b = cv2.split(src_img)
            _, thresh = cv2.threshold(r, binary_th_R, 255, Binary_INV_Type)
        elif binary_th_G_Type:
            # RGB 채널로 분리
            r, g, b = cv2.split(src_img)
            _, thresh = cv2.threshold(g, binary_th_G, 255, Binary_INV_Type)
        elif binary_th_B_Type:
            # RGB 채널로 분리
            r, g, b = cv2.split(src_img)
            _, thresh = cv2.threshold(b, binary_th_B, 255, Binary_INV_Type)
        else :
            _, thresh = cv2.threshold(gray, binary_th, 255, Binary_INV_Type)

        kernel = np.ones((Morph_Kernel, Morph_Kernel), np.uint8)
        if Morph_Type == "CLOSE":
            # 팽창(Dilation)
            dilation = cv2.dilate(thresh, kernel, iterations=dilation_iter)
            # 침식(Erosion)
            erosion = cv2.erode(dilation, kernel, iterations=erosion_iter)
            result_img = erosion
        elif Morph_Type == "OPEN":
            # 침식(Erosion)
            erosion = cv2.erode(thresh, kernel, iterations=erosion_iter)
            # 팽창(Dilation)
            dilation = cv2.dilate(erosion, kernel, iterations=dilation_iter)
            result_img = dilation

        if Morph_TOPHAT and Morph_BLACKHAT == False and erosion_iter != 0 and dilation_iter != 0:
            erosion = cv2.erode(thresh, kernel, iterations=erosion_iter)
            dilation = cv2.dilate(erosion, kernel, iterations=dilation_iter)
            tophat = cv2.subtract(dilation, gray)
            result_img = tophat
        elif Morph_BLACKHAT and Morph_TOPHAT == False and erosion_iter != 0 and dilation_iter != 0:
            dilation = cv2.dilate(thresh, kernel, iterations=dilation_iter)
            erosion = cv2.erode(dilation, kernel, iterations=erosion_iter)
            blackhat = cv2.subtract(erosion, gray)
            result_img = blackhat

        current_image_pre = result_img
        return result_img

    def display_image(self):
        global CalBlobFlag, blobs

        if type(current_image) != str:
            img = current_image
            if (BRIEF_Flag == False) and (CANNY_Flag == False) and (ORB_Flag == False):
                pre_img = self.preprocessing_img(current_image)  # 이미지 전처리
                pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
            elif (CANNY_Flag == True) and (BRIEF_Flag == False) and (ORB_Flag == False):
                pre_img = CANNY_result
            elif (BRIEF_Flag == True) and (CANNY_Flag == False) and (ORB_Flag == False):
                pre_img = BRIEF_result
            elif (ORB_Flag == True) and (BRIEF_Flag == False) and (CANNY_Flag == False):
                pre_img = ORB_result

            if CalBlobFlag:
                img = current_image_blob
                CalBlobFlag = False
                pre_img = current_image_pre_blob
                blobs = []

            # 이미지타입 변환 : opencv 이미지 타입 -> PIL 이미지 타입
            img = Image.fromarray(img)
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
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
                # img = img.resize((new_width, new_height), Image.LANCZOS)
                # 이미지 확대/축소 : 마우스 휠 이벤트
                img = img.resize((int(new_width * self.scale_factor), int(new_height * self.scale_factor)),
                                 Image.LANCZOS)
                pre_img = pre_img.resize((int(new_width * self.scale_factor), int(new_height * self.scale_factor)),
                                         Image.LANCZOS)

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

            self.Pre_image_on_canvas = PreprocessingImageCanvas.create_image(0, 0, anchor=tk.NW,
                                                                             image=resize_pre_image)
            PreprocessingImageCanvas.config(scrollregion=PreprocessingImageCanvas.bbox(tk.ALL))
            PreprocessingImageCanvas.image = resize_pre_image



def select_image_folder():
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

def display_selected_image(event):
    global current_image, select_image_name, blobs
    # 선택된 이미지 파일 경로 가져오기
    selected_index = image_listbox.curselection()
    if selected_index:
        select_image_name = image_listbox.get(selected_index)
        selected_image_path = image_full_path[selected_index[0]].replace("\\", "/")
        image_path_array = np.fromfile(selected_image_path, np.uint8)
        img_ori = cv2.imdecode(image_path_array, cv2.IMREAD_COLOR)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)
        #img = img_ori.copy()
        #if img.shape[0] * img.shape[1] >= pixel_limit :
        #    down_size = (img.shape[1]//10, img.shape[0]//10)
        #    img = cv2.resize(img, down_size, interpolation=cv2.INTER_LINEAR)

        current_image = img_ori
        blobs = []
        image_viewer.display_image()

def button_click_Equalize():
    global Equalize
    if Equalize == False:
        Equalize = True
        button_Equalize.config(text="Equalize ON", bg="gray")  # 눌려진 상태
    else:
        Equalize = False
        button_Equalize.config(text="Equalize OFF", bg="SystemButtonFace")  # 안눌려진 상태
    image_viewer.display_image()

def button_click_Otsu():
    global Otsu
    if Otsu == False:
        Otsu = True
        button_Otsu.config(text="Otsu ON", bg="gray")  # 눌려진 상태로 만듬
    else:
        Otsu = False
        button_Otsu.config(text="Otsu OFF", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

def button_Click_Binary_INV():
    global Binary_INV_Type
    if Binary_INV_Type == cv2.THRESH_BINARY:
        Binary_INV_Type = cv2.THRESH_BINARY_INV
        button_binary_Inv.config(text="Binary Inv ON", bg="gray")  # 눌려진 상태로 만듬
    else:
        Binary_INV_Type = cv2.THRESH_BINARY
        button_binary_Inv.config(text="Binary Inv OFF", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

def button_Click_Adaptive_G():
    global Adaptive_G_Type
    if Adaptive_G_Type == None:
        Adaptive_G_Type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        button_Adaptive_G.config(text="Adaptive G ON", bg="gray")  # 눌려진 상태로 만듬
    else:
        Adaptive_G_Type = None
        button_Adaptive_G.config(text="Adaptive G OFF", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

def button_Click_R_binary_th():
    global binary_th_R_Type
    if binary_th_R_Type == False:
        binary_th_R_Type = True
        button_R_binary_th.config(text="R", bg="gray")  # 눌려진 상태로 만듬
    else:
        binary_th_R_Type = False
        button_R_binary_th.config(text="R", bg="SystemButtonFace")  # 안눌려진 상태로 만듬
    image_viewer.display_image()

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

def bar_changed_MIN_Blob_size(value):
    global blob_size_MIN
    blob_size_MIN = int(value)
    image_viewer.display_image()

def bar_changed_MAX_Blob_size(value):
    global blob_size_MAX
    blob_size_MAX = int(value)
    image_viewer.display_image()

def bar_changed_ROI_devide_per(value):
    global ROI_devide_Per
    ROI_devide_Per = int(value)
    image_viewer.display_image()

def button_click_Set_ROI():
    global current_image, current_image_pre

    height, width = current_image.shape[:2]
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    # 이미지 평활화
    equalized = cv2.equalizeHist(gray)
    # 이진화 : Otsu 알고리즘 -> th 자동조절
    th, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 수평 및 수직 프로젝션 계산
    h_proj = np.sum(binary, axis=1)
    #v_proj = np.sum(binary, axis=0)

    # 수평 프로젝션의 미분 계산
    h_diff = np.gradient(h_proj)

    # 수평 프로젝션의 미분의 평균, 표준편차 계산
    h_diff_arg = np.average(h_diff)
    h_diff_std = np.std(h_diff)

    # 수평 프로젝션의 미분의 피크점 계산
    peaks, _ = find_peaks(h_diff, height=h_diff_arg + h_diff_std, distance=len(h_diff) // 3)

    h_edge_top = peaks[0]
    h_edge_down = peaks[1]

    # ROI에 추가시킬 범위 지정
    add_pixel_per = 0.01
    add_y = int((h_edge_down - h_edge_top) * add_pixel_per)

    ROI_x1 = 0
    ROI_y1 = h_edge_top - add_y
    ROI_x2 = width
    ROI_y2 = h_edge_down

    current_image = current_image[ROI_y1:ROI_y2, ROI_x1:ROI_x2]
    current_image_pre = current_image_pre[ROI_y1:ROI_y2, ROI_x1:ROI_x2]

    image_viewer.display_image()

def button_click_Cal_blob():
    global current_image, current_image_pre, blobs, blobs_Bbox

    src_image = current_image_pre
    ROI_w = src_image.shape[1]
    ROI_h = src_image.shape[0]

    contours, _ = cv2.findContours(src_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 전체 Blob들의 크기 평균 계산
    areas = [cv2.contourArea(c) for c in contours]
    #area_avg = np.mean(areas)
    #area_std = np.std(areas)

    for contour in contours:
        area = cv2.contourArea(contour)
        if blob_size_MIN <= area <= blob_size_MAX:

            # Blob의 중심 좌표 계산
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Blob의 경계 사각형
            x, y, w, h = cv2.boundingRect(contour)

            if ROI_h * (ROI_devide_Per * 0.01) <= cy <= ROI_h * (1 - (ROI_devide_Per * 0.01)):
                # Blob 정보를 리스트에 추가
                blobs.append({
                    'area': area,
                    'centroid': (cx, cy),
                    'bounding_box': (x, y, w, h)
                })
            blobs_Bbox = blobs[:]
    draw_blob()

def draw_blob():
    global blobs, current_image, CalBlobFlag, current_image_blob, current_image_pre_blob

    if len(blobs) == 0:
        print("blobs is Empty")
        return

    current_image_blob = current_image.copy()
    current_image_pre_blob = current_image_pre.copy()
    current_image_pre_blob = cv2.cvtColor(current_image_pre_blob, cv2.COLOR_GRAY2BGR)
    for i, blob in enumerate(blobs):
        CalBlobFlag = True
        #print(f"Blob {i + 1}:")
        #print("  Area:", blob['area'])
        #print("  Centroid:", blob['centroid'])
        #print("  Bounding Box:", blob['bounding_box'])
        #print()
        x = blob['bounding_box'][0]
        y = blob['bounding_box'][1]
        w = blob['bounding_box'][2]
        h = blob['bounding_box'][3]
        pos_str = str(x) + ", " + str(y)
        cv2.rectangle(current_image_blob, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(current_image_blob, pos_str, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(current_image_pre_blob, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(current_image_pre_blob, pos_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    image_viewer.display_image()

def button_click_Save_preImage():
    global current_image_pre, select_image_name
    image_name, image_type = select_image_name.split(".")
    if not os.path.exists(Save_PreImgae_path):
        os.makedirs(Save_PreImgae_path)
    #save_img_ori = save_dir + image_name + '.' + image_type
    save_img_pre = Save_PreImgae_path + image_name + '_Binary.' + image_type
    save_img_pre_PIL = Save_PreImgae_path + image_name + '_Binary_PIL.' + image_type
    #save_img_pre = save_dir + image_name + '_Binary.' + 'jpg'

    #cv2.imwrite(save_img_ori, cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
    #print("Save Original Image :", save_img_ori)

    current_image_pre = cv2.cvtColor(current_image_pre, cv2.COLOR_GRAY2BGR)  # 그레이 스케일에서 BGR로 변환
    cv2.imwrite(save_img_pre, current_image_pre)

    current_image_pre_PIL = Image.fromarray(current_image_pre)
    current_image_pre_PIL.save(save_img_pre_PIL, format='PNG')  # 저장할 파일명과 포맷을 지정합니다

    print("Save Preprocessing Image :", save_img_pre)

def button_click_Save_BBox():
    global blobs_Bbox

    if not os.path.exists(Save_BBoxTxt_path):
        os.makedirs(Save_BBoxTxt_path)

    if len(blobs_Bbox) != 0 :
        image_name, image_type = select_image_name.split(".")
        file_path = Save_BBoxTxt_path + image_name + ".txt"
        try:
            with open(file_path, 'w') as file:
                for BBox in blobs_Bbox:
                    x = BBox['bounding_box'][0]
                    y = BBox['bounding_box'][1]
                    w = BBox['bounding_box'][2]
                    h = BBox['bounding_box'][3]

                    BBox_Info = '\t' + str(x) + '\t' + str(y) + '\t' + str(w) + '\t' + str(h) + '\n'
                    # 파일에 쓸 내용 작성
                    file.write(BBox_Info)
            print("complete save bbox")
        except:
            print("save bbox error")
    blobs_Bbox = []

def button_click_Canny():
    global BRIEF_Flag, CANNY_Flag, ORB_Flag, CANNY_result
    try:
        if CANNY_Flag == False:
            CANNY_Flag = True
            BRIEF_Flag = False
            ORB_Flag = False

            img = current_image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            CANNY_result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            button_Canny.config(bg="gray")
            button_BRIEF.config(bg="SystemButtonFace")
            button_ORB.config(bg="SystemButtonFace")
        else:
            CANNY_Flag = False
            button_Canny.config(bg="SystemButtonFace")
    except Exception as e:
        print("Canny Edge Detection Error:", e)

    image_viewer.display_image()

def button_click_BRIEF():
    global BRIEF_Flag, CANNY_Flag, ORB_Flag, BRIEF_result
    try:
        if BRIEF_Flag == False:
            CANNY_Flag = False
            BRIEF_Flag = True
            ORB_Flag = False
            img = current_image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img2 = None

            # FAST 키포인트 검출기를 생성합니다
            fast = cv2.FastFeatureDetector_create()
            # 키포인트를 검출합니다
            kp = fast.detect(img, None)
            # BRIEF 추출기 개시
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            # 키포인트와 함께 BRIEF 디스크립터를 계산합니다
            kp, des = brief.compute(img, kp)

            BRIEF_result = cv2.drawKeypoints(img, kp, img2, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            button_Canny.config(bg="SystemButtonFace")
            button_BRIEF.config(bg="gray")
            button_ORB.config(bg="SystemButtonFace")
        else :
            BRIEF_Flag = False
            button_BRIEF.config(bg="SystemButtonFace")
    except:
        print("BRIEF 연산 Error")

    image_viewer.display_image()

def button_click_ORB():
    global BRIEF_Flag, CANNY_Flag, ORB_Flag, ORB_result
    try:
        if ORB_Flag == False:
            BRIEF_Flag = False
            CANNY_Flag = False
            ORB_Flag = True

            img = current_image
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img2 = None

            orb = cv2.ORB_create()
            orb.setMaxFeatures(200)
            kp = orb.detect(img, None)
            kp, des = orb.compute(img, kp)

            ORB_result = cv2.drawKeypoints(img, kp, img2, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            button_Canny.config(bg="SystemButtonFace")
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
#root.geometry("2560x1080")
#root.geometry("2500x1000+2560+0")
root.geometry("1920x860")
root.title("MNV FPCB Image Test Program")

# 이미지 폴더 선택 버튼 생성
select_button = tk.Button(root, text="Select Image Folder", command=select_image_folder)
select_button.place(x=10, y=10)

# 이미지 파일 경로를 표시할 리스트박스 생성
image_listbox = tk.Listbox(root)
image_listbox.place(x=10, y=40, height=260, width=480)

# Preprocessing Parameter 프레임
PreprocessingParamFrame = LabelFrame(root, text="Preprocessing Param")
PreprocessingParamFrame.place(x=10, y=310, height=540, width=480)
PreprocessingParam = Label(PreprocessingParamFrame)

# Original Image 프레임 : in root
OriginImageFrame = LabelFrame(root, text="Original Image")
OriginImageFrame.place(x=500, y=10, height=840, width=700)
OriginImageCanvas = tk.Canvas(OriginImageFrame, bg="#F0F0F0")
OriginImageCanvas.pack(fill="both", expand=True)

# Preprocessing Image 프레임 : in root
PreprocessingImageFrame = LabelFrame(root, text="Preprocessing Image")
PreprocessingImageFrame.place(x=1210, y=10, height=840, width=700)
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
bar_Sharpening = tk.Scale(PreprocessingParamFrame, from_= 0, to_=40, orient=tk.HORIZONTAL, command=bar_changed_Sharpening)
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
bar_Adaptive_G = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 1000, orient=tk.HORIZONTAL, command=bar_changed_bar_Adaptive_G)
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
bar_Morph_Kernel = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 50, orient=tk.HORIZONTAL, command=bar_changed_Moph_Kernel)
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
bar_dilation_iter.set(erosion_iter)
bar_erosion_iter.place(x=370, y=280, height=40, width=100)

# Set_ROI 버튼 : in Preprocessing Parameter 프레임
button_Set_ROI = tk.Button(PreprocessingParamFrame, text="Set ROI", width=120, anchor='center', font=("Consolas", 10), command=button_click_Set_ROI)
button_Set_ROI.place(x=10, y=330, height=25, width=120)

# Cal_BLob 버튼 : in Preprocessing Parameter 프레임
button_Cal_BLob = tk.Button(PreprocessingParamFrame, text="Calculate Blob", width=120, anchor='center', font=("Consolas", 10), command=button_click_Cal_blob)
button_Cal_BLob.place(x=135, y=330, height=25, width=120)

# Save Pre-Image 버튼 : in Preprocessing Parameter 프레임
button_Save_PreImage = tk.Button(PreprocessingParamFrame, text="Save Pre-Image", width=120, anchor='center', font=("Consolas", 10), command=button_click_Save_preImage)
button_Save_PreImage.place(x=260, y=330, height=25, width=120)

# Save BBox 버튼 : in Preprocessing Parameter 프레임
button_Save_BBox = tk.Button(PreprocessingParamFrame, text="Save BBox", width=80, anchor='center', font=("Consolas", 10), command=button_click_Save_BBox)
button_Save_BBox.place(x=385, y=330, height=25, width=80)

# MIN_Blob_size 드래그 바 : in Preprocessing Parameter 프레임
label_MIN_Blob_size = Label(PreprocessingParamFrame, text=" MIN Blob size : ", width=30, anchor='w', font=("Consolas", 10))
label_MIN_Blob_size.place(x=5, y=360, height=40, width=150)
bar_MIN_Blob_size = tk.Scale(PreprocessingParamFrame, from_= 0, to_= 500, orient=tk.HORIZONTAL, command=bar_changed_MIN_Blob_size)
bar_MIN_Blob_size.set(blob_size_MIN)
bar_MIN_Blob_size.place(x=200, y=360, height=40, width=270)

# MAX_Blob_size 드래그 바 : in Preprocessing Parameter 프레임
label_MAX_Blob_size = Label(PreprocessingParamFrame, text=" MAX Blob size : ", width=30, anchor='w', font=("Consolas", 10))
label_MAX_Blob_size.place(x=5, y=400, height=40, width=150)
bar_MAX_Blob_size = tk.Scale(PreprocessingParamFrame, from_= 1, to= 200000, orient=tk.HORIZONTAL, command=bar_changed_MAX_Blob_size)
bar_MAX_Blob_size.set(blob_size_MAX)
bar_MAX_Blob_size.place(x=200, y=400, height=40, width=270)

# ROI_devide_per 드래그 바 : in Preprocessing Parameter 프레임
label_ROI_devide_per = Label(PreprocessingParamFrame, text=" Devide ROI % : ", width=30, anchor='w', font=("Consolas", 10))
label_ROI_devide_per.place(x=5, y=440, height=40, width=150)
bar_ROI_devide_per = tk.Scale(PreprocessingParamFrame, from_= 0, to_ = 10, orient=tk.HORIZONTAL, command=bar_changed_ROI_devide_per)
bar_ROI_devide_per.set(ROI_devide_Per)
bar_ROI_devide_per.place(x=200, y=440, height=40, width=270)

# Canny 버튼 추가
button_Canny = tk.Button(PreprocessingParamFrame, text="Canny OFF", anchor='center', font=("Consolas", 10), command=button_click_Canny)
button_Canny.place(x=10, y=485, height=25, width=120)

# BRIEF 버튼 추가
button_BRIEF = tk.Button(PreprocessingParamFrame, text="BRIEF", anchor='center', font=("Consolas", 10), command=button_click_BRIEF)
button_BRIEF.place(x=140, y=485, height=25, width=120)

# ORB 버튼 추가
button_ORB = tk.Button(PreprocessingParamFrame, text="ORB", anchor='center', font=("Consolas", 10), command=button_click_ORB)
button_ORB.place(x=270, y=485, height=25, width=120)


image_viewer = ImageViewer(root)
# 이미지를 표시할 때 리스트박스에서 이미지 선택하도록 설정
image_listbox.bind("<Double-Button-1>", display_selected_image)

root.mainloop()
