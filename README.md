# Hand Gesture Volume Control 🎵✋
Control your **system volume** using **hand gestures** in real-time with **computer vision** and **AI**.

[![Watch on YouTube](https://img.shields.io/badge/YouTube-Demo-red?logo=youtube)](https://www.youtube.com/watch?v=czzw8jrtobU)

---

## 📜 Description
This application detects the position of your **thumb** and **index finger** using **MediaPipe Hands** and **OpenCV**, then calculates the distance between them to adjust the system volume via **PyCaw**.

- **Close fingers together** → volume decreases 🔉  
- **Spread fingers apart** → volume increases 🔊  

---

## 🧠 How It Works (Scientific & AI-based)
1. **Video Capture**: Webcam captures frames in real-time.
2. **Hand Detection**: The **MediaPipe Hands** deep learning model identifies 21 key landmarks of the hand.
3. **Landmark Extraction**: Focus on landmark ID `4` (thumb tip) and `8` (index finger tip).
4. **Distance Calculation**: Using Euclidean distance (Pythagoras theorem).
5. **Mapping Range**: Distance is mapped to the system volume range using NumPy interpolation.
6. **Volume Control**: **PyCaw** communicates with Windows audio API to set master volume (in dB).
7. **Visual Feedback**: Circles and lines are drawn between fingers to visualize control.

**Technologies Used**:
- Computer Vision (**OpenCV**)
- Pose Estimation (**MediaPipe Hands**)
- Signal Mapping (**NumPy interpolation**)
- System Audio Control (**PyCaw**)

---

## 🖥 Installation
Make sure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

-📦 Requirements.txt : 
    -opencv-python
    -cvzone
    -mediapipe
    -pycaw
    -comtypes
    -numpy



---
# کنترل صدای سیستم با حرکات دست 🎵✋

این پروژه به شما امکان می‌دهد که **صدای سیستم** را با استفاده از **ژست دست** و به صورت **بلادرنگ** (Real-Time) کنترل کنید.


[![ویدیو آزمایش پروژه در یوتویوب](https://img.shields.io/badge/YouTube-Demo-red?logo=youtube)](https://www.youtube.com/watch?v=czzw8jrtobU)



## 📜 توضیحات
برنامه با استفاده از **MediaPipe Hands** و **OpenCV** موقعیت انگشت شست و اشاره را شناسایی کرده و فاصله بین آن‌ها را محاسبه می‌کند. سپس با استفاده از **PyCaw** صدای سیستم تغییر می‌کند.

- **نزدیک کردن انگشت‌ها** → کاهش صدا 🔉
- **دور کردن انگشت‌ها** → افزایش صدا 🔊

## 🧠 روند علمی و مبتنی بر هوش مصنوعی
1. **دریافت ویدیو**: وبکم تصویر را به صورت زنده دریافت می‌کند.
2. **تشخیص دست**: مدل یادگیری عمیق **MediaPipe Hands**، ۲۱ نقطه کلیدی روی دست را شناسایی می‌کند.
3. **استخراج نقاط مهم**: تمرکز روی نقاط `4` (نوک شست) و `8` (نوک اشاره).
4. **محاسبه فاصله**: استفاده از فرمول فیثاغورس برای محاسبه فاصله بین این دو نقطه.
5. **نگاشت به بازه صدا**: فاصله به کمک `np.interp` به بازه صدای سیستم نگاشت می‌شود.
6. **تغییر صدا**: کتابخانه **PyCaw** با API صدای ویندوز ارتباط گرفته و ولوم سیستم را تغییر می‌دهد.
7. **نمایش گرافیکی**: با رسم دایره و خط بین انگشت‌ها، روند کنترل صدا بصری‌سازی می‌شود.

این پروژه ترکیبی از:
- **بینایی کامپیوتر (OpenCV)**
- **تشخیص حالت (MediaPipe Hands)**
- **نگاشت سیگنال (NumPy interpolation)**
- **کنترل صدای سیستم (PyCaw)**




## 🖥 نصب
## 🖥 Installation




```bash
pip install -r requirements.txt
```



```bash
import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = interface.QueryInterface(IAudioEndpointVolume)

min_vol, max_vol, increment = volume.GetVolumeRange()
print(f"Volume range: {min_vol} dB to {max_vol} dB")

current_vol = volume.GetMasterVolumeLevel()
print(f"Current volume: {current_vol} dB")

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:
    succsses, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
        lm_list = []
        for id, lm in enumerate(hand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])
            if len(lm_list) > 8:
                x1, y1 = lm_list[4][1], lm_list[4][2]
                x2, y2 = lm_list[8][1], lm_list[8][2]
                cx, cy = int(x1 + x2) // 2, int(y1 + y2) // 2
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                length = int(math.hypot(x2 - x1, y2 - y1))
                hand_range = [30, 110]
                vol = int(np.interp(length, hand_range, [-50.0, 0.0]))
                volume.SetMasterVolumeLevel(vol, None)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

