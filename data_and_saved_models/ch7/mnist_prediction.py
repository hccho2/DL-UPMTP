
'''
MNIST 분류 모델을 마우스로 입력받은 숫자로 테스트

'''
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
from PIL import Image
import numpy as np
import ctypes
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.drop_rate = 0.3

        self.L1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (28,28,32)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (14,14,32)
            torch.nn.Dropout(p=self.drop_rate))

        self.L2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  #(14,14,64)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), #(7,7,64)
            torch.nn.Dropout(p=self.drop_rate))

        self.L3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (7,7,128)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # (4,4,128)
            torch.nn.Dropout(p=self.drop_rate))

        self.L4 = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 128, 512, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_rate))

        self.L5 = torch.nn.Linear(512, 10, bias=True)

    def forward(self, x):
        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        out = torch.flatten(out,start_dim=1) #(N,4*4*128)
        out = self.L4(out)
        out = self.L5(out)
        return out # logit

transform_test = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28),interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])
def predict(model,img):
    # img: PIL.Image.Image
    model.eval()
    img_tensor = transform_test(img).unsqueeze(0).to(device) # (1,1,28,28)
    prob = model(img_tensor).softmax(-1).to('cpu').detach().numpy()
    return prob[0]

######################### Model Loading  ############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNNModel().to(device)

model_filename = 'saved_models/ch7/mnist_cnn_20.pt'
model.load_state_dict(torch.load(model_filename, map_location = device))
model.eval()

canvas = np.ones((600,600), dtype="uint8") * 255 # 255=white
canvas[100:500,100:500] = 0                      # 0 = black

start_point = None
end_point = None
is_drawing = False

def draw_line(img,start_at,end_at):
    cv2.line(img,start_at,end_at,255,15) # ~(이미지,시작점,종료점,색상,두께)~

def on_mouse_events(event,x,y,flags,params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_drawing == False:
            is_drawing=True
            start_point = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point = (x,y)
            draw_line(canvas,start_point,end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

name = "Test Canvas: c(clear), d(done)"
cv2.namedWindow(name)
cv2.setMouseCallback(name, on_mouse_events)

while(True):
    cv2.imshow(name, canvas)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('c'):  # c: clear
        canvas[100:500,100:500] = 0
    elif key == ord('d'):  # d: ~입력 완료~
            # drawing_image: uint8(0-255) numpy array(400,400) --> ~0 또는 255 밖에 없다. 중간 값이 없다.~
        drawing_image = canvas[100:500,100:500] 
        break

cv2.destroyAllWindows()
plt.imshow(drawing_image,cmap='gray')

draw_image=Image.fromarray(drawing_image) # PIL.Image.Image
prob = predict(model,draw_image) # ~확률값 return~

pred = np.argsort(prob)[::-1]  # ~정렬(내림차순)~
prob=prob[pred]

ctypes.windll.user32.MessageBoxW(0, str(pred[0]), "Prediction", 0) # ~Windows 10에서만 작동~

print('prediction: ', pred[0])
print('top 3 probability: ')
for i in range(3):
    print(f'{pred[i]}: {prob[i]*100:.2f}%')
