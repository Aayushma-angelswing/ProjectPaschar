import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = plt.imread("img.png")
# plt.imshow(img)
# plt.show()
# exit()

# l = [[347, 20], [531, 30], [598, 323], [87, 321]]

video_file = 0
desk_view = True

l = []
setup = True

cap = cv2.VideoCapture(video_file)
ret, frame = cap.read()
if video_file == 0:
    frame = cv2.flip(frame, 1)
img = frame.copy()
input_height, input_width, _ = img.shape

output_width = int(input_width / 1.5)
output_height = int(input_height / 1.5)


def getTransform(points):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[output_width,0],[output_width,output_height],[0,output_height]])
    return cv2.getPerspectiveTransform(pts1,pts2)


def processMouse(event,x,y,flags,params):
    global l, img, setup
    if setup:
        if event == cv2.EVENT_LBUTTONDOWN:
            l.append([x,y])
            cv2.circle(img, (x,y), 5,(255,0,0),3)
            cv2.imshow('SELECT',img)
            if(len(l)) == 4:
                pts = np.array(l, np.int32)
                img = cv2.polylines(img, [pts], True, (255,0,0), 2)
                cv2.imshow('SELECT',img)
                cv2.destroyAllWindows()
                cap.release()
                setup = False


def Fram_connect(frame1, frame2, Video_w, Video_h,  Video_w2, Video_h2):
    frame2 = cv2.resize(frame2, (int(Video_w2), int(Video_h)), interpolation = cv2.INTER_AREA)
    BG = cv2.resize(frame1, (int(Video_w + Video_w2), int(Video_h)), interpolation = cv2.INTER_AREA)
    BG[0:int(Video_h),0:int(Video_w)] = frame1
    BG[0:int(Video_h),int(Video_w):int(Video_w+ Video_w2)] = frame2
    return (BG)


cv2.namedWindow('SELECT')
cv2.setMouseCallback("SELECT",processMouse)
cv2.imshow('SELECT',img)
cv2.waitKey(0)


cap = cv2.VideoCapture(video_file)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4',fourcc, 29.97, (input_width+output_width,input_height))
# out1 = cv2.VideoWriter('output1.avi',fourcc, 29.97, (640,360))
# out2 = cv2.VideoWriter('output2.avi',fourcc, 29.97, (400,400))

while(True):
    ret, frame = cap.read()
    if video_file == 0:
        frame = cv2.flip(frame, 1)
    
    if not ret:
        break
    
    img_ori = frame.copy()

    pts = np.array(l, np.int32)
    img1 = cv2.polylines(frame, [pts], True, (255,0,0), 2)
    # out1.write(img1)
    # cv2.imshow('IMG1',img1)

    img = img_ori.copy()
    base = np.zeros((img_ori.shape[:2]))

    m = getTransform(l)
    img2 = img_ori.copy()
    dst =   cv2.warpPerspective(np.float32(img2), m, (output_width,output_height))
    dst = np.array(dst, dtype='uint8')
    
    if video_file == 0 and desk_view == True:
        dst = cv2.flip(dst,0)
    
    # out2.write(dst)
    # cv2.imshow('IMG2',dst)

    connect = Fram_connect(img1, dst, input_width, input_height,  output_width, output_height)
    cv2.imshow('Connect',connect)
    out.write(connect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
# out1.release()
# out2.release()
cap.release()
cv2.destroyAllWindows()







