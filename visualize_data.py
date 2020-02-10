import numpy as np
import cv2
import pandas as pd

# Open a sample videos available in sample-videos
#vcap = cv2.VideoCapture('https://s3.amazonaws.com/ava-dataset/trainval/TzaVHtLXOzY.mkv')
vcap = cv2.VideoCapture('./videos/-5KQ66BBWC4.mkv')
data_frame = pd.read_csv('./data/ava_activespeaker_train_v1.0/-5KQ66BBWC4-activespeaker.csv')
#vcap = cv2.VideoCapture('https://s3.amazonaws.com/ava-dataset/trainval/P60OxWahxBQ.mkv')
#data_frame = pd.read_csv('./data/ava_activespeaker_test_v1.0/P60OxWahxBQ-activespeaker.csv')
#vcap = cv2.VideoCapture('https://s3.amazonaws.com/ava-dataset/trainval/053oq2xB3oU.mkv')
#data_frame = pd.read_csv('./data/ava_activespeaker_test_v1.0/053oq2xB3oU-activespeaker.csv')
#if not vcap.isOpened():
#    print "File Cannot be Opened"

frame_width = int(round(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)))
frame_height = int(round(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = vcap.get(cv2.CAP_PROP_FPS)

print("FRAME WIDTH ", frame_width)
print("FRAME HGITH ", frame_height)
print("FPS", fps)

prev_frame = 0

for i, each_row in data_frame.iterrows():
    # Capture frame-by-frame
    time = each_row[1]
    current_frame = int(round(time*fps))
    top_left = (int(round(each_row[2] * frame_width)), int(round(each_row[3] * frame_height)))
    bot_right = (int(round(each_row[4] * frame_width)), int(round(each_row[5] * frame_height)))
    color = (0,0,0)
    if each_row[6] == 'SPEAKING_AUDIBLE':
        color = (0,255,0)
    elif each_row[6] == 'NOT_SPEAKING':
        color = (0,0,255)
    elif each_row[6] == 'SPEAKING_NOT_AUDIBLE':
        color = (124,255,0)
    else:
        print(each_row[6])
    #print(current_frame, prev_frame)
    if(current_frame - prev_frame > 1):
        vcap.set(1, current_frame)
    ret, frame = vcap.read()
    cv2.rectangle(frame, top_left, bot_right, color, 3)
    prev_frame = current_frame
    #vcap.set(1, 2000)
    #print cap.isOpened(), ret
    if frame is not None:
        # Display the resulting frame
        cv2.imshow('frame',frame)
        # Press q to close the videos windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print("Frame is None")
        break

# When everything done, release the capture
vcap.release()
cv2.destroyAllWindows()