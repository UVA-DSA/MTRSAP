import cv2
import pandas as pd
from tqdm import tqdm

label_mapping ="""G1  Reaching for needle with right hand
G2  Positioning needle
G3  Pushing needle through tissue
G4  Transferring needle from left to right
G5  Moving to center with needle in grip
G6  Pulling suture with left hand
G7  Pulling suture with right hand
G8  Orienting needle
G9  Using right hand to help tighten suture
G10  Loosening more suture
G11  Dropping suture at end and moving to end points
G12  Reaching for needle with left hand
G13  Making C loop around right hand
G14  Reaching for suture with right hand
G15  Pulling suture with both hands"""


label_mapping = {g.split('  ')[0]: g.split('  ')[1] for g in label_mapping.strip().split('\n')}
print(label_mapping)

def get_label(frame, labels):
    label = None
    for i, row in labels.iterrows():
        if row["start"] <= frame <= row["stop"]:
            label =  row["gesture"]
            break
    return label_mapping.get(label, "")

task = 'Needle_Passing'
s = 5
t = 1
video_path = f"./Datasets/dV/{task}/video/{task}_S0{s}_T0{t}_Left.avi"
annot_path = f"./Datasets/dV/{task}/gestures/{task}_S0{s}_T0{t}.txt"
annot_df = pd.read_csv(annot_path, delimiter=' ', header=None).set_axis(["start", 'stop', 'gesture', 'X'], axis=1)

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(f'annotated_{task}_S{s}_T{t}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

i = 0
while(True):
      
    # Capture frames in the video
    ret, frame = cap.read()

    font = cv2.FONT_HERSHEY_SIMPLEX
    sl= get_label(i, annot_df)
    cv2.putText(frame,
                sl, 
                (50, 50), 
                font, .6, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    
    i += 1
    
    # Display the resulting frame
    try:
        out.write(frame)
        # cv2.imshow('video', frame)
    except:
        pass
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if not ret:
        break
  
# release the cap object
cap.release()
out.release()
# close all windows
cv2.destroyAllWindows()
