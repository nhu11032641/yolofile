import torch
import time
import numpy as np
import librosa
import pyaudio
import wave
import matplotlib.pyplot as plt
from collections import Counter
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights_m32_500/best.pt')

def run(img):
    # img = 't.png'
    arr = []
    w = ""
    results = model(img)
    for det in results.xyxy[0].cpu().numpy().tolist():
        class_id = int(det[5])  # 提取类别信息
        labelnum = class_id
        arr.append(labelnum)
    # print(arr)
    if(len(arr) == 0):
        w = "沒有目標"
    else:
        m = np.argmax(np.bincount(arr))
        if(m == 0): 
            # print("諸羅樹蛙") 
            w = "諸羅樹蛙"
        elif(m == 1): 
            # print("台北樹蛙") 
            w = "台北樹蛙"
    return w


def audio_to_img(path):
        ##print(i)
        y,sr = librosa.load(path=path, sr=44000)
        # 計算梅爾頻譜
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            # 繪製梅爾頻譜圖
        plt.figure()
        S_dB = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(S_dB,sr=sr)
        plt.axis('off')  # 不顯示坐標軸和邊框
        plt.savefig("run.png", bbox_inches='tight', pad_inches=0)
        plt.close()
    

res = []
e = True
while(e):
    chunk = 1024                     
    sample_format = pyaudio.paInt16  
    channels = 1                     
    fs = 44100                       
    seconds = 3                      
    filename = "test.mp3"  
    p = pyaudio.PyAudio()
    print("開始錄音...")

# 開啟錄音串流
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)

    frames = []                      

    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)          

    stream.stop_stream()             
    stream.close()                   
    p.terminate()

    print('錄音結束...')

    wf = wave.open(filename, 'wb')  
    wf.setnchannels(channels)        
    wf.setsampwidth(p.get_sample_size(sample_format)) 
    wf.setframerate(fs)              
    wf.writeframes(b''.join(frames)) 
    wf.close()
    audio_to_img('test.mp3')
    bb = run('run.png')
    res.append(bb)
    if(len(res) == 5):
        tt = time.time()
        print(res)
        maxlabel = max(res,key=res.count)
        fw = open("result.txt","a")
        fw.write(maxlabel+": "+ time.ctime(tt) +"\n")
        fw.close()
        print(maxlabel)
        res = []