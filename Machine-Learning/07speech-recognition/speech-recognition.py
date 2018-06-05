# -*- coding: utf-8 -*-
"""
@Created on 2018/4/9 0009 下午 4:41

@author: ZhifengFang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 读取和绘制音频数据
'''
# 读取音频文件
sampling_freq, audio = wavfile.read('input_read.wav')

# 打印相关信息
print('shape:',audio.shape)
print('sampling_freq:',sampling_freq)
print('datatype:',audio.dtype)
print('duration:',round(audio.shape[0]/float(sampling_freq),3),'seconds')

# 标准化数值
audio=audio/(2.**15)  # 2.**15表示2的15次方

# 提取前30个值，并将其画出
audio=audio[:30]

# x轴为时间轴，创建这个轴，并且x轴应该按照频率因子进行缩放
x_values=np.arange(0,len(audio),1)/float(sampling_freq)

# 将单位转换成秒
x_values*=1000

# 将其画出
plt.plot(x_values,audio,color='black')
plt.xlabel('Time(ms)')
plt.ylabel('Amplitude')
plt.title('audio')
plt.show()
'''
# 将音频信号转换成频域
'''
sampling_freq, audio = wavfile.read('input_freq.wav')
audio = audio / (2. ** 15)
len_audio = len(audio)

# 傅里叶变换
transformed_signal = np.fft.fft(audio)
half_lenth = int(np.ceil((len(audio) + 1) / 2.0))
print(half_lenth)
transformed_signal = abs(transformed_signal[0:half_lenth])
transformed_signal /= float(len_audio)
transformed_signal **= 2

# 提取信号的长度
len_ts = len(transformed_signal)

# 将部份信号乘以2
if len_audio % 2 == 0:
    transformed_signal[1:len_ts] *= 2
else:
    transformed_signal[1:len_ts - 1] *= 2

# 功率信号用下面的公式获得
power=10* np.log10(transformed_signal)

# x轴是时间轴，对其进行缩放
x_values=np.arange(0,half_lenth,1)*(sampling_freq/len_audio)/1000.0

# 绘制信号
plt.figure()
plt.plot(x_values,power,color='black')
plt.show()
'''
# 自定义参数生成音域信号
