"""Create the input data pipeline using `tf.data`"""

from multiprocessing import Pool
from scipy import signal
from scipy import fft
from scipy.io import wavfile
import scipy as sp
import librosa
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import math


def readfile(soundname):
    fs,wave = sp.io.wavfile.read(soundname)
    wave_fft = librosa.stft(wave, n_fft=512, hop_length=128, win_length=512)
    wave_fft = np.abs(wave_fft) ** 2
    wave_fft =  wave_fft[4:232]
    image = wave_fft
    return image 

def batchreadFile(fileList):
    signal = readfile(fileList[0])
    if len(fileList)>1:
        dampening_factor = 1./float(len(fileList[1]))
        randNoise=fileList[1]
        for i in xrange(len(randNoise)):
            signal += dampening_factor*readfile(randNoise[i])
    return np.expand_dims(np.log(signal),axis=-1) 
 
def readfiles(minibatch, noiseNames, add_noise_augmentation=False, number_of_thread=1):
    numberNoise=3
    fileList = []
    if add_noise_augmentation:
        for fileName in minibatch[0]:
            fileList.append([fileName,np.random.choice(noiseNames, numberNoise)]) 
    else:
        for fileName in minibatch[0]:
            fileList.append([fileName])
    p = Pool(number_of_thread)
    #print(len(fileList),len(minibatch[1]))
    returned = [np.array(p.map(batchreadFile, fileList)),minibatch[1]]
    p.close()
    return returned


class Batchiterator:
    def __init__(self,fileNames, labels, noiseNames=None, params=None):
       self.minibatches = random_mini_batches(fileNames,labels, mini_batch_size=params.batch_size) 
       self.minibatchesIndex=0
       self.params = params
       self.noiseNames=noiseNames
       self.labels= labels
       self.fileNames = fileNames
       if self.noiseNames is None:
            self.params.add_noise_augmentation=False
    def __iter__(self):
        return self
    def __next__(self):
        if self.minibatchesIndex == len(self.minibatches):
            raise StopIteration()
        minibatch = readfiles(self.minibatches[self.minibatchesIndex], noiseNames=self.noiseNames, number_of_thread=self.params.num_parallel_calls, add_noise_augmentation=self.params.add_noise_augmentation)
        self.minibatchesIndex+=1
        assert len(minibatch[0]) == len(minibatch[1])
        return minibatch
    def __initial__(self):
        self.minibatchesIndex=0
        self.minibatches = random_mini_batches(self.fileNames,self.labels, mini_batch_size=self.params.batch_size)
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = len(X)                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



        
