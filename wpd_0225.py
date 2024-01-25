import numpy as np

class WaveletPacket:  

    def __init__(self, max_level):
        self.max_level = max_level  # max_level
        self.level_num = [2**x for x in range(self.max_level)] 
        self.W = np.zeros((self.max_level+1, self.level_num[self.max_level-1]*2)).tolist()  
        self.__q = np.array([self.__p[len(self.__p)-n-1]*(-1)**n for n in range(len(self.__p))])  
        self.extract_num = 2**self.max_level
        self.output_signal = []  

    def WPD(self, X):

        self.W[0][0] = np.array(X)  
        le = len(X) 
        if self.W[0][0].ndim != 1:
            print("error,check input")
            print("now :", self.W[0][0].ndim)
            return

        for j in range(1, self.max_level+1):
            for i in range(self.level_num[j-1]):  
                __s0 = self.W[j-1][i]
                self.W[j][2*i] = np.array([np.inner(self.__p, np.array([__s0[(n+2*times)%le] for n in range(len(self.__p))]))
                                           for times in range(int((le-2)/2))])
                self.W[j][2*i+1] = np.array([np.inner(self.__q, np.array([__s0[(n+2*times)%le] for n in range(len(self.__q))]))
                                             for times in range(int((le-2)/2))])
            le = int((le-2)/2)  

    def output(self, level, extract_num):
        if level > self.max_level or level < 0:  
            level = self.max_level
        if extract_num < self.extract_num:
            self.extract_num = extract_num
        self.output_signal = [self.W[level][k+1] for k in range(self.extract_num)]
        return self.output_signal
