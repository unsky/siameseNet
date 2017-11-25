# written by longriyao
#coding: utf-8

import caffe
import cv2
import numpy as np
import yaml
import os
class EuclideanAccLayer(caffe.Layer):
    def _get_param(self):
        # parse
        layer_params = yaml.load(self.param_str)
        # get the param

    def setup(self,bottom,top):
        self._get_param()
        self._i = 0
        top[0].reshape(1)
        pass

    def forward(self,bottom,top):
        if(bottom[0] is None):
            print "this is no images"
            return 0
        data = np.array(bottom[0].data)
        data_p = np.array(bottom[1].data)
        label = np.array(bottom[2].data)
        

        p_ = np.sqrt(np.sum(np.square(data - data_p),axis =1))
        p = np.zeros((p_.shape))
  
        p[p_<1] =1
        p[p_>=1] =0
        acc = np.sum(p.flat==label.flat)*1.0/p.shape[0]

        top[0].data[...] = acc
    def backward(self,bottom,top):
        pass
    def reshape(self,bottom,top):
        pass
