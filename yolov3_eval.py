from functools import reduce
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D,Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from functools import wraps


def compose(*funcs):
    if funcs:
        return reduce(lambda f,g:lambda *a,**kw:g(f(*a,**kw)),funcs)
    else:
        raise ValueError('Composition of empty sequence not supported')

@wraps(Conv2D)
def DarknetConv2d(*args,**kwargs):
    '''Wrapper to Darknet parameters for Convolution2D'''
    darknet_conv_kwargs = {'kernel_regularizer':l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args,**darknet_conv_kwargs)

def DarknetConv2D_BN_Leak(*args,**kwargs):
    '''Darknet Convolution2D followed by BatchNormalization and LeakyReLU'''
    no_bias_kwargs = {'use_bias':False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2d(*args,**no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leak(num_filters,(3,3),strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leak(num_filters//2,(1,1)),
            DarknetConv2D_BN_Leak(num_filters,(3,3)))(x)
        x = Add()([x,y])
    return x


def darknet_body(x):
    '''Darknet body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leak(32,(3,3))(x) # 1 convolutional layer
    x = resblock_body(x,64,1) # 1+1*2 convolutional layer
    x = resblock_body(x,128,2) # 1+2*2 convolutional layer
    x = resblock_body(x,256,8) # 1+2*8 convolutional layer
    x = resblock_body(x,512,8) # 1+2*8 convolutional layer
    x = resblock_body(x,1024,4) #1+2*4 convolutional layer
    return x # 1+(1+1*2)+(1+2*2)+(1+2*8)+(1+2*8)+(1+2*4)= 52 convolutional layer

def make_last_layers(x, num_filters, out_filters):
    ''' This is for the prediction layer
    6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leak(num_filters,(1,1)), #1
        DarknetConv2D_BN_Leak(num_filters*2,(3,3)), #2
        DarknetConv2D_BN_Leak(num_filters,(1,1)), #3
        DarknetConv2D_BN_Leak(num_filters*2,(3,3)), #4
        DarknetConv2D_BN_Leak(num_filters,(1,1)))(x)#5
    y = compose(
        DarknetConv2D_BN_Leak(num_filters*2,(3,3)),#6
        DarknetConv2d(out_filters,(1,1)))(x) # the prediction layer use linear activation
    return x,y

def yolo_body(inputs, num_anchors, num_classes):
    '''
        For the defalut input image size
        YoloV3 predicts around (13*13+26*26+52*52)*num_anchors bounding boxes,
    '''
    # Create Darknet body
    darknet = Model(inputs,darknet_body(inputs))

    # Create First prediction layer feature map size (h,w)//32
    # the default value (h,w)=(416,416), feature map size (416,416)//2**5 (13,13), therefore, the maximum bounding boxes is 13*13*num_anchors
    x,y1 = make_last_layers(darknet.output,512,num_anchors*(num_classes+5))

    # The second prediction layer feature map size (h,w)//16
    # the default value (h,w)=(416,416), feature map size (416,416)//2**4 (26,26), therefore, the maximum bounding boxes is 26*26*num_anchors
    x = compose(
        DarknetConv2D_BN_Leak(256,(1,1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x,y2 = make_last_layers(x,256,num_anchors*(num_classes+5))

    # The third prediction layer feature map size (h,w)//8
    # the defealt value (h,w)=(416,416), feature map size （416,416)//2**3 (52,52), therefore, the maximum bounding boxes is 52*52*num_anchors
    x = compose(
        DarknetConv2D_BN_Leak(128,(1,1)),
        UpSampling2D(2)(x)
    )
    x = Concatenate()([x,darknet.layers[92].output])
    x,y3 = make_last_layers(x,128,num_anchors*(num_classes+5))

    return Model(inputs,[y1,y2,y3]),darknet



class YOLOv3(object):
    def __init__(self,input_shape,anchor_path, classes_path, weights_path):
        self.anchors = self.get_anchors(anchor_path)
        self.class_name = self.get_classes(classes_path)
        self.model_body,self.darknet_body = self.create_model(input_shape,self.anchors,len(self.class_name),\
                                                              weights_path=weights_path)

    def create_model(self, input_shape, anchors, num_classes, weights_path=None):
        image_input = Input(shape=(None,None,3))
        h,w = input_shape
        num_anchors = len(anchors)

        y_trues = []
        for i in [32,16,8]:
            y_true = Input(shape=(h//i,w//i,num_anchors//3,num_classes+5))
            y_trues.append(y_true)
        model_body,darknet_body = yolo_body(image_input,num_anchors//3,num_classes)

        if weights_path is not None:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

        return model_body,darknet_body

    def get_classes(self,classes_path):
        '''loads the classes'''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
