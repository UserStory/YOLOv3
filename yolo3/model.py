
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from yolo3.utils import compose
from functools import wraps

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
    return x # 1+ (1+1*2)+(1+2*2)+(1+2*8)+(1+2*8)+(1+2*4)= 52 convolutional layer

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
        from this point of view, there is no different between the YoloV3 and the tradition object detection alogrithms
        which exhaust sliding windows
    '''
    # Create Darknet body
    darknet = Model(inputs,darknet_body(inputs))

    # Create First prediction layer feature map size (h,w)//32
    # the default value (h,w)=(416,416), feature map size (416,416)//2**5 (13,13), therefore, the maximum bounding boxes is 13*13*num_anchors
    # usually, this layer is to predict big bounding boxes
    # x is the output of darknet body follwed by 5 convolutional layer
    x,y1 = make_last_layers(darknet.output,512,num_anchors*(num_classes+5))

    # The second prediction layer feature map size (h,w)//16
    # the default value (h,w)=(416,416), feature map size (416,416)//2**4 (26,26), therefore, the maximum bounding boxes is 26*26*num_anchors
    # usually, this layer is to preidct medium bounding boxes
    x = compose(
        DarknetConv2D_BN_Leak(256,(1,1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x,y2 = make_last_layers(x,256,num_anchors*(num_classes+5))

    # The third prediction layer feature map size (h,w)//8
    # the defealt value (h,w)=(416,416), feature map size （416,416)//2**3 (52,52), therefore, the maximum bounding boxes is 52*52*num_anchors
    # usually, this layer is to predict the smallest bounding boxes
    x = compose(
        DarknetConv2D_BN_Leak(128,(1,1)),
        UpSampling2D(2)(x)
    )
    x = Concatenate()([x,darknet.layers[92].output])
    x,y3 = make_last_layers(x,128,num_anchors*(num_classes+5))

    return Model(inputs,[y1,y2,y3])

def yolo_head(feats,anchors,num_classes,input_shape,calc_loss=False):
    '''This function convert the final layer outputs to bounding box parameters
        feats.shape = [batch_size,feats_h,feats_w,num_anchors*(num_classes+5)]
        anchors.shape = [num_anchors,2]
    '''

    num_anchors = len(anchors)
    grid_shape = K.shape(feats)[1:3]
    #feats.shape = [batch_size,13,13,num_anchors,num_classes+len(cx,cy,w,h,confidence)]
    feats = K.reshape(feats,[-1,grid_shape[0],grid_shape[1],num_anchors,num_classes+5])

    # anchors_tensor.shape = [1,1,1,num_anchors,2] for broadcasting
    anchors_tensor = K.reshape(K.constant(anchors),[1,1,1,num_anchors,2]) # this is for broadcasting

    # create feature map grid.
    # The yolo method predicts the bounding box relative to the left top point of the cell
    # Considering np.meshgrid to understand the following code
    grid_shape = K.shape(feats)

    # for the default value (416,416) and 5 maxpooling
    grid_y = K.arange(0,grid_shape[0]) # shape = (13,)
    grid_y = K.expand_dims(grid_y,axis=1) # shape = (13,1)
    grid_y = K.tile(grid_y,[1,grid_shape[1]]) # shape = (13,13)

    grid_x = K.arange(0,grid_shape[1]) # shape = (13,)
    grid_x = K.expand_dims(grid_x,axis=0) # shape = (1,13)
    grid_x = K.tile(grid_x,[grid_shape[0],1]) # shape = (13,13)

    grid = K.stack((grid_x,grid_y),axis=-1) # shape = (13,13,2)
    grid = K.expand_dims(grid,axis=2) # shape = (13,13,1,2) this is for broadcasting
    grid = K.expand_dims(grid,axis=0) # shape = (1,13,13,1,2) this is for broadcasting
    grid = K.cast(grid,K.dtype(feats)) # shape = (1,13,13,1,2) the last dim x,y

    # adjust prediction to each spatial grid point and anchor size
    box_xy = (K.sigmoid(feats[...,:2])+grid)/K.cast(grid_shape[::-1],K.dtype(feats))
    box_wh = K.exp(feats[...,2:4])*anchors_tensor / K.cast(input_shape[::-1],K.dtype(feats))
    box_confidence = K.sigmoid(feats[...,4:5]) # shape = [batch_size,13,13,num_anchors,1]
    box_class_probs = K.sigmoid(feats[...,5:]) # shape = [batch_size,13,13,num_anchors,num_classes]

    if calc_loss == True:
        return grid, feats, box_xy, box_wh

    return box_xy,box_wh,box_confidence,box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''
    This function adjust the predicted bounding box to fit the original image

    that is, the boudning box are predicted relative to a input shape (416,416), and we need to scale them to fit the original image

    :param box_xy:
    :param box_wh:
    :param input_shape:
    :param image_shape:
    :return:
    '''
    box_yx = box_xy[...,::-1]
    box_hw = box_wh[...,::-1]
    input_shape = K.cast(input_shape,K.dtype(box_yx))
    image_shape = K.cast(image_shape,K.dtype(box_yx))
    new_shape = K.round(image_shape*K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape


    box_yx = (box_yx-offset)*input_shape # To find the absolute value of the box_center
    box_hw = box_hw*input_shape # To find the absolute value of the box width and height

    box_yx = box_yx/new_shape
    box_hw = box_hw/new_shape

    box_mins = box_yx-(box_hw/2.0)
    box_maxes = box_yx+(box_hw/2.0)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min 0:1 can keep the last dimension, if we use 0 rather than 0:1, we need to change concatenate to stack
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ],axis=-1)

    #scale boxes back to original image shape

    boxes  = boxes * K.concatenate([image_shape,image_shape])
    return boxes

def yolo_boxes_and_scores(feats,anchors,num_classes,input_shape,image_shape):
    '''
        Process Convlayer output
        This function can only be applied to one image.
        Because the box scores and boxes are reshape to 2 dimensional tensors
    '''
    box_xy,box_wh,box_confidence,box_class_probs = yolo_head(feats,anchors,num_classes,input_shape)
    boxes = yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape)

    # boxes.shape = [batch_size,13,13,num_anchors,4]
    boxes = K.reshape(boxes,[-1,4]) # boxes.shape = [1*13*13*num_anchors,4]

    # box_confidence.shape = [batch_size,13,13,num_anchors,1]
    # box_class_probs.shape = [batch_size,13,13,num_anchors,num_classes]

    box_scores = box_confidence*box_class_probs # [bathc_size,13,13, num_anchors, num_classes]
    box_scores = K.reshape(box_scores,[-1,num_classes]) #[1*13*13*num_anchors,num_classes]
    return boxes,box_scores


def yolo_eval(yolo_outputs,anchors,num_classes,image_shape,max_boxes=20,score_threshold=0.6,iou_threshold=0.5):
    '''
        This function can only be used to detection one image
    '''
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8],[3,4,5],[0,1,2]] if num_layers == 3 else [[3,4,5],[1,2,3]]
    input_shape = K.shape(yolo_outputs[0])[1:3]*32
    boxes = []
    box_scores = []

    for l in range(num_layers):
        #_boxes.shape = [1*13*13*num_anchors,4]
        #_box_scores.shape = [1*13*13*num_anchors,num_classes]
        _boxes,_box_scores = yolo_boxes_and_scores(yolo_outputs[1],anchors[anchor_mask[l]],num_classes,input_shape,image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes,axis=0) #shape = [1*13*13*num_anchors*3,4]
    box_scores = K.concatenate( box_scores,axis=0) #shape = [1*13*13*num_anchors*3,num_classes]

    mask = box_scores >= score_threshold # shape = [1*13*13*num_anchors*3,num_classes] boolean value
    max_boxes_tensor = K.constant(max_boxes,dtype='int32')

    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(num_classes):

        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_




def preproce_true_boxes(true_boxes,input_shape,anchors,num_classes,num_layers=3,anchor_mask=[[6,7,8],[3,4,5],[0,1,2]],num_layer_poolings=[5,4,3]):
    '''Preprocess true boxes to training input format


    :param true_boxes: array, shape=(m, T, 5) m= batch_size, T=max_boxes
        Absolute x_min, y_min, x_max, y_max, and class_id
    :param input_shape: array, hw, multiples of 32
    :param anchors: array, shape=(N,2) w,h
    :param num_classes: integer
    :return:
    y_true: list of array, shape like yolo_outputs, xywh are relative value

    first, for the gt box in each image, find the best anchor in all the pre-defined anchors
    second, find the prediction layer that the bset anchor sits and set the right x,y position of each gt boxes.

    '''

    assert len(true_boxes.shape) == 3 and true_boxes.shape[-1] == 5
    assert len(input_shape) == 2
    assert len(anchors.shape)== 2 and anchors.shape[-1] == 2
    assert num_layers == len(anchor_mask) and num_layers == len(num_layer_poolings)

    # true_boxes.shape = (batch_size,max_boxes,5) last_dim->[xmin,ymin,xmax,ymax,class_id]
    true_boxes = np.array(true_boxes,dtype='float32')
    # input_shape = [h,w]
    input_shape = np.array(input_shape,dtype='int32')
    # boxes_xy.shape = [batch_size, max_boxes, 2] last_dim->[cx,cy]
    boxes_xy = (true_boxes[...,0:2]+true_boxes[...,2:4]) // 2
    # boxes_wh.shape = [batch_size, max_boxes, 2] last_dim->[w,h]
    boxes_wh = true_boxes[...,2:4] - true_boxes[...,0:2]
    # valid_mask.shape = [batch_size, max_boxes] boolean value
    valid_mask = boxes_wh[...,0] > 0

    # transform absolute x1,y1,x2,y2 to cx,cy,w,h relative to input image size
    true_boxes[...,0:2] = boxes_xy/input_shape[::-1]
    true_boxes[...,2:4] = boxes_wh/input_shape[::-1]


    # m = batch_size
    m = true_boxes.shape[0]
    # for three prediction layers which used to for
    n_anchors = len(anchors)
    grid_shapes = [input_shape//2**t for t in num_layer_poolings]
    #
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),dtype='float32') for l in range(num_layers)]

    #For broadcasting
    anchors = np.expand_dims(anchors,0)#anchors.shape = [1,num_of_anchors,2]
    anchors = np.tile(anchors,[m,1,1])#anchors.shape = [num_valid_boxes,num_of_anchors,2]
    anchor_maxes = anchors/2.0
    anchor_mins = -anchor_maxes

    #To find the best anchor for each bounding box in each image
    for b in range(m):

        wh = boxes_wh[b,valid_mask[b],:] # wh.shape = [num_valid_boxes,2]
        if len(wh): # continue if the sample image has no valid bounding boxes
            continue
        wh = np.expand_dims(wh,1) # wh.shape = [num_valid_boxes,1,2]
        wh = np.tile(wh,[1,n_anchors,1]) # wh.shape = [num_valid_boxes,num_of_anchors,2]
        box_maxes = wh/2.0
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins,anchor_mins) # shape = [num_valid_boxes,num_anchors,2]
        intersect_maxes = np.minimum(box_maxes,anchor_maxes) # shape = [num_valid_boxes,num_anchors,2]
        intersect_wh = np.maximum(intersect_maxes-intersect_mins,0) # shape = [num_valid_boxes,num_anchors,2]
        intersect_area = intersect_wh[...,0]*intersect_wh[..., 1] # shape = [num_valid_boxes,num_anchors]
        box_area = wh[...,0]*wh[...,1] #shape = [num_valied_boxes,num_anchors]
        anchor_area = anchors[..., 0] * anchors[..., 1] #shape = [num_valid_boxes,num_anchors]
        iou = intersect_area/(box_area+anchor_area-intersect_area)#shape = [num_valid_boxes,num_anchors]

        best_anchor = np.argmax(iou,axis=1)#shape=[num_valid_boxes,num_anchors]

        # enumerate the boudning box
        for t,n in enumerate(best_anchor):
            for l in range(num_layers):
                # find which layer the best anchor sits
                if n in anchor_mask[l]:
                    #  find the x,y position of the bounding box sits in the layer
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1].astype('int32'))
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0].astype('int32'))
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t,4].astype('int32')

                    y_true[l][b,j,i,k,0:4]=true_boxes[b,t,0:4]
                    y_true[l][b,j,i,k,4] = 1
                    y_true[l][b,j,i,k,5+c] = 1
    return y_true


def box_iou(b1, b2):

    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5] #[b,13,13,num_anchors,1]
        true_class_probs = y_true[l][..., 5:] #[b,13,13,num_anchors,num_classes]

        # grid.shape = [13,13,1,2]
        # raw.pred = [b,13,13,num_anchors,5+num_classes]
        # pred_xy.shape = [b,13,13,num_anchors,2]
        # pred_wh.shape = [b,13,13,num_anchors,2]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)

        # pred_box.shape = [b,13,13,num_anchors,4]
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        # raw_true_xy.shape = [b,13,13,num_anchors,2]
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        # object_mask.shape = [b,13,13,num_anchors,1]
        # raw_true_wh.shape = [b,13,13,num_anchors,2]
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        # shape = [m,13,13,anchors,1]
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        # For some predicted bounding boxes, if the iou between these boxes and any true boxes is higher than a threshold,
        # then, we should give up to calculate the no object loss.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool') # shape = [b,13,13,num_anchors,1]
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            # iou.shape = [13,13,num_anchors,j] j the num of true boxes
            iou = box_iou(pred_box[b], true_box)
            # best_iou.shape = [13,13,num_anchors]
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        # ignore_mask a list of [13,13,num_anchors]
        # stack to [m,13,13,num_anchors]
        ignore_mask = ignore_mask.stack()
        # shape = [m,13,13,num_anchors,1]
        ignore_mask = K.expand_dims(ignore_mask, -1)


        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
