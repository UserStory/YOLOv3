from PIL import Image
from imgaug import augmenters as iaa
from functools import reduce
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import numpy as np


def compose(*funcs):
    if funcs:
        return reduce(lambda f,g:lambda *a,**kw:g(f(*a,**kw)),funcs)
    else:
        raise ValueError('Composition of empty sequence not supported')


def rand(a=0,b=1):
    '''
    create a random value in (a,b]
    :param a: the left bound of the random value
    :param b: the right bound of the random value (not included)
    :return: a random value between (a,b]
    '''
    return np.random.rand()*(b-a)+a

# def get_random_data(annotation_line, input_shape, random=True, max_boxes = 20, im_scale_l=-0.3, im_scale_r=0.3,im_trans=0.8, hue=.1, sat=1.5, val=1.5,proc_img=True):
#
#     line = annotation_line.split()
#     image = Image.open(line[0])
#     iw,ih = image.size
#     w,h = input_shape
#     boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
#
#     if not random:
#         # resize image
#         scale = min(w/iw, h/ih)
#         nw = int(iw*scale)
#         nh = int(ih*scale)
#         dx = (w-nw)//2
#         dy = (h-nh)//2
#
#         new_image = Image.new('RGB', (w,h), (128,128,128))
#         image = image.resize((nw,nh), Image.BICUBIC)
#         new_image.paste(image, (dx, dy))
#
#         if proc_img:
#             image_data = np.array(new_image)/255.
#         else:
#             image_data = np.array(new_image)
#
#
#         # correct boxes
#         boxes_data = np.zeros((max_boxes,5))
#         if len(boxes)>0:
#             np.random.shuffle(boxes)
#             if len(box)>max_boxes: boxes = boxes[:max_boxes]
#             boxes[:, [0,2]] = boxes[:, [0,2]]*scale + dx
#             boxes[:, [1,3]] = boxes[:, [1,3]]*scale + dy
#             boxes_data[:len(boxes)] = boxes
#
#         return image_data, boxes_data
#
#     scale = min(w/iw,h/ih)
#     scale = rand(1-im_scale_l,1+im_scale_r)*scale
#     nw,nh = int(iw*scale),int(ih*scale)
#     image = image.resize((nw,nh),Image.BICUBIC)
#
#     im_trans_x = rand(1-im_trans,1+im_trans)
#     im_trans_y = rand(1-im_trans,1+im_trans)
#     dx = int(im_trans_x*(w-nw)/2.0)
#     dy = int(im_trans_y*(h-nh)/2.0)
#
#     new_image = Image.new('RGB', (416,416), (128,128,128))
#     new_image.paste(image,(dx,dy))
#
#     image = new_image
#
#     flipped = rand()<0.5
#     if flipped:
#         image = image.transpose(Image.FLIP_LEFT_RIGHT)
#
#     # distort image
#     hue = rand(-hue, hue)
#     sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
#     val = rand(1, val) if rand()<.5 else 1/rand(1, val)
#     x = rgb_to_hsv(np.array(image)/255.)
#     x[..., 0] += hue
#     x[..., 0][x[..., 0]>1] -= 1
#     x[..., 0][x[..., 0]<0] += 1
#     x[..., 1] *= sat
#     x[..., 2] *= val
#     x[x>1] = 1
#     x[x<0] = 0
#     image_data = hsv_to_rgb(x)
#     boxes_data = np.zeros((max_boxes,5))
#
#     if len(boxes)>0:
#         np.random.shuffle(boxes)
#         boxes[:,[0,2]] = boxes[:,[0,2]]*nw/iw + dx
#         boxes[:,[1,3]] = boxes[:,[1,3]]*nh/ih + dy
#         if flipped:
#             print('flipped')
#             boxes[:, [0,2]] = w - boxes[:, [2,0]]
#
#         boxes_w_b = boxes[:, 2] - boxes[:, 0]
#         boxes_h_b = boxes[:, 3] - boxes[:, 1]
#
#         boxes[:,0] = np.maximum(boxes[:,0],0)
#         boxes[:,1] = np.maximum(boxes[:,1],0)
#         boxes[:,2] = np.minimum(boxes[:,2],w-1)
#         boxes[:,3] = np.minimum(boxes[:,3],h-1)
#         boxes_w_a = boxes[:, 2] - boxes[:, 0]
#         boxes_h_a = boxes[:, 3] - boxes[:, 1]
#
#         ratio = (boxes_w_a*boxes_h_a)/(boxes_w_b*boxes_h_b)
#         boxes = boxes[ratio>0.40]
#         if len(boxes)>max_boxes: boxes = boxes[:max_boxes]
#         boxes_data[:len(boxes)] = boxes
#
#     return image_data,boxes_data

seq = iaa.SomeOf((0,5),[
    iaa.OneOf([iaa.GaussianBlur(sigma=(0.0,1.0)),iaa.AverageBlur(k=3),iaa.MedianBlur(k=3)]),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.1*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Dropout((0.1, 0.2), per_channel=0.5),
    iaa.Add((-20, 20), per_channel=0.5)
],random_order=True)

def get_random_data(annotation_line, input_shape, random=True, max_boxes = 20, im_scale_l=-0.3, im_scale_r=0.3,im_trans=0.8, hue=.1, sat=1.5, val=1.5,proc_img=True):

    line = annotation_line.split()
    image = Image.open(line[0])
    iw,ih = image.size
    w,h = input_shape
    boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        new_image = Image.new('RGB', (w,h), (128,128,128))
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image.paste(image, (dx, dy))

        if proc_img:
            image_data = np.array(new_image)/255.
        else:
            image_data = np.array(new_image)


        # correct boxes
        boxes_data = np.zeros((max_boxes,5))
        if len(boxes)>0:
            np.random.shuffle(boxes)
            if len(boxes)>max_boxes: boxes = boxes[:max_boxes]
            boxes[:, [0,2]] = boxes[:, [0,2]]*scale + dx
            boxes[:, [1,3]] = boxes[:, [1,3]]*scale + dy
            boxes_data[:len(boxes)] = boxes

        return image_data, boxes_data

    scale = min(w/iw,h/ih)
    scale = rand(1-im_scale_l,1+im_scale_r)*scale
    nw,nh = int(iw*scale),int(ih*scale)
    image = image.resize((nw,nh),Image.BICUBIC)

    im_trans_x = rand(1-im_trans,1+im_trans)
    im_trans_y = rand(1-im_trans,1+im_trans)
    dx = int(im_trans_x*(w-nw)/2.0)
    dy = int(im_trans_y*(h-nh)/2.0)

    new_image = Image.new('RGB', (416,416), (128,128,128))
    new_image.paste(image,(dx,dy))

    image = new_image

    flipped = rand()<0.5
    if flipped:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    image = np.array(image)
    image = seq.augment_image(image)
    image_data = image/255.0


    boxes_data = np.zeros((max_boxes,5))

    if len(boxes)>0:
        np.random.shuffle(boxes)
        boxes[:,[0,2]] = boxes[:,[0,2]]*nw/iw + dx
        boxes[:,[1,3]] = boxes[:,[1,3]]*nh/ih + dy
        if flipped:
            boxes[:, [0,2]] = w - boxes[:, [2,0]]

        boxes_w_b = boxes[:, 2] - boxes[:, 0]
        boxes_h_b = boxes[:, 3] - boxes[:, 1]

        boxes[:,0] = np.maximum(boxes[:,0],0)
        boxes[:,1] = np.maximum(boxes[:,1],0)
        boxes[:,2] = np.minimum(boxes[:,2],w-1)
        boxes[:,3] = np.minimum(boxes[:,3],h-1)
        boxes_w_a = boxes[:, 2] - boxes[:, 0]
        boxes_h_a = boxes[:, 3] - boxes[:, 1]

        ratio = (boxes_w_a*boxes_h_a)/(boxes_w_b*boxes_h_b)
        boxes = boxes[ratio>0.40]
        if len(boxes)>max_boxes: boxes = boxes[:max_boxes]
        boxes_data[:len(boxes)] = boxes

    return image_data,boxes_data
