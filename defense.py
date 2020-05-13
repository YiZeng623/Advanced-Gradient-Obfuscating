from albumentations import augmentations
from scipy.fftpack import dct, idct, rfft, irfft
import tensorflow as tf
import numpy as np
from albumentations import *
from random import randint, uniform
import PIL
import PIL.Image
from io import BytesIO
import cv2
import random

# This file contains the defense methods compared in the paper.
# The FD algorithm's source code is from:
#   https://github.com/zihaoliu123/Feature-Distillation-DNN-Oriented-JPEG-Compression-Against-Adversarial-Examples/blob/master/utils/feature_distillation.py
# The FD algorithm is refer to the paper:
#   https://arxiv.org/pdf/1803.05787.pdf
# Some of the defense methods' code refering to Anish & Carlini's github: https://github.com/anishathalye/obfuscated-gradients


def defended(funcname, adv):
    func = eval('defend_'+funcname)
    defendadv = np.zeros((adv.shape[0],299,299,3))
    for i in range(adv.shape[0]):
        defendadv[i] = func(adv[i])
    return defendadv

def remap(img,index):
    n = img.shape[0]
    m = img.shape[1]
    if np.ndim(img)>2:
        outimgnp = np.zeros((n,m,img.shape[2]))
    else:
        outimgnp = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            outimgnp[j,i]=img[tuple(index[i][j])]
    return outimgnp


# GD algorithm that runs in the session (for EOT)
def tftensorGD(img,distort_limit = 0.25):
    num_steps = 10
    upflag = tf.round(tf.random_uniform((1,), 0, 1, dtype=tf.float32))
    leftflag = tf.round(tf.random_uniform((1,), 0, 1, dtype=tf.float32))
    xstep = tf.constant(1.0) + tf.random_uniform((num_steps + 1,),
                                                 -distort_limit,
                                                 distort_limit,
                                                 dtype=tf.float32)
    ystep = tf.constant(1.0) + tf.random_uniform((num_steps + 1,),
                                                 -distort_limit,
                                                 distort_limit,
                                                 dtype=tf.float32)
    img_shape = tf.shape(img)
    height, width = img_shape[0], img_shape[1]
    x_step = width // num_steps
    y_step = height // num_steps
    xs = tf.range(0.0, tf.dtypes.cast(width, tf.float32), delta=x_step)
    ys = tf.range(0.0, tf.dtypes.cast(height, tf.float32), delta=y_step)
    prev = tf.constant(0.0)
    listvec_x = tf.zeros((1, 1))
    for i in range(num_steps + 1):
        start = tf.cast(xs[i], tf.int32)
        end = tf.cast(xs[i], tf.int32) + x_step
        cur = tf.cond(end > width, lambda: tf.cast(width, tf.float32),
                      lambda: prev + tf.cast(x_step, tf.float32) * xstep[i])
        end = tf.cond(end > width, lambda: width, lambda: end)
        listvec_x = tf.concat([listvec_x, tf.reshape(tf.linspace(prev, cur, end - start), (1, -1))], -1)
        prev = cur
    xx = tf.cast(tf.clip_by_value(tf.round(listvec_x), 0, 298), tf.int32)
    map_x = tf.tile(xx[:, 1:], (299, 1))
    xx2 = tf.reverse((298 * tf.ones_like(xx, dtype=tf.int32) - xx), [1])
    map_x2 = tf.tile(xx2[:, :299], (299, 1))
    prev = tf.constant(0.0)
    listvec_y = tf.zeros((1, 1))
    for i in range(num_steps + 1):
        start = tf.cast(ys[i], tf.int32)
        end = tf.cast(ys[i], tf.int32) + y_step
        cur = tf.cond(end > width, lambda: tf.cast(height, tf.float32),
                      lambda: prev + tf.cast(y_step, tf.float32) * ystep[i])
        end = tf.cond(end > width, lambda: width, lambda: end)
        listvec_y = tf.concat([listvec_y, tf.reshape(tf.linspace(prev, cur, end - start), (1, -1))], -1)
        prev = cur
    yy = tf.cast(tf.clip_by_value(tf.round(listvec_y), 0, 298), tf.int32)
    map_y = tf.tile(tf.transpose(yy)[1:, :], (1, 299))
    yy2 = tf.reverse((298 * tf.ones_like(yy, dtype=tf.int32) - yy), [1])
    map_y2 = tf.tile(tf.transpose(yy2)[:299, :], (1, 299))
    index_x = tf.cond(leftflag[0] > 0.5, lambda: tf.identity(map_x), lambda: tf.identity(map_x2))
    index_y = tf.cond(upflag[0] > 0.5, lambda: tf.identity(map_y), lambda: tf.identity(map_y2))
    index = tf.stack([index_y, index_x], 2)
    x_gd = tf.gather_nd(img, index)
    return x_gd

# GD algorithm that runs outside the session (for BPDA)
def defend_GD(img,distort_limit = 0.25):
    num_steps = 10

    xsteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    ysteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]

    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur
    xx = np.round(xx).astype(int)
    yy = np.round(yy).astype(int)
    xx[xx >= 299] = 298
    yy[yy >= 299] = 298

    map_x, map_y = np.meshgrid(xx, yy)

    #     index=np.dstack((map_y,map_x))
    #     outimg = remap(img,index)
    #     if np.ndim(img)>2:
    #         outimg = outimg.transpose(1,0,2)
    #     else:
    #         outimg = outimg.T

    # to speed up the mapping procedure, OpenCV 2 is adopted
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    outimg = cv2.remap(img, map1=map_x, map2=map_y, interpolation=1, borderMode=4, borderValue=None)
    return outimg


# FD algorithm
T = np.array([
        [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
        [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
        [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
        [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
        [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
        [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
        [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
        [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]
    ])
num = 8
q_table = np.ones((num,num))*30
q_table[0:4,0:4] = 25

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
def rfft2 (block):
    return rfft(rfft(block.T).T)
def irfft2(block):
    return irfft(irfft(block.T).T)

# Feature distillation for batch
def FD_fuction(input_matrix):
    output = []
    input_matrix = input_matrix*255

    n = input_matrix.shape[0]
    h = input_matrix.shape[1]
    w = input_matrix.shape[2]
    c = input_matrix.shape[3]
    horizontal_blocks_num = w / num
    output2=np.zeros((c,h, w))
    output3=np.zeros((n,3,h, w))
    vertical_blocks_num = h / num
    n_block = np.split(input_matrix,n,axis=0)
    for i in range(n):
        c_block = np.split(n_block[i],c,axis =3)
        j=0
        for ch_block in c_block:
            vertical_blocks = np.split(ch_block, vertical_blocks_num,axis = 1)
            k=0
            for block_ver in vertical_blocks:
                hor_blocks = np.split(block_ver,horizontal_blocks_num,axis = 2)
                m=0
                for block in hor_blocks:
                    block = np.reshape(block,(num,num))
                    block = dct2(block)
                    # quantization
                    table_quantized = np.matrix.round(np.divide(block, q_table))
                    table_quantized = np.squeeze(np.asarray(table_quantized))
                    # de-quantization
                    table_unquantized = table_quantized*q_table
                    IDCT_table = idct2(table_unquantized)
                    if m==0:
                        output=IDCT_table
                    else:
                        output = np.concatenate((output,IDCT_table),axis=1)
                    m=m+1
                if k==0:
                    output1=output
                else:
                    output1 = np.concatenate((output1,output),axis=0)
                k=k+1
            output2[j] = output1
            j=j+1
        output3[i] = output2
    output3 = np.transpose(output3,(0,2,1,3))
    output3 = np.transpose(output3,(0,1,3,2))
    output3 = output3/255
    output3 = np.clip(np.float32(output3),0.0,1.0)
    return output3
def padresult(cleandata):
    pad = augmentations.transforms.PadIfNeeded(min_height=304, min_width=304, border_mode=4)
    paddata = np.ones((cleandata.shape[0],304,304,3))
    for i in range(paddata.shape[0]):
        paddata[i] = pad(image = cleandata[i])['image']
    return paddata
def cropresult(paddata):
    crop = augmentations.transforms.Crop(0,0,299,299)
    resultdata = np.ones((paddata.shape[0],299,299,3))
    for i in range(resultdata.shape[0]):
        resultdata[i] = crop(image = paddata[i])['image']
    return resultdata

def defend_FD(data):
    paddata = padresult(data)
    defendresult = FD_fuction(paddata)
    resultdata = cropresult(defendresult)
    return resultdata

# Feature distillation for single imput
def FD_fuction_sig(input_matrix):
    output = []
    input_matrix = input_matrix * 255

    h = input_matrix.shape[0]
    w = input_matrix.shape[1]
    c = input_matrix.shape[2]
    horizontal_blocks_num = w / num
    output2 = np.zeros((c, h, w))
    vertical_blocks_num = h / num

    c_block = np.split(input_matrix, c, axis=2)
    j = 0
    for ch_block in c_block:
        vertical_blocks = np.split(ch_block, vertical_blocks_num, axis=0)
        k = 0
        for block_ver in vertical_blocks:
            hor_blocks = np.split(block_ver, horizontal_blocks_num, axis=1)
            m = 0
            for block in hor_blocks:
                block = np.reshape(block, (num, num))
                block = dct2(block)
                # quantization
                table_quantized = np.matrix.round(np.divide(block, q_table))
                table_quantized = np.squeeze(np.asarray(table_quantized))
                # de-quantization
                table_unquantized = table_quantized * q_table
                IDCT_table = idct2(table_unquantized)
                if m == 0:
                    output = IDCT_table
                else:
                    output = np.concatenate((output, IDCT_table), axis=1)
                m = m + 1
            if k == 0:
                output1 = output
            else:
                output1 = np.concatenate((output1, output), axis=0)
            k = k + 1
        output2[j] = output1
        j = j + 1

    output2 = np.transpose(output2, (1, 0, 2))
    output2 = np.transpose(output2, (0, 2, 1))
    output2 = output2 / 255
    output2 = np.clip(np.float32(output2), 0.0, 1.0)
    return output2
def padresult_sig(cleandata):
    pad = augmentations.transforms.PadIfNeeded(min_height=304, min_width=304, border_mode=4)
    paddata = pad(image=cleandata)['image']
    return paddata
def cropresult_sig(paddata):
    crop = augmentations.transforms.Crop(0, 0, 299, 299)
    resultdata = crop(image=paddata)['image']
    return resultdata

def defend_FD_sig(data):
    paddata = padresult_sig(data)
    defendresult = FD_fuction_sig(paddata)
    resultdata = cropresult_sig(defendresult)
    return resultdata


#####################################################################
# Defense methods used to compare with different indifferentiable functions that combined with GD algorithm
def rand_jpeg(img):
    jpeg = JpegCompression(quality_lower=10, quality_upper=80, p=1)
    augmented = jpeg(image=((img*255).astype(np.uint8)))
    auged = (augmented['image']/255).astype(float)
    return auged

def defend_RCDfense(img):
    # attach an random Jpeg compression procedure before the GD algorithm
    img = rand_jpeg(img)
    num_steps = 10
    distort_limit = 0.1

    xsteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    ysteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]

    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur
    xx = np.round(xx).astype(int)
    yy = np.round(yy).astype(int)
    xx[xx >= 299] = 298
    yy[yy >= 299] = 298

    map_x, map_y = np.meshgrid(xx, yy)
    #     index=np.dstack((map_y,map_x))

    #     outimg = remap(img,index)
    #     if np.ndim(img)>2:
    #         outimg = outimg.transpose(1,0,2)
    #     else:
    #         outimg = outimg.T

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    outimg = cv2.remap(img, map1=map_x, map2=map_y, interpolation=1, borderMode=4, borderValue=None)

    return outimg

#####################################################################
# Defense methods used to compare the performance over the BPDA

def defend_BitReduct(arr, depth=3):
    arr = (arr * 255.0).astype(np.uint8)
    shift = 8 - depth
    arr = (arr >> shift) << shift
    arr = arr.astype(np.float32)/255.0
    return arr

def defend_FixedJpeg(input_array):
    pil_image = PIL.Image.fromarray((input_array*255.0).astype(np.uint8))
    f = BytesIO()
    pil_image.save(f, format='jpeg', quality=75) # quality level specified in paper
    jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32)/255.0
    return jpeg_image

def bregman(image, mask, weight, eps=1e-3, max_iter=100):
    rows, cols, dims = image.shape
    rows2 = rows + 2
    cols2 = cols + 2
    total = rows * cols * dims
    shape_ext = (rows2, cols2, dims)

    u = np.zeros(shape_ext)
    dx = np.zeros(shape_ext)
    dy = np.zeros(shape_ext)
    bx = np.zeros(shape_ext)
    by = np.zeros(shape_ext)

    u[1:-1, 1:-1] = image
    # reflect image
    u[0, 1:-1] = image[1, :]
    u[1:-1, 0] = image[:, 1]
    u[-1, 1:-1] = image[-2, :]
    u[1:-1, -1] = image[:, -2]

    i = 0
    rmse = np.inf
    lam = 2 * weight
    norm = (weight + 4 * lam)

    while i < max_iter and rmse > eps:
        rmse = 0

        for k in range(dims):
            for r in range(1, rows + 1):
                for c in range(1, cols + 1):
                    uprev = u[r, c, k]

                    # forward derivatives
                    ux = u[r, c + 1, k] - uprev
                    uy = u[r + 1, c, k] - uprev

                    # Gauss-Seidel method
                    if mask[r - 1, c - 1]:
                        unew = (lam * (u[r + 1, c, k] +
                                       u[r - 1, c, k] +
                                       u[r, c + 1, k] +
                                       u[r, c - 1, k] +
                                       dx[r, c - 1, k] -
                                       dx[r, c, k] +
                                       dy[r - 1, c, k] -
                                       dy[r, c, k] -
                                       bx[r, c - 1, k] +
                                       bx[r, c, k] -
                                       by[r - 1, c, k] +
                                       by[r, c, k]
                                       ) + weight * image[r - 1, c - 1, k]
                                ) / norm
                    else:
                        # similar to the update step above, except we take
                        # lim_{weight->0} of the update step, effectively
                        # ignoring the l2 loss
                        unew = (u[r + 1, c, k] +
                                u[r - 1, c, k] +
                                u[r, c + 1, k] +
                                u[r, c - 1, k] +
                                dx[r, c - 1, k] -
                                dx[r, c, k] +
                                dy[r - 1, c, k] -
                                dy[r, c, k] -
                                bx[r, c - 1, k] +
                                bx[r, c, k] -
                                by[r - 1, c, k] +
                                by[r, c, k]
                                ) / 4.0
                    u[r, c, k] = unew

                    # update rms error
                    rmse += (unew - uprev) ** 2

                    bxx = bx[r, c, k]
                    byy = by[r, c, k]

                    # d_subproblem
                    s = ux + bxx
                    if s > 1 / lam:
                        dxx = s - 1 / lam
                    elif s < -1 / lam:
                        dxx = s + 1 / lam
                    else:
                        dxx = 0
                    s = uy + byy
                    if s > 1 / lam:
                        dyy = s - 1 / lam
                    elif s < -1 / lam:
                        dyy = s + 1 / lam
                    else:
                        dyy = 0

                    dx[r, c, k] = dxx
                    dy[r, c, k] = dyy

                    bx[r, c, k] += ux - dxx
                    by[r, c, k] += uy - dyy

        rmse = np.sqrt(rmse / total)
        i += 1

    return np.squeeze(np.asarray(u[1:-1, 1:-1]))
def defend_TotalVarience(input_array, keep_prob=0.5, lambda_tv=0.03):
    mask = np.random.uniform(size=input_array.shape[:2])
    mask = mask < keep_prob
    return bregman(input_array, mask, weight=2.0 / lambda_tv)


def make_defend_quilt(sess):
    # setup for quilting
    quilt_db = np.load('data/quilt_db.npy')
    quilt_db_reshaped = quilt_db.reshape(1000000, -1)
    TILE_SIZE = 5
    TILE_OVERLAP = 2
    tile_skip = TILE_SIZE - TILE_OVERLAP
    K = 10
    db_tensor = tf.placeholder(tf.float32, quilt_db_reshaped.shape)
    query_imgs = tf.placeholder(tf.float32, (TILE_SIZE * TILE_SIZE * 3, None))
    norms = tf.reduce_sum(tf.square(db_tensor), axis=1)[:, tf.newaxis] \
            - 2 * tf.matmul(db_tensor, query_imgs)
    _, topk_indices = tf.nn.top_k(-tf.transpose(norms), k=K, sorted=False)

    def min_error_table(arr, direction):
        assert direction in ('horizontal', 'vertical')
        y, x = arr.shape
        cum = np.zeros_like(arr)
        if direction == 'horizontal':
            cum[:, -1] = arr[:, -1]
            for ix in range(x - 2, -1, -1):
                for iy in range(y):
                    m = arr[iy, ix + 1]
                    if iy > 0:
                        m = min(m, arr[iy - 1, ix + 1])
                    if iy < y - 1:
                        m = min(m, arr[iy + 1, ix + 1])
                    cum[iy, ix] = arr[iy, ix] + m
        elif direction == 'vertical':
            cum[-1, :] = arr[-1, :]
            for iy in range(y - 2, -1, -1):
                for ix in range(x):
                    m = arr[iy + 1, ix]
                    if ix > 0:
                        m = min(m, arr[iy + 1, ix - 1])
                    if ix < x - 1:
                        m = min(m, arr[iy + 1, ix + 1])
                    cum[iy, ix] = arr[iy, ix] + m
        return cum

    def index_exists(arr, index):
        if arr.ndim != len(index):
            return False
        return all(i > 0 for i in index) and all(index[i] < arr.shape[i] for i in range(arr.ndim))

    def assign_block(ix, iy, tile, synth):
        posx = tile_skip * ix
        posy = tile_skip * iy

        if ix == 0 and iy == 0:
            synth[posy:posy + TILE_SIZE, posx:posx + TILE_SIZE, :] = tile
        elif iy == 0:
            # first row, only have horizontal overlap of the block
            tile_left = tile[:, :TILE_OVERLAP, :]
            synth_right = synth[:TILE_SIZE, posx:posx + TILE_OVERLAP, :]
            errors = np.sum(np.square(tile_left - synth_right), axis=2)
            table = min_error_table(errors, direction='vertical')
            # copy row by row into synth
            xoff = np.argmin(table[0, :])
            synth[posy, posx + xoff:posx + TILE_SIZE] = tile[0, xoff:]
            for yoff in range(1, TILE_SIZE):
                # explore nearby xoffs
                candidates = [(yoff, xoff), (yoff, xoff - 1), (yoff, xoff + 1)]
                index = min((i for i in candidates if index_exists(table, i)), key=lambda i: table[i])
                xoff = index[1]
                synth[posy + yoff, posx + xoff:posx + TILE_SIZE] = tile[yoff, xoff:]
        elif ix == 0:
            # first column, only have vertical overlap of the block
            tile_up = tile[:TILE_OVERLAP, :, :]
            synth_bottom = synth[posy:posy + TILE_OVERLAP, :TILE_SIZE, :]
            errors = np.sum(np.square(tile_up - synth_bottom), axis=2)
            table = min_error_table(errors, direction='horizontal')
            # copy column by column into synth
            yoff = np.argmin(table[:, 0])
            synth[posy + yoff:posy + TILE_SIZE, posx] = tile[yoff:, 0]
            for xoff in range(1, TILE_SIZE):
                # explore nearby yoffs
                candidates = [(yoff, xoff), (yoff - 1, xoff), (yoff + 1, xoff)]
                index = min((i for i in candidates if index_exists(table, i)), key=lambda i: table[i])
                yoff = index[0]
                synth[posy + yoff:posy + TILE_SIZE, posx + xoff] = tile[yoff:, xoff]
        else:
            # glue cuts along diagonal
            tile_up = tile[:TILE_OVERLAP, :, :]
            synth_bottom = synth[posy:posy + TILE_OVERLAP, :TILE_SIZE, :]
            errors_up = np.sum(np.square(tile_up - synth_bottom), axis=2)
            table_up = min_error_table(errors_up, direction='horizontal')
            tile_left = tile[:, :TILE_OVERLAP, :]
            synth_right = synth[:TILE_SIZE, posx:posx + TILE_OVERLAP, :]
            errors_left = np.sum(np.square(tile_left - synth_right), axis=2)
            table_left = min_error_table(errors_left, direction='vertical')
            glue_index = -1
            glue_value = np.inf
            for i in range(TILE_OVERLAP):
                e = table_up[i, i] + table_left[i, i]
                if e < glue_value:
                    glue_value = e
                    glue_index = i
            # copy left part first, up to the overlap column
            xoff = glue_index
            synth[posy + glue_index, posx + xoff:posx + TILE_OVERLAP] = tile[glue_index, xoff:TILE_OVERLAP]
            for yoff in range(glue_index + 1, TILE_SIZE):
                # explore nearby xoffs
                candidates = [(yoff, xoff), (yoff, xoff - 1), (yoff, xoff + 1)]
                index = min((i for i in candidates if index_exists(table_left, i)), key=lambda i: table_left[i])
                xoff = index[1]
                synth[posy + yoff, posx + xoff:posx + TILE_OVERLAP] = tile[yoff, xoff:TILE_OVERLAP]
            # copy right part, down to overlap row
            yoff = glue_index
            synth[posy + yoff:posy + TILE_OVERLAP, posx + glue_index] = tile[yoff:TILE_OVERLAP, glue_index]
            for xoff in range(glue_index + 1, TILE_SIZE):
                # explore nearby yoffs
                candidates = [(yoff, xoff), (yoff - 1, xoff), (yoff + 1, xoff)]
                index = min((i for i in candidates if index_exists(table_up, i)), key=lambda i: table_up[i])
                yoff = index[0]
                synth[posy + yoff:posy + TILE_OVERLAP, posx + xoff] = tile[yoff:TILE_OVERLAP, xoff]
            # copy rest of image
            synth[posy + TILE_OVERLAP:posy + TILE_SIZE, posx + TILE_OVERLAP:posx + TILE_SIZE] = tile[TILE_OVERLAP:,
                                                                                                TILE_OVERLAP:]

    KNN_MAX_BATCH = 1000

    def quilt(arr, graphcut=True):
        h, w, c = arr.shape
        assert (h - TILE_SIZE) % tile_skip == 0
        assert (w - TILE_SIZE) % tile_skip == 0
        horiz_blocks = (w - TILE_SIZE) // tile_skip + 1
        vert_blocks = (h - TILE_SIZE) // tile_skip + 1
        num_patches = horiz_blocks * vert_blocks
        patches = np.zeros((TILE_SIZE * TILE_SIZE * 3, num_patches))
        idx = 0
        for iy in range(vert_blocks):
            for ix in range(horiz_blocks):
                posx = tile_skip * ix
                posy = tile_skip * iy
                patches[:, idx] = arr[posy:posy + TILE_SIZE, posx:posx + TILE_SIZE, :].ravel()
                idx += 1

        ind = []
        for chunk in range(num_patches // KNN_MAX_BATCH + (1 if num_patches % KNN_MAX_BATCH != 0 else 0)):
            start = KNN_MAX_BATCH * chunk
            end = start + KNN_MAX_BATCH
            # for some reason, the code below is 10x slower when run in a Jupyter notebook
            # not sure why...
            indices_ = sess.run(topk_indices, {db_tensor: quilt_db_reshaped, query_imgs: patches[:, start:end]})
            for i in indices_:
                ind.append(np.random.choice(i))

        synth = np.zeros((299, 299, 3))

        idx = 0
        for iy in range(vert_blocks):
            for ix in range(horiz_blocks):
                posx = tile_skip * ix
                posy = tile_skip * iy
                tile = quilt_db[ind[idx]]
                if not graphcut:
                    synth[posy:posy + TILE_SIZE, posx:posx + TILE_SIZE, :] = tile
                else:
                    assign_block(ix, iy, tile, synth)
                idx += 1
        return synth

    return quilt


def nearest_neighbour_scaling(label, new_h, new_w, patch_size, patch_n, patch_m):
    if len(label.shape) == 2:
        label_new = np.zeros([new_h, new_w])
    else:
        label_new = np.zeros([new_h, new_w, label.shape[2]])
    n_pos = np.arange(patch_n)
    m_pos = np.arange(patch_m)
    n_pos = n_pos.repeat(patch_size)[:299]
    m_pos = m_pos.repeat(patch_size)[:299]
    n_pos = n_pos.reshape(n_pos.shape[0], 1)
    n_pos = np.tile(n_pos, (1, new_w))
    m_pos = np.tile(m_pos, (new_h, 1))
    assert n_pos.shape == m_pos.shape
    label_new[:, :] = label[n_pos[:, :], m_pos[:, :]]
    return label_new
def jpeg(input_array, quali):
    pil_image = PIL.Image.fromarray((input_array * 255.0).astype(np.uint8))
    f = BytesIO()
    pil_image.save(f, format='jpeg', quality=quali)  # quality level specified in paper
    jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0
    return jpeg_image
def defend_SHIELD(x, qualities=(20, 40, 60, 80), patch_size=8):
    n = x.shape[0]
    m = x.shape[1]
    patch_n = n / patch_size
    patch_m = m / patch_size
    num_qualities = len(qualities)
    n = x.shape[0]
    m = x.shape[1]
    if n % patch_size > 0:
        patch_n = np.int(n / patch_size) + 1
        delete_n = 1
    if m % patch_size > 0:
        patch_m = np.int(m / patch_size) + 1
        delet_m = 1

    R = np.tile(np.reshape(np.arange(n), (n, 1)), [1, m])
    C = np.reshape(np.tile(np.arange(m), [n]), (n, m))
    mini_Z = (np.random.rand(patch_n, patch_m) * num_qualities).astype(int)
    Z = (nearest_neighbour_scaling(mini_Z, n, m, patch_size, patch_n, patch_m)).astype(int)
    indices = np.transpose(np.stack((Z, R, C)), (1, 2, 0))
    # x = img_as_ubyte(x)
    x_compressed_stack = []

    for quali in qualities:
        processed = jpeg(x, quali)
        x_compressed_stack.append(processed)

    x_compressed_stack = np.asarray(x_compressed_stack)
    x_slq = np.zeros((n, m, 3))
    for i in range(n):
        for j in range(m):
            x_slq[i, j] = x_compressed_stack[tuple(indices[i][j])]
    return x_slq

def defend_pixel_deflection(img, deflections=600, window=10):
    img = np.copy(img)
    H, W, C = img.shape
    while deflections > 0:
        #for consistency, when we deflect the given pixel from all the three channels.
        for c in range(C):
            x,y = randint(0,H-1), randint(0,W-1)
            while True: #this is to ensure that PD pixel lies inside the image
                a,b = randint(-1*window,window), randint(-1*window,window)
                if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
            # calling pixel deflection as pixel swap would be a misnomer,
            # as we can see below, it is one way copy
            img[x,y,c] = img[x+a,y+b,c]
        deflections -= 1
    return img

#####################################################################
# Defense methods used to compare the performance over the BPDA + EOT

# x is a square image (3-tensor)
def defend_crop(x, crop_size=90, ensemble_size=30):
    x_size = tf.to_float(x.shape[1])
    frac = crop_size/x_size
    start_fraction_max = (x_size - crop_size)/x_size
    def randomizing_crop(x):
        start_x = tf.random_uniform((), 0, start_fraction_max)
        start_y = tf.random_uniform((), 0, start_fraction_max)
        return tf.image.crop_and_resize([x], boxes=[[start_y, start_x, start_y+frac, start_x+frac]],
                                 box_ind=[0], crop_size=[crop_size, crop_size])

    return tf.concat([randomizing_crop(x) for _ in range(ensemble_size)], axis=0)

# for single sample
def randomizing_crop(x):
    x_size = tf.to_float(x.shape[1])
    crop_size=90
    frac = crop_size/x_size
    start_fraction_max = (x_size - crop_size)/x_size
    start_x = tf.random_uniform((), 0, start_fraction_max)
    start_y = tf.random_uniform((), 0, start_fraction_max)
    return tf.image.crop_and_resize([x], boxes=[[start_y, start_x, start_y+frac, start_x+frac]],
                             box_ind=[0], crop_size=[crop_size, crop_size])

PAD_VALUE = 0.5
def defend_randomization(input_tensor):
    rnd = tf.random_uniform((), 299, 400, dtype=tf.int32)
    rescaled = tf.image.crop_and_resize(input_tensor, [[0, 0, 1, 1]], [0], [rnd, rnd])
    h_rem = 400 - rnd
    w_rem = 400 - rnd
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=PAD_VALUE)
    padded.set_shape((1, 400, 400, 3))
    return padded

from skimage import transform
def defend_onlyrand(img):
    rnd = np.random.randint(299,400,(1,))[0]
    rescaled = transform.resize(img,(rnd,rnd))
    h_rem = 400 - rnd
    w_rem = 400 - rnd
    pad_left = np.random.randint(0,w_rem,(1,))[0]
    pad_right = w_rem - pad_left
    pad_top = np.random.randint(0,h_rem,(1,))[0]
    pad_bottom = h_rem - pad_top
    padded = np.pad(rescaled,((pad_top,pad_bottom),(pad_left,pad_right),(0,0)),'constant',constant_values = 0.5)
    padded = transform.resize(padded,(299,299))
    return padded






