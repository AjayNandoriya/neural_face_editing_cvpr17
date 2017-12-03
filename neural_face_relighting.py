from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images", required=True)
parser.add_argument("--train_file", type=str, help="path to train filenames text file", default='')
parser.add_argument("--val_file", type=str, help="path to val filenames text file", default='')
parser.add_argument("--test_file", type=str, help="path to test filenames text file", default='')
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")

parser.add_argument("--val_freq", type=int, default=100, help="run validation every val_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=100,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=32, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=64, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--mask_weight", type=float, default=1.0, help="weight on mask for generator")
parser.add_argument("--lcoeff_weight", type=float, default=1.0, help="weight on light coefficient for generator")
parser.add_argument("--normal_weight", type=float, default=1.0, help="weight on normal for generator")
parser.add_argument("--albedo_weight", type=float, default=1.0, help="weight on albedo for generator")
parser.add_argument("--shading_weight", type=float, default=1.0, help="weight on shading for generator")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id to be used")
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 64

Examples = collections.namedtuple("Examples",
                                  "paths, inputs, targets, masks, normals, lcoeffs, count, steps_per_epoch, mode")
Model = collections.namedtuple("Model",
                               "outputs, masks, normals, lcoeffs, shadings, albedos, img_bg, img_fg, \
                               predict_real, predict_fake, \
                               discrim_loss, discrim_grads_and_vars, \
                               gen_loss_GAN, gen_loss_L1, gen_lcoeff_loss, gen_mask_loss_L1, gen_normal_loss, gen_albedo_loss_smoothness, gen_shading_loss_smoothness, \
                               gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels],
                                      [1, 2, 2, 1], padding="SAME")
        return conv

def gradient_x( img ):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx

def gradient_y( img ):
    gy = img[:, :-1, :, :] - img[:, 1:, :, :]
    return gy


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((
                                                                 srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                                                                                                  xyz_normalized_pixels ** (
                                                                                                  1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (
                                                                                       fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (
            1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


# ~ def read_image(image_path):
# ~ path_length = string_length_tf(image_path)[0]
# ~ file_extension = tf.substr(image_path, path_length - 3, 3)
# ~ file_cond = tf.equal(file_extension, 'jpg')

# ~ print(file_cond)
# ~ image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))
# ~ image  = tf.image.convert_image_dtype(image,  tf.float32)
# ~ return

def read_image(image_path):
    # ~ path_length = tf.size(image_path)
    # ~ file_extension = tf.substr(image_path, path_length - 3, 3)
    # ~ file_cond = tf.equal(file_extension, 'jpg')
    path_split = tf.string_split([image_path], ".").values
    N = tf.size(path_split)
    ext = path_split[N - 1]
    file_cond = tf.equal(ext, 'jpg')
    # ~ file_cond = tf.Print(file_cond, [file_cond,image_path,N], message="This is data: ")
    image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                    lambda: tf.image.decode_png(tf.read_file(image_path)))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def load_examples(mode='train'):  # "val", "test"
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for up-scaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    print("load_examples begin.")
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist :" + a.input_dir)
    if mode == "train" and not os.path.isfile(a.train_file):
        raise Exception("train_file does not exist:" + a.train_file)
    if mode == "val" and not os.path.isfile(a.val_file):
        raise Exception("val_file does not exist:" + a.val_file)
    if mode == "test" and not os.path.isfile(a.test_file):
        raise Exception("test_file does not exist:" + a.test_file)
    if mode == "train":
        from_file = a.train_file
    elif mode == "val":
        from_file = a.val_file
    elif mode == "test":
        from_file = a.test_file
    else:
        raise Exception("Mode has to be one of the following: a) train, b) val, c) test")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    # Read image list from text file as (data_path + "input_name output_name")
    input_queue = tf.train.string_input_producer([from_file], shuffle=mode=='train')
    line_reader = tf.TextLineReader()
    _, line = line_reader.read(input_queue)
    # line is of format('input-path','output-path','mask-path','normals-path',27 float lightening coefficients)

    split_line = tf.string_split([line]).values
    a_image_path = tf.string_join([a.input_dir, split_line[0]])
    # ~ a_image_path  = tf.Print(a_image_path,[a_image_path],"a_image_path:")
    paths = a_image_path
    # ~ paths = tf.Print(paths,[paths],message="paths :")
    f = open(from_file, 'r')
    input_paths = f.readlines()
    f.close()
    a_image_o = read_image(a_image_path)

    a_assertion = tf.assert_equal(tf.shape(a_image_o)[2], 3, data=[a_image_o, a_image_path],
                                  message="image-a does not have 3 channels")
    with tf.control_dependencies([a_assertion]):
        a_image_o = tf.identity(a_image_o)

    a_image_o.set_shape([None, None, 3])
    a_images = preprocess(a_image_o)

    # if a.mode != 'test':
    b_image_path = tf.string_join([a.input_dir, split_line[1]])
    b_image_o = read_image(b_image_path)
    b_assertion = tf.assert_equal(tf.shape(b_image_o)[2], 3, message="image-b does not have 3 channels")
    with tf.control_dependencies([b_assertion]):
        b_image_o = tf.identity(b_image_o)

    b_image_o.set_shape([None, None, 3])
    b_images = preprocess(b_image_o)

    # Mask Image
    mask_image_path = tf.string_join([a.input_dir, split_line[2]])
    mask_image_o = read_image(mask_image_path)
    mask_assertion = tf.assert_equal(tf.shape(mask_image_o)[2], 1, message="image-mask does not have 1 channels")
    with tf.control_dependencies([mask_assertion]):
        mask_image_o = tf.identity(mask_image_o)

    mask_image_o.set_shape([None, None, 1])
    mask_images = transform(mask_image_o)

    # Normal Image
    normal_image_path = tf.string_join([a.input_dir, split_line[3]])
    normal_image_o = read_image(normal_image_path)
    normal_assertion = tf.assert_equal(tf.shape(normal_image_o)[2], 3, message="image-normal does not have 3 channels")
    with tf.control_dependencies([normal_assertion]):
        normal_image_o = tf.identity(normal_image_o)

    normal_image_o.set_shape([None, None, 3])
    normal_images = transform(preprocess(normal_image_o))

    c = []
    for k in range(27):
        c.append(tf.string_to_number(split_line[k + 4], out_type=tf.float32))
    l_coeff = tf.reshape(tf.gather(c, range(27)), [1, 1, -1])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")





    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch, masks_batch, normals_batch, lcoeffs_batch = tf.train.batch(
        [paths, input_images, target_images, mask_images, normal_images, l_coeff], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        masks=masks_batch,
        normals=normals_batch,
        lcoeffs=lcoeffs_batch,
        count=tf.constant(len(input_paths),dtype=tf.int64),
        steps_per_epoch=tf.constant(steps_per_epoch,dtype=tf.int64),
        mode=tf.constant(mode,dtype=tf.string),
    )


def decoder_stack(input_layer):
    layers = [input_layer]
    layer_specs = [a.ngf * 2, a.ngf * 2, a.ngf * 2, a.ngf * 2, a.ngf * 2, a.ngf]
    for filtersize in layer_specs:
        with tf.variable_scope("decoder_%d" % (len(layers))):
            # s = tf.shape(layers[-1])
            s = layers[-1].get_shape().as_list()
            # h = tf.maximum(s[1] * 2, tf.constant(3,dtype=s[1].dtype))
            # w = tf.maximum(s[2] * 2, tf.constant(3,dtype=s[2].dtype))
            upsampled = tf.image.resize_nearest_neighbor(layers[-1], [2*s[1], 2*s[2]])
            conv1 = tf.layers.conv2d(inputs=upsampled, filters=filtersize, kernel_size=3, padding='same',
                                     activation=tf.nn.relu)
            print("conv1_%d shape=", len(layers),layers[-1].get_shape().as_list())
            layers.append(conv1)
    return layers[-1]


def decoder_stack_with_skip(input_layer, encoder_layers):
    layers = [input_layer]
    layersN = len(encoder_layers)-1
    with tf.variable_scope("upsample_%d" % (layersN + 1)):
        shape1 = encoder_layers[-2].get_shape().as_list()
        upsampled1 = tf.image.resize_nearest_neighbor(layers[-1], [shape1[1], shape1[2]])
        conv1 = tf.layers.conv2d(inputs=upsampled1, filters=shape1[3], kernel_size=3, padding='same',
                                 activation=tf.nn.relu)
        layers.append(conv1)

    for k in range(layersN - 1, 0, -1):
        with tf.variable_scope("decoder_%d" % (k)):
            skip_layer = encoder_layers[k]
            concat1 = tf.concat([layers[-1], skip_layer], axis=3)
            shape1 = encoder_layers[k-1].get_shape().as_list()
            upsampled1 = tf.image.resize_nearest_neighbor(concat1, [shape1[1], shape1[2]])
            conv1 = tf.layers.conv2d(inputs=upsampled1, filters=shape1[3], kernel_size=3, padding='same',
                                     activation=tf.nn.relu)
            layers.append(conv1)
    return layers[-1]


def create_generator(generator_inputs):
    layers = [generator_inputs]

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        conv1 = tf.layers.conv2d(inputs=generator_inputs, filters=a.ngf, kernel_size=3, padding='same',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        layers.append(pool1)
    with tf.variable_scope("encoder_2"):
        conv2 = tf.layers.conv2d(inputs=layers[-1], filters=a.ngf * 2, kernel_size=3, padding='same',
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        layers.append(pool2)
    with tf.variable_scope("encoder_3"):
        conv3 = tf.layers.conv2d(inputs=layers[-1], filters=a.ngf * 2, kernel_size=3, padding='same',
                                 activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        layers.append(pool3)
    with tf.variable_scope("flatten"):
        shape = layers[-1].get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        fullyconnected = tf.reshape(layers[-1], [-1, 1, 1, dim])
        layers.append(fullyconnected)

    with tf.variable_scope("encoder_background"):
        z_bg = tf.layers.conv2d(inputs=layers[-1], filters=a.ngf * 4, kernel_size=1, padding='same',
                                activation=tf.nn.relu)
    with tf.variable_scope("encoder_mask"):
        z_mask = tf.layers.conv2d(inputs=layers[-1], filters=a.ngf * 4, kernel_size=1, padding='same',
                                  activation=tf.nn.relu)
    with tf.variable_scope("encoder_normal"):
        z_normal = tf.layers.conv2d(inputs=layers[-1], filters=a.ngf * 4, kernel_size=1, padding='same',
                                    activation=tf.nn.relu)
    with tf.variable_scope("encoder_light"):
        z_light = 5*tf.layers.conv2d(inputs=layers[-1], filters=27, kernel_size=1, padding='same', activation=tf.nn.tanh)
    with tf.variable_scope("encoder_albedo"):
        z_albedo = tf.layers.conv2d(inputs=layers[-1], filters=a.ngf * 4, kernel_size=1, padding='same',
                                    activation=tf.nn.relu)
    with tf.variable_scope("decoder_stack_bg"):
        feature_bg = decoder_stack_with_skip(z_bg, encoder_layers=layers)
        I_bg = tf.layers.conv2d(inputs=feature_bg, filters=3, kernel_size=3, padding='same', activation=tf.nn.tanh)
    with tf.variable_scope("decoder_stack_mask"):
        feature_mask = decoder_stack_with_skip(z_mask, encoder_layers=layers)
        mask_single = tf.layers.conv2d(inputs=feature_mask, filters=1, kernel_size=3, padding='same',
                                       activation=tf.nn.sigmoid)
        mask = tf.tile(mask_single, [1, 1, 1, 3])
    with tf.variable_scope("decoder_stack_normal"):
        feature_normal = decoder_stack(z_normal)
        conv1 = tf.layers.conv2d(inputs=feature_normal, filters=3, kernel_size=3, padding='same', activation=tf.nn.tanh)
        normals = tf.nn.l2_normalize(5*conv1, dim=3, name='surfaceNormal')
        print("z_normals shape=", z_normal.get_shape().as_list())
        print("normals shape=", feature_normal.get_shape().as_list())

    with tf.variable_scope("decoder_stack_albedo"):
        feature_albedo = decoder_stack(z_albedo)
        albedo = 5*tf.layers.conv2d(inputs=feature_albedo, filters=3, kernel_size=3, padding='same',
                                  activation=tf.nn.sigmoid)

    with tf.variable_scope("shader"):
        C = tf.constant([0.429043, 0.511664, 0.743125, 0.886227, 0.247708], dtype='float32')
        Nshape = normals.get_shape().as_list()
        print("Nshape=",Nshape)
        one_tensor = tf.ones([1,1,1,1],dtype=normals.dtype)
        one_tensor = tf.tile(one_tensor,[Nshape[0], Nshape[1], Nshape[2],1])
        N = tf.reshape(tf.concat([normals,one_tensor], axis=3),
                       [Nshape[0], Nshape[1], Nshape[2], 1, 1, 4])
        N = tf.tile(N,[1, 1, 1, 3, 1, 1])
        Nshape = N.get_shape().as_list()
        Nt = tf.transpose(N, perm=[0,1,2,3,5,4])
        L_flat = tf.reshape(z_light, [-1, 27])
        # Kmat = tf.gather(
        #     [[C[0] * L_flat[:, 8 + 9 * x] for x in range(3)], [C[0] * L_flat[:, 4 + 9 * x] for x in range(3)],
        #      [C[0] * L_flat[:, 7 + 9 * x] for x in range(3)], [C[1] * L_flat[:, 3 + 9 * x] for x in range(3)],
        #      [C[0] * L_flat[:, 4 + 9 * x] for x in range(3)], [-C[0] * L_flat[:, 8 + 9 * x] for x in range(3)],
        #      [C[0] * L_flat[:, 5 + 9 * x] for x in range(3)], [C[1] * L_flat[:, 1 + 9 * x] for x in range(3)],
        #      [C[0] * L_flat[:, 7 + 9 * x] for x in range(3)], [C[0] * L_flat[:, 5 + 9 * x] for x in range(3)],
        #      [C[2] * L_flat[:, 6 + 9 * x] for x in range(3)], [C[1] * L_flat[:, 2 + 9 * x] for x in range(3)],
        #      [C[1] * L_flat[:, 3 + 9 * x] for x in range(3)], [C[1] * L_flat[:, 1 + 9 * x] for x in range(3)],
        #      [C[1] * L_flat[:, 2 + 9 * x] for x in range(3)],
        #      [(C[3] * L_flat[:, 0 + 9 * x] - C[4] * L_flat[:, 6]) for x in range(3)]],
        #     indices=np.arange(16))
        Kmat = tf.gather(
            [[C[0] * L_flat[:, 8 * 3 + x] for x in range(3)], [C[0] * L_flat[:, 4 * 3 + x] for x in range(3)],
             [C[0] * L_flat[:, 7 * 3 + x] for x in range(3)], [C[1] * L_flat[:, 3 * 3 + x] for x in range(3)],
             [C[0] * L_flat[:, 4 * 3 + x] for x in range(3)], [-C[0] * L_flat[:, 8 * 3 + x] for x in range(3)],
             [C[0] * L_flat[:, 5 * 3 + x] for x in range(3)], [C[1] * L_flat[:, 1 * 3 + x] for x in range(3)],
             [C[0] * L_flat[:, 7 * 3 + x] for x in range(3)], [C[0] * L_flat[:, 5 * 3 + x] for x in range(3)],
             [C[2] * L_flat[:, 6 * 3 + x] for x in range(3)], [C[1] * L_flat[:, 2 * 3 + x] for x in range(3)],
             [C[1] * L_flat[:, 3 * 3 + x] for x in range(3)], [C[1] * L_flat[:, 1 * 3 + x] for x in range(3)],
             [C[1] * L_flat[:, 2 * 3 + x] for x in range(3)],
             [(C[3] * L_flat[:, 0 * 3 + x] - C[4] * L_flat[:, 6]) for x in range(3)]],
            indices=np.arange(16))
        K = tf.reshape(tf.transpose(Kmat, perm=[2, 1, 0]), [Nshape[0], 1, 1, 3, 4, 4])
        K = tf.tile(K, [1, Nshape[1], Nshape[2], 1, 1, 1])
        NK = tf.matmul(N, K)
        S = tf.squeeze(tf.matmul(NK, Nt), axis=[4, 5])
    with tf.variable_scope("renderer"):
        I_fg = (albedo * S)*2-1
    with tf.variable_scope("matt_layer"):
        reconstructed = I_fg * mask + (1 - mask) * I_bg

    model = dict()
    model['reconstructed'] = reconstructed
    model['albedo'] = albedo
    model['normals'] = normals
    model['mask'] = mask
    model['I_bg'] = I_bg
    model['I_fg'] = I_fg
    model['L'] = L_flat
    model['shading'] = S
    model['img_bg'] = I_bg
    model['img_fg'] = I_fg
    return model


def create_model( input_data):
    def create_discriminator( discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                layer_out_channels = a.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], layer_out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    # Separate input data variables
    inputs = input_data.inputs
    targets = input_data.targets
    normals = input_data.normals
    masks = input_data.masks
    lcoeffs = input_data.lcoeffs

    with tf.variable_scope("generator"):
        gen_outputs = create_generator(inputs)
        outputs = gen_outputs['reconstructed']

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs((targets - gen_outputs['I_fg'])*masks)) + tf.reduce_mean(tf.abs((targets - outputs)))
        gen_mask_loss_L1 = tf.reduce_mean(tf.square(tf.abs(masks - gen_outputs['mask'])))
        gen_normal_loss = tf.reduce_mean(tf.square(tf.abs(normals - gen_outputs['normals'])))
        gen_lcoeff_loss = tf.reduce_mean(tf.square(tf.abs(lcoeffs - gen_outputs['L'])))
        gen_bg_loss = tf.reduce_mean(tf.square(tf.abs(gen_outputs['mask']*gen_outputs['I_bg'])))

        # albedo
        gen_albedo_loss_smoothness = tf.reduce_mean(tf.abs(gradient_x(gen_outputs['albedo']))) + tf.reduce_mean(tf.abs(gradient_y(gen_outputs['albedo'])))


        # shading
        gen_shading_loss_smoothness = tf.reduce_mean(tf.square(tf.abs(gradient_x(gen_outputs['shading'])))) + tf.reduce_mean(tf.square(tf.abs(gradient_y(gen_outputs['shading']))))


        # total loss
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight + gen_mask_loss_L1*a.mask_weight + gen_lcoeff_loss*a.lcoeff_weight + gen_albedo_loss_smoothness*a.albedo_weight + gen_shading_loss_smoothness*a.shading_weight + gen_normal_loss*a.normal_weight + 100*gen_bg_loss

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
            print("gen_tvars list:")
            for grad,var in gen_grads_and_vars:
                print(var.op.name)


    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1,gen_mask_loss_L1,gen_normal_loss,gen_lcoeff_loss,gen_albedo_loss_smoothness,gen_shading_loss_smoothness])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)


    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_mask_loss_L1=ema.average(gen_mask_loss_L1),
        gen_normal_loss=ema.average(gen_normal_loss),
        gen_lcoeff_loss=ema.average(gen_lcoeff_loss),
        gen_albedo_loss_smoothness=ema.average(gen_albedo_loss_smoothness),
        gen_shading_loss_smoothness=ema.average(gen_shading_loss_smoothness),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        masks=gen_outputs['mask'],
        normals=gen_outputs['normals'],
        shadings=gen_outputs['shading'],
        albedos=gen_outputs['albedo'],
        lcoeffs=gen_outputs['L'],
        img_bg=gen_outputs['img_bg'],
        img_fg=gen_outputs['img_fg'],
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None, mode="train"):
    image_dir = os.path.join(a.output_dir, mode, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    print(image_dir)
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False, mode="train"):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='%s/images/%s'></td>" % (mode, fileset[kind]))

        index.write("</tr>")
    return index_path




# def fill_dict(sample_examples):
#     return {
#     paths_ph : sample_examples.paths,
#     inputs_ph : sample_examples.input,
#     targets_ph : sample_examples.targets,
#     masks_ph : sample_examples.masks,
#     normals_ph : sample_examples.normals_ph,
#     lcoeffs_ph : sample_examples.lcoeffs_ph,
#     count_ph : sample_examples.count_ph,
#     steps_per_epoch_ph : sample_examples.steps_per_epoch,
#     mode_ph : sample_examples.mode    }

def placeholder_input():
    input_ph = tf.placeholder(tf.float32, shape=(a.batch_size, CROP_SIZE, CROP_SIZE, 3))
    target_ph = tf.placeholder(tf.float32, shape=(a.batch_size, CROP_SIZE, CROP_SIZE, 3))
    paths_ph = tf.placeholder(tf.string, shape=(a.batch_size))
    mode_ph = tf.placeholder(tf.string, shape=None)
    normals_ph = tf.placeholder(tf.float32, shape=(a.batch_size, CROP_SIZE, CROP_SIZE, 3))
    masks_ph = tf.placeholder(tf.float32, shape=(a.batch_size, CROP_SIZE, CROP_SIZE, 1))
    lcoeffs_ph = tf.placeholder(tf.float32, shape=(a.batch_size, 1, 1, 27))
    count_ph = tf.placeholder(tf.int32, shape=None)
    steps_per_epoch_ph = tf.placeholder(tf.int32, shape=None)
    return Examples(
        paths=paths_ph,
        inputs=input_ph,
        targets=target_ph,
        masks=masks_ph,
        normals=normals_ph,
        lcoeffs=lcoeffs_ph,
        count=count_ph,
        steps_per_epoch=steps_per_epoch_ph,
        mode=mode_ph,
    )


def placeholder_display():
    input_ph = tf.placeholder(dtype=tf.uint8, shape=(1, CROP_SIZE, CROP_SIZE, 3));
    output_ph = tf.placeholder(dtype=tf.uint8, shape=(1, CROP_SIZE, CROP_SIZE, 3));
    target_ph = tf.placeholder(dtype=tf.uint8, shape=(1, CROP_SIZE, CROP_SIZE, 3));

    return input_ph, output_ph, target_ph;


def main():
    if tf.__version__ != "1.0.1" and tf.__version__ != "1.0.0":
        raise Exception("Tensorflow version 1.0.1 required but " + tf.__version__ + " found")

    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))




    print(a.test_file,a.train_file,a.val_file)
    if a.test_file is not '':
        test_examples = load_examples("test")
    if a.val_file is not '':
        train_examples = load_examples("train")
    if a.train_file is not '':
        val_examples = load_examples("val")

    # with tf.Graph().as_default():

    # Create placeholders for input data
    batch_input = placeholder_input()
    model = create_model(batch_input)

    # undo colorization splitting on images that we use for display/output
    inputs = deprocess(batch_input.inputs)
    targets = deprocess(batch_input.targets)
    outputs = deprocess(model.outputs)
    masks = model.masks
    normals = deprocess(model.normals)
    shadings = deprocess(model.shadings)
    albedos = deprocess(model.albedos)
    img_bg = deprocess(model.img_bg)
    img_fg = deprocess(model.img_fg)
    normals_gt = deprocess(batch_input.normals)
    masks_gt = batch_input.masks

    def convert( image,image_str ):
        with tf.name_scope("convert_%s" % image_str):
            if a.aspect_ratio != 1.0:
                # upscale to correct aspect ratio
                size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
                image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user

    converted_inputs = convert(inputs,'inputs')
    converted_targets = convert(targets,'targets')
    converted_outputs = convert(outputs,'outputs')
    converted_masks = convert(masks,'masks')
    converted_normals = convert(normals,'normals')
    converted_shadings = convert(shadings,'shadings')
    converted_albedos = convert(albedos, 'albedos')
    converted_img_bg = convert(img_bg, 'img_bg')
    converted_img_fg = convert(img_fg, 'img_fg')
    converted_normals_gt = convert(normals_gt, 'normals_gt')
    converted_masks_gt = convert(masks_gt, 'masks_gt')

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": batch_input.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    # train_converted_inputs, train_converted_outputs, train_converted_targets = placeholder_display();


    # summery images
    tf.summary.image("inputs", converted_inputs)
    tf.summary.image("targets", converted_targets)
    tf.summary.image("outputs", converted_outputs)
    tf.summary.image("masks", converted_masks)
    tf.summary.image("normals", converted_normals)
    tf.summary.image("shadings", converted_shadings)
    tf.summary.image("albedos", converted_albedos)
    tf.summary.image("img_bg", converted_img_bg)
    tf.summary.image("img_fg", converted_img_fg)
    tf.summary.image("normals_gt", converted_normals_gt)
    tf.summary.image("masks_gt", converted_masks_gt)

    tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
    tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    # summery scalars
    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_lcoeff_loss", model.gen_lcoeff_loss)
    tf.summary.scalar("generator_mask_loss", model.gen_mask_loss_L1)
    tf.summary.scalar("generator_normal_loss", model.gen_normal_loss)
    tf.summary.scalar("generator_shading_loss", model.gen_shading_loss_smoothness)
    tf.summary.scalar("generator_albedo_loss", model.gen_albedo_loss_smoothness)

    # summery histograms
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars:
        print(var.op.name)
        if(grad is None or var.op.name is None):
            continue
        tf.summary.histogram(var.op.name + "/gradients", grad)
    for grad, var in model.gen_grads_and_vars:
        print(var.op.name)
        if (grad is None or var.op.name is None):
            continue
        tf.summary.histogram(var.op.name + "/gradients", grad)

    # parameter count
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1,var_list={v.op.name: v for v in tf.trainable_variables()})

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))
        train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(logdir + '/val', )

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(test_examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                batch_sample = sess.run(test_examples)
                results = sess.run(display_fetches, feed_dict={batch_input: batch_sample})
                filesets = save_images(results, step, mode="test")
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets, step, mode="test")

            print("wrote index at", index_path)
        else:
            steps_per_epoch = sess.run(train_examples.steps_per_epoch)

            if a.max_epochs is not None:
                max_steps = steps_per_epoch * a.max_epochs
            if a.max_steps is not None:
                max_steps = a.max_steps
            # training
            start = time.time()

            for step in range(max_steps):
                # mode_model = "train";
                def should( freq ):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq or a.val_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    fetches["gen_mask_loss_L1"] = model.gen_mask_loss_L1
                    fetches["gen_normal_loss"] = model.gen_normal_loss
                    fetches["gen_lcoeff_loss"] = model.gen_lcoeff_loss
                    fetches["gen_albedo_loss"] = model.gen_albedo_loss_smoothness
                    fetches["gen_shading_loss"] = model.gen_shading_loss_smoothness


                if should(a.summary_freq or a.val_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq or a.val_freq):
                    fetches["display"] = display_fetches

                batch_sample = sess.run(train_examples)

                results = sess.run(fetches, options=options, run_metadata=run_metadata,
                                   feed_dict={batch_input: batch_sample})
                # print("global_step = ",results["global_step"])

                if should(a.summary_freq):
                    print("recording summary")
                    # sv.summary_writer.add_summary(results["summary"], results["global_step"])
                    train_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"], mode='train')
                    append_index(filesets, step=True, mode="train")
                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / steps_per_epoch)
                    train_step = (results["global_step"] - 1) % steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                    for key_name in results.keys():
                        if key_name.find('loss') != -1:
                            print(key_name, results[key_name])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)
                if should(a.val_freq):
                    del fetches['train']
                    # for trials in range(10):
                    batch_sample = sess.run(val_examples)
                    val_results = sess.run(fetches, feed_dict={batch_input: batch_sample})
                    val_writer.add_summary(val_results["summary"], val_results["global_step"])
                    filesets = save_images(val_results["display"], step=val_results["global_step"], mode="val")
                    append_index(filesets, step=True, mode="val")

                    # print all losses from val_results
                    for key_name in val_results.keys():
                        if key_name.find('loss') != -1:
                            print(key_name, val_results[key_name])

                if sv.should_stop():
                    break


main()
