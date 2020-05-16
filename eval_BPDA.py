import os
import time
import tqdm
import argparse
import numpy as np
import tensorflow as tf

import utils
import defense
import inceptionv3

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def defend_BPDA(data, defense_func, targets, max_steps=1000, lam=1e-6, epsilon=0.05, lr=0.1):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    input_shape = data.shape[1:]
    xs = tf.placeholder(tf.float32, input_shape)
    l2_x = tf.placeholder(tf.float32, input_shape)
    l2_orig = tf.placeholder(tf.float32, input_shape)
    label = tf.placeholder(tf.int32, ())
    one_hot = tf.expand_dims(tf.one_hot(label, 1000), axis=0)

    logits, preds = inceptionv3.model(sess, tf.expand_dims(xs, axis=0))
    l2_loss = tf.sqrt(2 * tf.nn.l2_loss(l2_x - l2_orig) / np.product(input_shape))

    labels = tf.tile(one_hot, (logits.shape[0], 1))
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    loss = xent + lam * tf.maximum(l2_loss - epsilon, 0)
    grad, = tf.gradients(loss, xs)

    adv = np.copy(data)
    for index in range(data.shape[0]):

        adv_bpda = np.copy(adv[index])

        for i in tqdm.tqdm(range(max_steps)):
            adv_def = defense.defend_FD_sig(adv_bpda)
            adv_def = defense.defended(defense_func, np.expand_dims(adv_def, axis=0))
            p, l2 = sess.run([preds, l2_loss], {xs: adv_def[0], l2_x: adv_bpda, l2_orig: data[index]})
            if p == targets[index] and l2 < epsilon:
                print("Found AE. Iter: {}. L2: {}.".format(i, l2))
                break
            elif l2 > epsilon:
                print("Can't find AE under l2-norm 0.05.")
                break

            g = sess.run(grad, {xs: adv_def[0], label: targets[index]})
            adv_bpda -= lr * g
            adv_bpda = np.clip(adv_bpda, 0, 1)

        adv[index] = adv_bpda

    return adv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--output-path', type=str, default='./outputs')
    parser.add_argument('--output-name', type=str, default='adv_bpda.npy')
    parser.add_argument('--defense', type=str, default='GD')

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    data = np.load(os.path.join(args.data_path, "clean100data.npy"))
    labels = np.load(os.path.join(args.data_path, "clean100label.npy"))
    targets = np.load(os.path.join(args.data_path, "random_targets.npy"))

    adv = defend_BPDA(data, args.defense, targets)

    np.save(os.path.join(args.output_path, args.output_name), adv)


if __name__ == '__main__':
    main()