import os
import time
import tqdm
import argparse
import numpy as np
import tensorflow as tf

import utils
import defense
import inceptionv3

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def defend_BPDA_EOT(data, defense_func, targets, max_steps=1000, lam=1e-6, epsilon=0.05, ens_size=30, lr=0.1):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    input_shape = data.shape[1:]
    xs = tf.placeholder(tf.float32, input_shape)
    l2_x = tf.placeholder(tf.float32, input_shape)
    l2_orig = tf.placeholder(tf.float32, input_shape)
    label = tf.placeholder(tf.int32, ())
    one_hot = tf.expand_dims(tf.one_hot(label, 1000), axis=0)

    xs_def = tf.expand_dims(defense.tftensorGD(xs), axis=0)
    logits, preds = inceptionv3.model(sess, xs_def)
    l2_loss = tf.sqrt(2 * tf.nn.l2_loss(l2_x - l2_orig) / np.product(input_shape))

    xs_ens = tf.stack([defense.tftensorGD(xs) for _ in range(ens_size)], axis=0)
    logits_ens, preds_ens = inceptionv3.model(sess, xs_ens)

    labels_ens = tf.tile(one_hot, (logits_ens.shape[0], 1))
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_ens, labels=labels_ens))
    loss_ens = xent + lam * tf.maximum(l2_loss - epsilon, 0)
    grad_ens, = tf.gradients(loss_ens, xs)

    adv = np.copy(data)
    for index in range(data.shape[0]):

        adv_eot = np.copy(adv[index])

        for i in tqdm.tqdm(range(max_steps)):
            adv_def = defense.defend_FD_sig(adv_eot)
            p, l2 = sess.run([preds, l2_loss], {xs: adv_def, l2_x: adv_eot, l2_orig: data[index]})
            if p == targets[index] and l2 < epsilon:
                print("Found AE. Iter: {}. L2: {}.".format(i, l2))
                break
            elif l2 > epsilon:
                print("Can't find AE under l2-norm 0.05.")
                break

            g_ens, p_ens = sess.run([grad_ens, preds_ens], {xs: adv[index], label: targets[index]})
            adv_eot -= lr * g_ens
            adv_eot = np.clip(adv_eot, 0, 1)

        adv[index] = adv_eot

    return adv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--output-path', type=str, default='./outputs')
    parser.add_argument('--output-name', type=str, default='adv_bpda_eot.npy')
    parser.add_argument('--defense', type=str, default='GD')

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    data = np.load(os.path.join(args.data_path, "clean100data.npy"))
    labels = np.load(os.path.join(args.data_path, "clean100label.npy"))
    targets = np.load(os.path.join(args.data_path, "random_targets.npy"))

    adv = defend_BPDA_EOT(data, args.defense, targets)

    np.save(os.path.join(args.output_path, args.output_name), adv)


if __name__ == '__main__':
    main()