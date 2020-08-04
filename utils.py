import PIL.Image
from imagenet_labels import label_to_name
import matplotlib.pyplot as plt
from defense import *

# Some of the code refering to Anish & Carlini's github: https://github.com/anishathalye/obfuscated-gradients

def getabatch(dataset,labelset,numbatch,batchsize):
    databatch = dataset[numbatch*batchsize:(numbatch*batchsize+batchsize)]
    labelbatch = labelset[numbatch*batchsize:(numbatch*batchsize+batchsize)]
    return databatch,labelbatch


def linf_distortion(img1, img2):
    if len(img1.shape) == 4:
        n = img1.shape[0]
        l = np.mean(np.max(np.abs(img1.reshape((n, -1)) - img2.reshape((n, -1))), axis=1), axis=0)
    else:
        l = np.max(np.abs(img1 - img2))

    return l

def l2_distortion(img1, img2):
    if len(img1.shape) == 4:
        n = img1.shape[0]
        l = np.mean(np.sqrt(np.sum((img1.reshape((n, -1)) - img2.reshape((n, -1)))
                                   ** 2, axis=1) / np.product(img1.shape[1:])), axis=0)
    else:
        l = np.sqrt(np.sum((img1 - img2) ** 2) / np.product(img1.shape))

    return l

def one_hot(index, total):
    arr = np.zeros((total))
    arr[index] = 1.0
    return arr

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def load_image(path):
    return (np.asarray(PIL.Image.open(path).resize((299, 299)))/255.0).astype(np.float32)

def make_classify(sess, input_, probs):
    def classify(img, correct_class=None, target_class=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        fig.sca(ax1)
        p = sess.run(probs, feed_dict={input_: img})[0]
        ax1.imshow(img)
        fig.sca(ax1)

        topk = list(p.argsort()[-10:][::-1])
        topprobs = p[topk]
        barlist = ax2.bar(range(10), topprobs)
        if target_class in topk:
            barlist[topk.index(target_class)].set_color('r')
        if correct_class in topk:
            barlist[topk.index(correct_class)].set_color('g')
        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(10),
                   [label_to_name(i)[:15] for i in topk],
                   rotation='vertical')
        fig.subplots_adjust(bottom=0.2)
        plt.show()
    return classify
