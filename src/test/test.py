import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from network import Network

IMAGE_SIZE = 256
LOCAL_SIZE = 128
HOLE_MIN = 48
HOLE_MAX = 120
BATCH_SIZE = 1

test_npy = './lfw.npy'

def test_single():
    from torch.utils.serialization import load_lua
    datamean = load_lua('/home/lrl/Siggraph2018/glcic/src/completionnet_places2.t7').mean
    
    def cvimg2tensor(src):
        out = src.copy()
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = out.transpose((2,0,1)).astype(np.float64)
        out = out / 255
        return out
    
    import os
    import torch
    import cv2
    import numpy as np
    
    # load data
    input_img = cv2.imread('example.png')
    input_img = cv2.resize(input_img, (IMAGE_SIZE, IMAGE_SIZE))
    I = torch.from_numpy(cvimg2tensor(input_img)).float()
    input_mask = cv2.imread('example_mask.png')
    input_mask = cv2.resize(input_mask, (IMAGE_SIZE, IMAGE_SIZE))
    M = torch.from_numpy(cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY) / 255).float()
    M[M <= 0.2] = 0.0
    M[M > 0.2] = 1.0
    M = M.view(1, M.size(0), M.size(1))
    assert I.size(1) == M.size(1) and I.size(2) == M.size(2)
    for i in range(3):
        I[i, :, :] = I[i, :, :] - datamean[i]
    # make mask_3ch
    M_3ch = torch.cat((M, M, M), 0)
    Im = I * (M_3ch*(-1)+1)
    # set up input
    input = torch.cat((Im, M), 0)
    input = input.view(1, input.size(0), input.size(1), input.size(2)).float().numpy()
    print (input.shape, np.min(input))
    x_batch = input[0, 0:3, :, :][np.newaxis, :, :, :]
    mask_batch = input[0, 3, :, :][np.newaxis, np.newaxis, :, :]
    print (x_batch.shape, mask_batch.shape)
    x_batch = x_batch.transpose(0, 2, 3, 1)
    mask_batch = mask_batch.transpose(0, 2, 3, 1)
    
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    #saver = tf.train.Saver()
    #saver.restore(sess, '../backup/latest')
    
#     for variable in tf.all_variables():
#         if 'generator' in variable.name:
#             value = variable.initial_value.eval(session=sess)
#             print( variable.name, variable.get_shape(), np.max(value), np.min(value), np.mean(value))

#     x_test = cv2.imread('example.png')
#     x_test = cv2.resize(x_test, (IMAGE_SIZE, IMAGE_SIZE))
#     x_test = cv2.cvtColor(x_test, cv2.COLOR_BGR2RGB)/255.0
#     x_batch = x_test[np.newaxis, :, :, :] - [ 0.4560,  0.4472,  0.4155]
    
#     x_mask = cv2.imread('example_mask.png')
#     x_mask = cv2.resize(x_mask, (IMAGE_SIZE, IMAGE_SIZE))
#     x_mask = cv2.cvtColor(x_mask, cv2.COLOR_BGR2GRAY)/255.0
#     x_mask[x_mask<0.2] = 0
#     x_mask[x_mask>=0.2] = 1
#     mask_batch = x_mask[np.newaxis, :, :, np.newaxis]
    
    
    
    
    completion = sess.run(model.imitation, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
    cnt = 0
    for i in range(BATCH_SIZE):
#         raw = x_batch[i]
#         raw = np.array((raw + [ 0.4560,  0.4472,  0.4155]) * 255.0, dtype=np.uint8)
#         masked = raw * (1 - mask_batch[i]) + np.ones_like(raw) * mask_batch[i] * 255
        img = completion[i]
        print (img.shape, np.max(img), np.min(img))
        img = np.array((img + [ 0.4560,  0.4472,  0.4155]) * 255.0, dtype=np.uint8)
        print (img.shape, np.max(img), np.min(img))
        cv2.imwrite('output.jpg', img[:,:,::-1])
        #dst = './output/{}.jpg'.format("{0:06d}".format(cnt))
        #output_image([['Input', masked], ['Output', img], ['Ground Truth', raw]], dst)

def test():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    #saver = tf.train.Saver()
    #saver.restore(sess, '../backup/latest')

    x_test = np.load(test_npy)
    np.random.shuffle(x_test)
    x_test = np.array([a / 127.5 - 1 for a in x_test])

    step_num = int(len(x_test) / BATCH_SIZE)

    cnt = 0
    for i in tqdm.tqdm(range(step_num)):
        x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        _, mask_batch = get_points()
        completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
        for i in range(BATCH_SIZE):
            cnt += 1
            raw = x_batch[i]
            raw = np.array((raw + 1) * 127.5, dtype=np.uint8)
            masked = raw * (1 - mask_batch[i]) + np.ones_like(raw) * mask_batch[i] * 255
            img = completion[i]
            img = np.array((img + 1) * 127.5, dtype=np.uint8)
            dst = './output/{}.jpg'.format("{0:06d}".format(cnt))
            output_image([['Input', masked], ['Output', img], ['Ground Truth', raw]], dst)


def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h
        
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)
    

def output_image(images, dst):
    fig = plt.figure()
    for i, image in enumerate(images):
        text, img = image
        fig.add_subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.xlabel(text)
    plt.savefig(dst)
    plt.close()


if __name__ == '__main__':
    test_single()
    
