from layer import *
import numpy as np

model_path = '/home/lrl/Siggraph2018/glcic/src/completionnet_places2.t7'

class Network:
    def __init__(self, x, mask, local_x, global_completion, local_completion, is_training, batch_size):
        # pretrain
        import numpy as np
        from torch.utils.serialization import load_lua
        self.initload = load_lua(model_path)
        self.init = self.initload.model
        self.datamean = np.array(self.initload.mean)

        self.batch_size = batch_size
        self.imitation = self.generator_pretrain(tf.concat([x * (1 - mask), mask], 3), is_training)
        self.completion = self.imitation * mask + (x + self.datamean) * (1 - mask)
        self.real = self.discriminator(x, local_x, reuse=False)
        self.fake = self.discriminator(global_completion, local_completion, reuse=True)
        self.g_loss = self.calc_g_loss(x, self.completion)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    '''
    def generator(self, x, is_training):
        with tf.variable_scope('generator'):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [5, 5, 3, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 256], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated1'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated2'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 4)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated3'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 8)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated4'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 16)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv1'):
                x = deconv_layer(x, [4, 4, 128, 256], [self.batch_size, 128, 128, 128], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv9'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [4, 4, 64, 128], [self.batch_size, 256, 256, 64], 2)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv10'):
                x = conv_layer(x, [3, 3, 64, 32], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv11'):
                x = conv_layer(x, [3, 3, 32, 3], 1)
                x = tf.nn.tanh(x)
        return x
    '''
    
    def generator_pretrain(self, x, is_training):
        '''
        0 nn.SpatialConvolution(4 -> 64, 5x5, 1, 1, 2, 2)
        1 nn.SpatialBatchNormalization
        2 nn.ReLU
        3 nn.SpatialConvolution(64 -> 128, 3x3, 2, 2, 1, 1)
        4 nn.SpatialBatchNormalization
        5 nn.ReLU
        6 nn.SpatialConvolution(128 -> 128, 3x3, 1, 1, 1, 1)
        7 nn.SpatialBatchNormalization
        8 nn.ReLU
        9 nn.SpatialConvolution(128 -> 256, 3x3, 2, 2, 1, 1)
        10 nn.SpatialBatchNormalization
        11 nn.ReLU
        12 nn.SpatialDilatedConvolution(256 -> 256, 3x3, 1, 1, 1, 1, 1, 1)
        13 nn.SpatialBatchNormalization
        14 nn.ReLU
        15 nn.SpatialDilatedConvolution(256 -> 256, 3x3, 1, 1, 1, 1, 1, 1)
        16 nn.SpatialBatchNormalization
        17 nn.ReLU
        18 nn.SpatialDilatedConvolution(256 -> 256, 3x3, 1, 1, 2, 2, 2, 2)
        19 nn.SpatialBatchNormalization
        20 nn.ReLU
        21 nn.SpatialDilatedConvolution(256 -> 256, 3x3, 1, 1, 4, 4, 4, 4)
        22 nn.SpatialBatchNormalization
        23 nn.ReLU
        24 nn.SpatialDilatedConvolution(256 -> 256, 3x3, 1, 1, 8, 8, 8, 8)
        25 nn.SpatialBatchNormalization
        26 nn.ReLU
        27 nn.SpatialDilatedConvolution(256 -> 256, 3x3, 1, 1, 16, 16, 16, 16)
        28 nn.SpatialBatchNormalization
        29 nn.ReLU
        30 nn.SpatialDilatedConvolution(256 -> 256, 3x3, 1, 1, 1, 1, 1, 1)
        31 nn.SpatialBatchNormalization
        32 nn.ReLU
        33 nn.SpatialDilatedConvolution(256 -> 256, 3x3, 1, 1, 1, 1, 1, 1)
        34 nn.SpatialBatchNormalization
        35 nn.ReLU
        36 nn.SpatialFullConvolution(256 -> 128, 4x4, 2, 2, 1, 1)
        37 nn.SpatialBatchNormalization
        38 nn.ReLU
        39 nn.SpatialConvolution(128 -> 128, 3x3, 1, 1, 1, 1)
        40 nn.SpatialBatchNormalization
        41 nn.ReLU
        42 nn.SpatialFullConvolution(128 -> 64, 4x4, 2, 2, 1, 1)
        43 nn.SpatialBatchNormalization
        44 nn.ReLU
        45 nn.SpatialConvolution(64 -> 32, 3x3, 1, 1, 1, 1)
        46 nn.SpatialBatchNormalization
        47 nn.ReLU
        48 nn.SpatialConvolution(32 -> 3, 3x3, 1, 1, 1, 1)
        49 nn.Sigmoid
        '''
        with tf.variable_scope('generator'):
            with tf.variable_scope('conv1'):
                #x = tf.Print(x, [tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x)], "Input: ")
                x = conv_layer(x, [5, 5, 4, 64], 1, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[0].weight.numpy(), [2,3,1,0])))
                #x = tf.Print(x, [x.get_shape(), tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x)], "After conv1: ")
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[1].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[1].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[1].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[1].running_var.numpy())
                                   )
                #x = tf.Print(x, [x.get_shape(), tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x)], "After BN: ")
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 128], 2, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[3].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[4].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[4].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[4].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[4].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 128, 128], 1, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[6].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[7].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[7].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[7].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[7].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 256], 2, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[9].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[10].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[10].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[10].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[10].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 256, 256], 1, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[12].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[13].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[13].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[13].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[13].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 1, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[15].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[16].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[16].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[16].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[16].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated1'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 2, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[18].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[19].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[19].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[19].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[19].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated2'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 4, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[21].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[22].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[22].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[22].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[22].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated3'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 8, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[24].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[25].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[25].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[25].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[25].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('dilated4'):
                x = dilated_conv_layer(x, [3, 3, 256, 256], 16, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[27].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[28].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[28].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[28].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[28].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 256], 1, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[30].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[31].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[31].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[31].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[31].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 256, 256], 1, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[33].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[34].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[34].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[34].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[34].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv1'):
                x = deconv_layer(x, [4, 4, 128, 256], [self.batch_size, 128, 128, 128], 2, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[36].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[37].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[37].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[37].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[37].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('conv9'):
                x = conv_layer(x, [3, 3, 128, 128], 1, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[39].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[40].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[40].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[40].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[40].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(x, [4, 4, 64, 128], [self.batch_size, 256, 256, 64], 2, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[42].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[43].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[43].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[43].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[43].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('conv10'):
                x = conv_layer(x, [3, 3, 64, 32], 1, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[45].weight.numpy(), [2,3,1,0])))
                x = batch_normalize(x, is_training,
                                    initializer_beta=tf.constant_initializer(self.init.modules[46].bias.numpy()),
                                    initializer_scale=tf.constant_initializer(self.init.modules[46].weight.numpy()), 
                                    initializer_mean=tf.constant_initializer(self.init.modules[46].running_mean.numpy()), 
                                    initializer_var=tf.constant_initializer(self.init.modules[46].running_var.numpy())
                                   )
                x = tf.nn.relu(x)
            with tf.variable_scope('conv11'):
                x = conv_layer(x, [3, 3, 32, 3], 1, 
                               initializer=tf.constant_initializer(np.transpose(self.init.modules[48].weight.numpy(), [2,3,1,0])))
                x = tf.nn.sigmoid(x) # tanh ?
        return x

    def discriminator(self, global_x, local_x, reuse):
        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv5'):
                    x = conv_layer(x, [5, 5, 512, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv6'):
                    x = conv_layer(x, [5, 5, 512, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        def local_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('local'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv2'):
                    x = conv_layer(x, [5, 5, 64, 128], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv3'):
                    x = conv_layer(x, [5, 5, 128, 256], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv4'):
                    x = conv_layer(x, [5, 5, 256, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('conv5'):
                    x = conv_layer(x, [5, 5, 512, 512], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        with tf.variable_scope('discriminator', reuse=reuse):
            global_output = global_discriminator(global_x)
            local_output = local_discriminator(local_x)
            with tf.variable_scope('concatenation'):
                output = tf.concat((global_output, local_output), 1)
                output = full_connection_layer(output, 1)
               
        return output


    def calc_g_loss(self, x, completion):
        loss = tf.nn.l2_loss(x - completion)
        return tf.reduce_mean(loss)


    def calc_d_loss(self, real, fake):
        alpha = 4e-4
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(d_loss_real, d_loss_fake) * alpha

