import tensorflow as tf
import numpy as np
from PIL import Image
import random

import glob

import os.path
import sys

# Model
class VAEGANModel:
        def __init__(self, latent_size=1024, mb_size=24, kappa=1.0, gamma=20.0):
                def vaegan_conv(inp, filters, filter_size, name, activation=tf.nn.relu):
                    return activation(tf.layers.batch_normalization(
                                        tf.layers.conv2d(inp, 
                                                filters, 
                                                filter_size, 
                                                strides=1, 
                                                padding='same',
                                                activation=None,
                                                name=name), training=True))

                def vaegan_deconv(inp, filters, filter_size, stride, name, activation=tf.nn.relu):
                    return activation(tf.layers.batch_normalization(
                                            tf.contrib.layers.convolution2d_transpose(inp,
                                                                            num_outputs=filters,
                                                                            kernel_size=filter_size,
                                                                            stride=stride,
                                                                            padding='same',
                                                                            activation_fn=None,
                                                                            scope=name), training=True))

                def vaegan_dense(inp, hus, name, activation=tf.nn.relu):
                    return activation(tf.layers.batch_normalization(
                                            tf.layers.dense(inp, hus, activation=None, name=name), training=True))

                X_raw = tf.placeholder(tf.float32, shape=(mb_size, 64, 64, 3))
                X = X_raw / 225.0

                # Encoder
                conv1 = vaegan_conv(X, 64, 5, "enc1")
                conv2 = vaegan_conv(conv1, 128, 3, "enc2")
                conv3 = vaegan_conv(conv2, 256, 3, "enc3")
                flt = tf.contrib.layers.flatten(conv3)
                enc_d1 = vaegan_dense(flt, 128, "enc4")
                mu_sig = vaegan_dense(enc_d1, latent_size * 2, "enc5") 
                mu = mu_sig[:, :latent_size]
                sigma = mu_sig[:, latent_size:]
                epsilon = tf.random_normal([mb_size, latent_size], 0.0, 1.0, dtype=tf.float32)
                z = mu + (tf.sqrt(tf.exp(sigma)) * epsilon)

                p_z = tf.random_normal([mb_size, latent_size], 0.0, 1.0, dtype=tf.float32)
                z_comb = tf.concat([z, p_z], 0)

                # Decoder / Reconstructor
                dec_d1 = vaegan_dense(z_comb, 256, "dec1")
                dec_d1_rshp = tf.reshape(dec_d1, [-1, 16, 16, 1])
                deconv_1 = vaegan_deconv(dec_d1_rshp, 256, 3, 1, "dec2")
                deconv_2 = vaegan_deconv(deconv_1, 128, 5, 1, "dec3")
                deconv_3 = vaegan_deconv(deconv_2, 32, 5, 2, "dec4")
                deconv_4 = vaegan_deconv(deconv_3, 3, 5, 2, "dec5", activation=tf.nn.sigmoid)
                X_hat = deconv_4[:mb_size]

                # GAN Discriminator
                X_gan = tf.concat([X, deconv_4], 0)
                gan_conv1 = tf.layers.conv2d(X_gan, 32, 5, 1, name="gan1", activation=tf.nn.relu) # No BN
                gan_conv2 = vaegan_conv(gan_conv1, 128, 5, "gan2")
                gan_conv3 = vaegan_conv(gan_conv2, 256, 5, "gan3")
                gan_conv4 = vaegan_conv(gan_conv3, 256, 5, "gan4")
                gan_dense1 = vaegan_dense(gan_conv4, 512, "gan5")
                gan_out = vaegan_dense(gan_dense1, 1, "gan6", activation=tf.nn.sigmoid)

                # Loss
                reconstruction_loss = -1.0 * tf.reduce_sum(X * tf.log(tf.clip_by_value(X_hat, 1e-10, 1.0))
                                            + (1.0 - X) * tf.log(tf.clip_by_value(1.0 - X_hat, 1e-10, 1.0)))

                KL_divergence = -1.0 * tf.reduce_sum(1.0 + sigma - mu**2 - tf.exp(sigma))

                GAN_X = gan_out[:mb_size]
                GAN_X_hat = gan_out[mb_size:mb_size*2]
                GAN_pz = gan_out[mb_size*2:]
                GAN_loss = -1.0 * tf.reduce_sum(tf.log(tf.clip_by_value(GAN_X, 1e-10, 1.0))
				        + tf.log(tf.clip_by_value(1.0 - GAN_X_hat, 1e-10, 1.0))
				        + tf.log(tf.clip_by_value(1.0 - GAN_pz, 1e-10, 1.0)))

                enc_loss = tf.reduce_mean(reconstruction_loss + kappa * KL_divergence)
                dec_loss = tf.reduce_mean(gamma * reconstruction_loss - GAN_loss)
                gan_loss = tf.reduce_mean(GAN_loss)

                # Define subnets
                enc = filter(lambda x: x.name.startswith("enc"), tf.trainable_variables())
                dec = filter(lambda x: x.name.startswith("dec"), tf.trainable_variables())
                gan = filter(lambda x: x.name.startswith("gan"), tf.trainable_variables())

                # Opt
                train_step_enc = tf.train.AdamOptimizer(0.001).minimize(enc_loss, var_list=enc)
                train_step_dec = tf.train.AdamOptimizer(0.001).minimize(dec_loss, var_list=dec)
                train_step_gan = tf.train.AdamOptimizer(0.001).minimize(gan_loss, var_list=gan)

                # Summary (Tensorboard)
                tf.summary.scalar("enc_loss", enc_loss)
                tf.summary.scalar("dec_loss", dec_loss)
                tf.summary.scalar("gan_loss", gan_loss)
                tf.summary.scalar("reconstruction_loss", reconstruction_loss)
                tf.summary.scalar("kl_divergence", KL_divergence)
                tf.summary.scalar("GAN_loss", GAN_loss)
                tf.summary.histogram("z", z)
                tf.summary.histogram("gan_out", gan_out)
                tf.summary.histogram("gan_X", GAN_X)
                tf.summary.histogram("gan_X_hat", GAN_X_hat)
                tf.summary.image("in", X, max_outputs=1)
                tf.summary.image("out", X_hat, max_outputs=1)
                summary_op = tf.summary.merge_all()

                # Model I/O
                saver = tf.train.Saver()

                # Outbound
                self.saver = saver
                self.summary_op = summary_op
                self.train_step_enc = train_step_enc
                self.train_step_dec = train_step_dec
                self.train_step_gan = train_step_gan
                self.reconstruction_loss = reconstruction_loss
                self.KL_divergence = KL_divergence
                self.X_hat = X_hat
                self.X_raw = X_raw
                self.z = z
                self.mu = mu

class Batcher():
    def __init__(self, root_path):
        self.paths = glob.glob("{}/*.jpg".format(root_path))
        self.i = 0

    def batch(self, mb_size):
        b = np.zeros((mb_size, 64, 64, 3))
        for i in xrange(mb_size):
            img = Image.open(self.paths[self.i])
            img.load()
            b[i] = np.asarray(img)
            self.i += 1
            if self.i >= len(self.paths):
                self.i = 0
                random.shuffle(self.paths)
        return b
        

if __name__ == "__main__":
    with tf.Session() as sess:
        model = VAEGANModel(mb_size=32)
        batcher = Batcher("data")        

        # Summary
        summary_writer = tf.summary.FileWriter('logs/vaegan', graph=sess.graph)

        # Load / Init model weights
        if os.path.isfile("models/vaegan.ckpt.meta"):
            print "* Restoring saved parameters"
            model.saver.restore(sess, "models/vaegan.ckpt")
        else:
            print "* Initializing parameters"
            sess.run(tf.global_variables_initializer())

        print "# Logging step and loss, see Tensorboard for more information..."
        for epoch in xrange(100):
            for mb_n in xrange(int(202597/32)):
                mb = batcher.batch(32)

                _, _, _, summary = sess.run([   model.train_step_enc,
                                                model.train_step_dec,
                                                model.train_step_gan, 
                                                model.summary_op], feed_dict={model.X_raw: mb})
                
                if mb_n % 3 == 0:
                    summary_writer.add_summary(summary, mb_n)

                print "MB {} of {} of epoch {}".format(mb_n, int(202597/8), epoch)

                if mb_n % 250 == 0:
                    print "* Saving model"
                    model.saver.save(sess, "models/vaegan.ckpt")
