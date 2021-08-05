import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os, time


class DoxieCGAN():
    def __init__(self, data, img_height=64, img_width=64, channels=3,
                 batch_size=64, lr_gen=2e-4, lr_disc=2e-4, noise_std=-1,
                 use_augments=False, labelsmoothing=1., weight_std=0.02,
                 save_checkpoints=True, name="doxie_generator"):
        # Assume that data comes normalised in [-1, 1] range
        self.data = data

        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.batch_size = batch_size

        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self._gen_opt = tf.keras.optimizers.Adam(self.lr_gen, beta_1=0.5)
        self._disc_opt = tf.keras.optimizers.Adam(self.lr_disc, beta_1=0.5)

        # Use if > 0
        self.noise_std = noise_std
        if self.noise_std > 0:
            self._noise_layer = layers.GaussianNoise(self.noise_std)
        else:
            self._noise_layer = None

        self.use_augments = use_augments
        self._augments = tf.keras.models.Sequential([
            layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
            layers.experimental.preprocessing.RandomRotation(0.2)
        ])

        self.labelsmoothing = labelsmoothing

        self.weight_std = weight_std
        self._weight_init = tf.keras.initializers\
            .RandomNormal(mean=0., stddev=self.weight_std)
        self._gamma_init = tf.keras.initializers\
            .RandomNormal(mean=1., stddev=self.weight_std)

        # Losses
        self._cross_entropy = tf.keras.losses\
            .BinaryCrossentropy(from_logits=False)

        self.name = name

        # Other

        self._noise_dim = 100  # size of input to G
        self._visualise_seed = tf.random.normal([16, self._noise_dim])

        self._generator = None
        self._discriminator = None
        self._centropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.history = None

        # Checkpoints

        self.save_checkpoints = save_checkpoints
        self.ckpt_dir = os.path.join('drive', 'MyDrive',
                                     'training_checkpoints')
        self.ckpt_prefix = 'ckpt'

    def _get_prefix(self):
        return os.path.join(self.ckpt_dir, self.ckpt_prefix)

    def _create_ckpt_mgr(self, ckpt):
        return tf.train.CheckpointManager(ckpt, self._get_prefix(),
                                          max_to_keep=10)

    def _strided_conv(self, filters, kernel_size, strides, padding):
        return layers.Conv2DTranspose(filters, kernel_size, strides=strides,
                                      padding=padding, use_bias=False,
                                      kernel_initializer=self._weight_init)

    def _bnorm(self):
        return layers.BatchNormalization(beta_initializer=self._weight_init,
                                         gamma_initializer=self._gamma_init)

    def _conv2d(self, filters, kernel_size, strides, padding):
        return layers.Conv2D(filters, kernel_size, strides=strides,
                             padding=padding, use_bias=False,
                             kernel_initializer=self._weight_init)

    def make_gen(self):
        gen_inputs = tf.keras.Input(shape=(self._noise_dim,))
        # (100, , )

        x = layers.Dense(4*4*1024, use_bias=False,
                         kernel_initializer=self._weight_init)(gen_inputs)
        x = self._bnorm()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Reshape((4, 4, 1024))(x)
        # (4, 4, 1024)

        x = self._strided_conv(512, 4, 2, 'same')(x)
        x = self._bnorm()(x)
        x = layers.LeakyReLU(0.2)(x)
        # (8, 8, 512)

        x = self._strided_conv(256, 4, 2, 'same')(x)
        x = self._bnorm()(x)
        x = layers.LeakyReLU(0.2)(x)
        # (16, 16, 256)

        x = self._strided_conv(128, 4, 2, 'same')(x)
        x = self._bnorm()(x)
        x = layers.LeakyReLU(0.2)(x)
        # (32, 32, 128)

        #x = strided_conv(64, 4, 2, 'same')(x)
        #x = bnorm()(x)
        #x = layers.LeakyReLU(0.2)(x)
        ## (64, 64, 64)
        x = self._strided_conv(3, 4, 2, 'same')(x)
        # (64, 64, 3)

        gen_outputs = layers.Activation('tanh')(x)
        self._generator = tf.keras.Model(gen_inputs, gen_outputs, name='generator')

    def make_disc(self):
        disc_inputs = tf.keras.Input((self.img_height,
                                      self.img_width,
                                      self.channels))
        # (64, 64, 3)
        #x = conv2d(32, 4, 2, 'same')(disc_inputs)
        #x = layers.LeakyReLU(0.2)(x)
        # (64, 64, 32)
        x = self._conv2d(64, 4, 2, 'same')(disc_inputs)
        #x = bnorm()(x)
        x = layers.LeakyReLU(0.2)(x)
        # (32, 32, 64)
        x = self._conv2d(128, 4, 2, 'same')(x)
        x = self._bnorm()(x)
        x = layers.LeakyReLU(0.2)(x)
        # (16, 16, 128)
        x = self._conv2d(256, 4, 2, 'same')(x)
        x = self._bnorm()(x)
        x = layers.LeakyReLU(0.2)(x)
        # (8, 8, 256)
        x = self._conv2d(512, 4, 2, 'same')(x)
        x = self._bnorm()(x)
        x = layers.LeakyReLU(0.2)(x)
        # (4, 4, 512)
        x = layers.Flatten()(x)
        disc_output = layers.Dense(1, activation='sigmoid', use_bias=False,
                                   kernel_initializer=self._weight_init)(x)

        self._discriminator = tf.keras.Model(disc_inputs, disc_output)

    def setCheckpoint(self):
        pass

    @tf.function
    def _train_step(self, images):
        # First update D:
        z = tf.random.normal([self.batch_size, self._noise_dim])

        # Here we should separate the Gradient updates with only real or only
        # generated image batches
        with tf.GradientTape() as disc_tape:
            T_x = self._augments(images)
            if self._noise_layer:
                T_x = self._noise_layer(T_x)
            if self._discriminator:
                D_x = self._discriminator(T_x, training=True)
            else:
                raise Exception("Please initialise the discriminator before " +
                                "training: DoxieCGAN.make_disc()")
            # log(D(x))
            # real_loss = cross_entropy(tf.ones_like(D_x), D_x)
            # We can do lable smoothing by passing 0.9 instead of 1.
            real_loss = self._centropy(tf.experimental.numpy
                                       .full_like(D_x, 0.9), D_x)

            if self._generator:
                G_z = self._generator(z, training=True)
            else:
                raise Exception("Please initialise the generator before " +
                                "training: DoxieCGAN.make_gen()")

            TG_z = self._augments(G_z)
            if self._noise_layer:
                TG_z = self._noise_layer(TG_z)

            DG_z = self._discriminator(TG_z, training=True)
            # log(1-D(G(z)))
            fake_loss = self._centropy(tf.zeros_like(DG_z), DG_z)
            # log(D(x)) + log(1-D(G(z)))
            total_loss = real_loss + fake_loss

        grad_D = disc_tape.gradient(total_loss,
                                    self._discriminator.trainable_variables)
        self._disc_opt.apply_gradients(zip(grad_D, self._discriminator.trainable_variables))

        # Now update G:
        with tf.GradientTape() as gen_tape:
            G_z = self._generator(z, training=True)
            TG_z = self._augments(G_z)
            if self._noise_layer:
                TG_z = self._noise_layer(TG_z)
            DG_z = self._discriminator(TG_z, training=True)
            # log(D(G(z)))
            gen_loss = self._centropy(tf.ones_like(DG_z), DG_z)

        grad_G = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
        self._gen_opt.apply_gradients(zip(grad_G,
                                          self._generator.trainable_variables))

        return total_loss, gen_loss, D_x, DG_z

    def _train_loop(self, epochs, from_checkpoint=False):
        batch_str = ""
        epoch_str = ""
        if self.data:
            n_batches = self.data.cardinality().numpy()
        else:
            raise Exception("No data loaded. Supply a tf.DataSet to be used " +
                            "with the discriminator.")
        D_losses = []
        G_losses = []
        D_x_mu = []
        DG_z_mu = []
        initial_epoch = 0
        if from_checkpoint or self.save_checkpoints:
            ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                       D_losses = D_losses,
                                       G_losses = G_losses,
                                       D_x_mu = D_x_mu,
                                       DG_z_mu = DG_z_mu,
                                       gen_opt = self._gen_opt,
                                       disc_opt = self._disc_opt,
                                       generator = self._generator,
                                       discrminator = self._discriminator)
            mgr = self._create_ckpt_mgr(ckpt)


        if from_checkpoint:
            ckpt.restore(mgr.checkpoints[-2])

            if mgr.latest_checkpoint:
                initial_epoch = int(ckpt.step) - 1
                D_losses = ckpt.D_losses
                G_losses = ckpt.G_losses
                D_x_mu = ckpt.D_x_mu
                DG_z_mu = ckpt.DG_z_mu
                self._gen_opt = ckpt.gen_opt
                self._disc_opt = ckpt.disc_opt
                self._generator = ckpt.generator
                self._discriminator = ckpt.discrminator
                print(f"Restored from {mgr.latest_checkpoint} at epoch"
                      + f"{initial_epoch + 1}.")
            else:
                print("Cannot load from checkpoint. Training from scratch.")

        for epoch in range(initial_epoch, epochs):
            start = time.time()

            i=1
            temp_str = ""
            for image_batch in self.data:
                DL, GL, Dx, DGz = self._train_step(image_batch)
                Dx = Dx.numpy().mean()
                DGz = DGz.numpy().mean()
                D_losses.append(DL.numpy())
                G_losses.append(GL.numpy())
                D_x_mu.append(Dx)
                DG_z_mu.append(DGz)
                temp_str = f"Batch: {i}/{n_batches}. Epoch: {epoch+1}. D_loss: {DL:.3}."
                temp_str += f" G_loss: {GL:.3}. D(x): {Dx:.3}. D(G(z)): {DGz:.3}."
                print('\r' + temp_str + epoch_str, end='', flush=True)
                i+=1

            print('\n')
            # Produce images for the GIF as you go
            # display.clear_output(wait=True)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.generate_and_save_images(epoch,
                                              self._visualise_seed)
            ckpt.step.assign_add(1)
            if (epoch + 1) % 50 == 0:
                ckpt.D_losses = D_losses
                ckpt.G_losses = G_losses
                ckpt.D_x_mu = D_x_mu
                ckpt.DG_z_mu = DG_z_mu
                ckpt.generator = self._generator
                ckpt.discriminator = self._discriminator
                ckpt.disc_opt = self._disc_opt
                ckpt.gen_opt = self._gen_opt
                save_path = mgr.save()
                print("Saved checkpoint for epoch "
                      + f"{int(ckpt.step)}: {save_path}.")
                #checkpoint.save(file_prefix = checkpoint_prefix)

            epoch_str = (f" Time for epoch {epoch+1} is" +
                         " {time.time()-start:.1f} sec.")

        # Generate after the final epoch
        #display.clear_output(wait=True)
        self.generate_and_save_images(epochs,
                                      self._visualise_seed)

        self.history = {'D_loss': D_losses, 'G_loss': G_losses, 'D(x)': D_x_mu,
                        'DG(z)': DG_z_mu}

    def train(self, epochs, from_checkpoint=False):
        start_train = time.time()
        self._train_loop(epochs, from_checkpoint)
        print(f"Total train time {(time.time() - start_train)//60}")

    def plotLoss(self):
        pass

    def plotDG(self):
        pass

    def generate_and_save_images(self, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self._generator(test_input, training=False)

        plt.figure(figsize=(8, 8))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i].numpy() * 0.5 + 0.5)
            plt.axis('off')

        # plt.savefig('gan_eg\\image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()
