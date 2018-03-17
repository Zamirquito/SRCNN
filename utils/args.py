import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("channels", 1, "Number of the input channels")
flags.DEFINE_integer("image_size", 33, "the size of input image to the model")
flags.DEFINE_integer("label_size", 21, "the size of label image")
flags.DEFINE_integer("scale", 3, "the scale of image you want to modify")
flags.DEFINE_integer("train_stride", 14, "stride of split a image per step")
flags.DEFINE_integer("test_stride", 21, "stride of split a image per step")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("epoch", 10000, "total epoch in training process")
flags.DEFINE_integer("cheak_freq", 50, "save the model per 500 epoch")

flags.DEFINE_float("lr", 1e-3, "learning rate")
# flags.DEFINE_float("lr_decay", 0.99, "learning rate decay")
# flags.DEFINE_float("lr_min", 1e-4, "learning rate min")

flags.DEFINE_string("data_dir", "train", "the dir of your training data")
flags.DEFINE_string("h5_dir", "train_tmp", "the dir of your training data(h5 file)")
flags.DEFINE_string("test_dir", "test", "the dir of your test data")
flags.DEFINE_string("test_h5_dir", "test_tmp", "the dir of your test data(h5 file)")
flags.DEFINE_string("model_dir", "model", "the path to save model while training")
flags.DEFINE_string("log_dir", "log", "the dir to use in tensorboard")
flags.DEFINE_string("mode", "train", "choose a mode to use(train, test or apply)")
flags.DEFINE_string("image_file", "image.jpg", "choose a image to test")


def get_args():
    return FLAGS

