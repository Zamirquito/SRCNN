import tensorflow as tf
from model import SRCNN
from utils import data_helper, args
import os

flags = args.get_args()


def main():
    if flags.mode == "train":
        train()
    elif flags.mode == "test":
        test()
    else:
        print("No mode called ", flags.mode)


def train():
    print("process the image to h5file.....")
    data_dir = flags.data_dir
    h5_dir = flags.h5_dir
    stride = flags.train_stride
    data_helper.gen_input_image(data_dir, h5_dir, stride)

    print("reading data......")
    h5_path = os.path.join(h5_dir, "data.h5")
    data, label = data_helper.load_data(h5_path)

    print("initialize the model......")
    model = SRCNN(flags)
    model.build_graph()
    model.train(data, label)


def test():
    print("process the image to h5file.....")
    test_dir = flags.test_dir
    test_h5_dir = flags.test_h5_dir
    stride = flags.test_stride
    if not os.path.exists(test_h5_dir):
        os.makedirs(test_h5_dir)

    test_set5 = os.path.join(test_dir, 'Set5')
    test_set14 = os.path.join(test_dir, 'Set14')
    path_set5 = os.path.join(test_h5_dir, 'Set5')
    path_set14 = os.path.join(test_h5_dir, 'Set14')
    data_helper.gen_input_image(test_set5, path_set5, stride)
    data_helper.gen_input_image(test_set14, path_set14, stride)

    print("initialize the model......")
    model_dir = flags.model_dir
    model = SRCNN(flags)
    model.build_graph()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(model.sess, ckpt.model_checkpoint_path)
    else:
        print("model info didn't exist!")
        raise ValueError

    print("test in Set5......")
    test_h5_path = os.path.join(path_set5, "data.h5")
    data_set5, label_set5 = data_helper.load_data(test_h5_path)
    accu = model.test(data_set5, label_set5)
    print("the accuracy in Set5 is %.5f", accu)

    print("test in Set14......")
    test_h5_path = os.path.join(path_set14, "data.h5")
    data_set14, label_set14 = data_helper.load_data(test_h5_path)
    accu2 = model.test(data_set14, label_set14)
    print("the accuracy in Set14 is %.5f", accu2)


if __name__ == "__main__":
    main()

