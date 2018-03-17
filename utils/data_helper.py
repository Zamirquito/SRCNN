import matplotlib.image as im
import os
from utils import args
import numpy as np
import PIL.Image as Image
import h5py
flags = args.get_args()


# 该函数用于对训练集进行预处理，并保存在dir中（保存成h5文件以便加快训练时读取速度）
def gen_input_image(data_dir, h5_dir, stride):
    if not os.path.exists(data_dir):
        print("the data path don't exist!")
        raise ValueError

    path = os.path.join(os.getcwd(), h5_dir)

    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, 'data.h5')
    if os.path.exists(path):
        return

    size_input = flags.image_size
    size_label = flags.label_size
    scale = flags.scale
    padding = abs(size_input - size_label) / 2

    filenames = os.listdir(data_dir)
    filelist = []
    for f in filenames:
        filelist.append(os.path.join(data_dir, f))
    dataset = []
    labelset = []

    # 对每一张图片进行处理(rgb2ycbcr, modcrop, resize, split)
    for f in filelist:
        image = im.imread(f)

        # rgb2ycbcr
        if image.shape[2] == 3:
            image = rgb2ycbcr(image)

        # modcrop
        image = image[:, :, 0]
        im_gnd = modcrop(image, scale)
        im_gnd = im_gnd / 255
        h, w = im_gnd.shape

        # resize
        im_tmp = Image.fromarray(im_gnd)
        im_tmp = im_tmp.resize((h//scale, w//scale), resample=Image.BICUBIC)
        im_tmp = im_tmp.resize((h, w), resample=Image.BICUBIC)
        im_in = np.asarray(im_tmp).T

        # split
        for x in range(0, h-size_input+1, stride):
            for y in range(0, w-size_input+1, stride):
                sub_input = im_in[x:x+size_input, y:y+size_input]
                sub_label = im_gnd[x+int(padding):x+int(padding)+size_label, y+int(padding):y+int(padding)+size_label]

                sub_input = sub_input.reshape([size_input, size_input, 1])
                sub_label = sub_label.reshape([size_label, size_label, 1])
                dataset.append(sub_input)
                labelset.append(sub_label)

    # 保存至h5中
    dataset = np.asarray(dataset)
    labelset = np.asarray(labelset)

    with h5py.File(path, 'w') as hf:
        hf.create_dataset('data', data=dataset)
        hf.create_dataset('label', data=labelset)


def load_data(h5_dir):
    with h5py.File(h5_dir, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


# 仅实现jpeg模式，具体公式可参照wiki, https://en.wikipedia.org/wiki/YCbCr
def rgb2ycbcr(image):
    kernel = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    ycbcr_image = image.dot(kernel)
    ycbcr_image[:, :, 1:] += 128
    return ycbcr_image


# 用于将image对齐
def modcrop(image, scale=3):
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
    return image

