import pickle as pkl
from PIL import Image, ImageOps
import os, sys
import gzip
import _pickle as pickle
import random
import numpy as np


def main(main_path):
    #framer(main_path, 'Blueberry') #C:/Users/Potato user 1/OneDrive/NnProject/DataBase/Blueberry
    #framer(main_path, 'Blue_Dream') #C:/Users/Potato user 1/OneDrive/NnProject/DataBase/Blue_Dream
    #framer(main_path, 'Lemon_Haze')
    #framer(main_path,'Sour_Diesel')

    data = unpacker(main_path+ '/pickles/Blueberry_images.pickle')
    print(data[0])



    print("finished main process")


def framer(main_path, name, x=150):
    # reframes the image into a 800x800 image with black border
    # how many images per pickle? 150

    imageList = np.ndarray(shape=(1, 800*800, 3), dtype='float32')
    i = 0
    cd = main_path + name
    for _ in os.listdir(cd):
        if _ != 'ready' and i < 1:
            try:
                # crop fuck and destroy original file
                # output = imageList = [[3,800x800], [3,800x800] ... [3,800x800]]
                img = Image.open(cd + '/' + _)

                old_size = img.size
                new_size = (800, 800 + x)
                img = img.crop((0, 0, old_size[0], old_size[1] - x))

                deltaw = int(new_size[0] - old_size[0])
                deltah = int(new_size[1] - old_size[1])
                ltrb_border = (int(deltaw / 2), int(deltah / 2), int(deltaw / 2), int(deltah / 2))
                img_with_border = ImageOps.expand(img, border=ltrb_border, fill='black')

                # img_with_border.save('Blueberry/ready/' + str(i) + '.png'
                img_with_border = img_with_border.resize(size=(800, 800))
                imageList[i] = (list(img_with_border.getdata()))
                i += 1

                if i % 1 == 0:
                    print(str(i) + ' ' + name + ' images processed.')
            except:
                raise

    print('A total of ' + str(i) + ' ' + name + ' images have been processed.')
    image_packer(main_path, name, imageList)


def image_packer(main_path, path, list):
    with gzip.open(main_path + 'pickles/' + path + '_images.pickle', 'wb') as f:
        print(path + 'image.pickle is being compressed and saved...')
        pickle.dump(list, f, protocol=-1)
    print(path + 'image.pickle has been saved at:' + main_path + 'pickles/' + path + '_images.pickle')

def mixed_packer(main_path, images_list, labels_list):
    with gzip.open(main_path + 'images.pickle', 'wb') as f:
        print('images.pickle is being compressed and saved...')
        pickle.dump(images_list, f, protocol=-1)
    print('images.pickle has been saved at:' + main_path + 'images.pickle')

    with gzip.open(main_path + 'labels.pickle', 'wb') as f:
        print('labels.pickle is being compressed and saved...')
        pickle.dump(labels_list, f, protocol=-1)
    print('labels.pickle has been saved at:' + main_path + 'labels.pickle')


def unpacker(path):
    with gzip.open(path, 'rb') as f:
        print(path + 'is uncompressing and loading...')
        data = pickle.load(f)
    print(path + ' has been uploaded')
    return data


def mixer(pickle_dir, image_count):
    data = np.ndarray(shape=(1, 1, 800*800, 3))
    num_files = len(os.listdir(pickle_dir))

    finalImage = np.ndarray(shape=(image_count, 800*800, 3))
    finalLabel = np.ndarray(shape=(image_count, 1, 4))

    labelList = []
    for _ in os.listdir(pickle_dir):
        temp_data = unpacker(pickle_dir + _)
        data[0] = temp_data
        labelList.append(_)

    for i in range(image_count):
        rand_1 = random.randint(0, 3)
        temp_data = data[rand_1]

        rand_2 = random.randint(0, len(temp_data) - 1)
        temp_image = temp_data[rand_2]

        temp_label = [0] * num_files
        temp_label[rand_1] = 1

        finalImage[i] = (temp_image)
        finalLabel[i] = (temp_label)

    mixed_packer(pickle_dir, finalImage, finalLabel)

    return finalImage, finalLabel


if __name__ == '__main__':
    main_path = 'C:/Users/Potato user 1/OneDrive/NnProject/DataBase/'
    main(main_path)
