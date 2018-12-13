import json
import cv2
import os

Speed = True
 

if Speed:
    dir_test_img = "640_360/images/"
    dir_test_segm = "640_360/label37/"
    with open("test_640_360.odgt", "a") as savefile:
        for file in os.listdir(dir_test_segm):
                img = cv2.imread(dir_test_segm + file)
                size = img.shape
                width = size[1]
                height = size[0]
                fpath_segm = str(dir_test_segm + file)
                fpath_img = dir_test_img + "img-" + file[-10:-4]+".jpg"
                print(fpath_img)
                simple_dict = {"width": width, "fpath_img": fpath_img,
                                "height": height, "fpath_segm": fpath_segm}
                json.dump(simple_dict, savefile, ensure_ascii=False)
                savefile.write('\n')
    dir_test_img = "1280_720/images/"
    dir_test_segm = "1280_720/label37/"
    with open("test_1280_720.odgt", "a") as savefile:
        for file in os.listdir(dir_test_segm):
                img = cv2.imread(dir_test_segm + file)
                size = img.shape
                width = size[1]
                height = size[0]
                fpath_segm = str(dir_test_segm + file)
                fpath_img = dir_test_img + "img-" + file[-10:-4]+".jpg"
                print(fpath_img)
                simple_dict = {"width": width, "fpath_img": fpath_img,
                                "height": height, "fpath_segm": fpath_segm}
                json.dump(simple_dict, savefile, ensure_ascii=False)
                savefile.write('\n')
    dir_test_img = "1920_1080/images/"
    dir_test_segm = "1920_1080/label37/"
    with open("test_1920_1080.odgt", "a") as savefile:
        for file in os.listdir(dir_test_segm):
                img = cv2.imread(dir_test_segm + file)
                size = img.shape
                width = size[1]
                height = size[0]
                fpath_segm = str(dir_test_segm + file)
                fpath_img = dir_test_img + "img-" + file[-10:-4]+".jpg"
                print(fpath_img)
                simple_dict = {"width": width, "fpath_img": fpath_img,
                                "height": height, "fpath_segm": fpath_segm}
                json.dump(simple_dict, savefile, ensure_ascii=False)
                savefile.write('\n')
else:
    dir_train_img = "SUN_RGBD/images/training/"
    # dir_train_segm = "SUN_RGBD/annotations/training/"
    dir_train_segm = "SUN_RGBD/label37/train/"

    ###### train.json #######
    with open("train_sun37.odgt", "a") as savefile:
        for file in os.listdir(dir_train_segm):
                img = cv2.imread(dir_train_segm + file)
                size = img.shape
                width = size[1]
                height = size[0]
                fpath_segm = str(dir_train_segm + file)
                # fpath_img = dir_train_img + file[:3]+file[-11:-4]+".jpg"
                fpath_img = dir_train_img + "img-" + file[-10:-4]+".jpg"
                print(fpath_img)
                simple_dict = {"width": width, "fpath_img": fpath_img,
                                "height": height, "fpath_segm": fpath_segm}
                json.dump(simple_dict, savefile, ensure_ascii=False)
                savefile.write('\n')


    ###### test.json #######
    dir_test_img = "SUN_RGBD/images/validation/"
    # dir_test_segm = "SUN_RGBD/annotations/validation/"
    dir_test_segm = "SUN_RGBD/label37/test/"
    with open("test_sun37.odgt", "a") as savefile:
        for file in os.listdir(dir_test_segm):
                img = cv2.imread(dir_test_segm + file)
                size = img.shape
                width = size[1]
                height = size[0]
                fpath_segm = str(dir_test_segm + file)
                # fpath_img = dir_test_img + file[:3]+file[-11:-4]+".jpg"
                fpath_img = dir_test_img + "img-" + file[-10:-4]+".jpg"
                print(fpath_img)
                simple_dict = {"width": width, "fpath_img": fpath_img,
                                "height": height, "fpath_segm": fpath_segm}
                json.dump(simple_dict, savefile, ensure_ascii=False)
                savefile.write('\n')