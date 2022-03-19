import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# 用来正常显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]

if __name__ == "__main__":
    img = cv2.imread("/home/cvnlp/A.png")
    # 将图片进行随机裁剪为280×280
    crop_img = tf.random_crop(img, [400, 400, 3])

    sess = tf.InteractiveSession()
    # 显示图片
    cv2.imwrite("/home/cvnlp/A_.png",crop_img.eval())
    plt.figure(1)
    plt.subplot(121)
    # 将图片由BGR转成RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("原始图片")
    plt.subplot(122)
    crop_img = cv2.cvtColor(crop_img.eval(), cv2.COLOR_BGR2RGB)
    # img = Image.fromarray(crop_img.astype('uint8')).convert('RGB')
    # print(img)
   # img.save("/home/cvnlp/", "A-.png")
    plt.title("裁剪后的图片")
    plt.imshow(crop_img)
    plt.show()
    sess.close()

