#!/usr/bin/env python3

import rclpy
import math
import numpy as np
from rclpy.node import Node
from example_interfaces.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

import sys

from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import cv2
print(cv2.__version__)
bridge = CvBridge()


class Semantic (Node):
    def __init__(self):
        super().__init__("Semantic_segmantation")
        self.create_subscription(Image,"/zed2/zed_node/rgb/image_rect_color",self.img_cb, rclpy.qos.qos_profile_sensor_data)
        self.get_logger().info("subscriber is started")

    def img_cb(self,message):
        net = model_zoo.get_model('mask_rcnn_fpn_resnet101_v1d_coco', pretrained=True)
        
        cv2_img = bridge.imgmsg_to_cv2(message)
        cv2.imwrite('saved_img.png', cv2_img)
        x, orig_img = data.transforms.presets.rcnn.load_test("saved_img.png")
        
        ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

        # paint segmentation mask on images directly
        width, height = orig_img.shape[1], orig_img.shape[0]
        #masks, _ = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
        orig_img = utils.viz.plot_mask(orig_img, masks)

        # identical to Faster RCNN object detection
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                                class_names=net.classes, ax=ax)
        plt.show()

        

              
def main (args=None):
    rclpy.init(args=args)
    node=Semantic()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=="__main__":
    main()

