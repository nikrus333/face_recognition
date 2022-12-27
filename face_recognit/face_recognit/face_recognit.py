import argparse
import cv2
import numpy as np
import torch
import math
import time
from backbones.init import get_model

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from face_detect.msg import Face 

class FaceRecognition(Node):
    def __init__(self):
        self.br = CvBridge()
        self.k = 0.52
        self.strat = time.time()
        super().__init__('minimal_subscriber')
        self.test_pub = self.create_publisher(Face, "/face", 1) 
        self.sub_image = self.create_subscription( Image, '/video_frames', self.listener_callback, qos_profile_sensor_data) 
        self.publisher_image = self.create_publisher(Image, '/face_frames', 10)  
        self.start = time.time()
        self.base_frame = None
        self.k = 0.55
        self.face_cascade = cv2.CascadeClassifier('/root/ws/src/face_recognition/face_recognit/cfg/haar_cascade.xml')

    @torch.no_grad()
    def inference(self, weight, name = 'r50', img = None):
        if img is None:
            img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        else:
            try:
                img = cv2.imread(img)
                img = cv2.resize(img, (112, 112))
            except:
                img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        net = get_model(name, fp16 = True)
        net.load_state_dict(torch.load(weight, map_location=torch.device('cpu')), strict=False)
        net.eval()
        feat = net(img).numpy()
        return feat

    def detection_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        self.face.face_detections = False
        for (x,y,w,h) in faces:
            frame = frame[y:y+h, x:x+w]
            self.face.face_detections = True
        #print(frame.shape)
        self.face.process = 'scan'
        self.publisher_image.publish(self.br.cv2_to_imgmsg(frame))
        return frame

    def recognition(self, data):
        frame = cv2.resize(data, (112, 112))
        end = time.time() - self.start
        if end <= 3:
            feat1 = self.inference('/root/work/insightface/recognition/arcface_torch/ms1mv3_arcface_r50/backbone.pth', 'r50', frame)   
        #feat1 = self.inference('/root/work/insightface/recognition/arcface_torch/ms1mv3_arcface_r50/backbone.pth', 'r50', frame)
        #print(end)
        if end >3 and end <5:
            feat1 = self.inference('/root/work/insightface/recognition/arcface_torch/ms1mv3_arcface_r50/backbone.pth', 'r50', frame)    
            self.base_frame = feat1
        feat2 = self.inference('/root/work/insightface/recognition/arcface_torch/ms1mv3_arcface_r50/backbone.pth', 'r50', frame)
        if end > 5:
            feat1 = self.base_frame
            cos_close = np.dot(feat1[0], feat2[0]) / (np.linalg.norm(feat1[0]) * np.linalg.norm(feat2[0]))
            print('cos_close ' , cos_close)
            if end > 5:
                if self.k < cos_close:
                    self.face.face_recognition = True
                    print('I know him')
                else:
                    self.face.face_recognition = False
                    print('dangered stranger')
        self.face.number_person = 1
    
    def listener_callback(self, data):
        self.face = Face()
        current_frame = self.br.imgmsg_to_cv2(data)
        current_frame = self.detection_face(current_frame)
        self.recognition(current_frame)
        self.test_pub.publish(self.face)
        #self.get_logger().info('Publishing: "%s"' % self.msg)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = FaceRecognition()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
