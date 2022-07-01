#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 00:28:30 2020

@author: mason
"""

''' import libraries '''
import time
import numpy as np

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import sys
import signal

def signal_handler(signal, frame): # ctrl + c -> exit program
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


''' class '''

class path_pub():
    def __init__(self):
        rospy.init_node('path_pubb', anonymous=True)
        self.frame_id = rospy.get_param("~frame_id", 'map')
        self.topic_name = rospy.get_param("~topic_name", '/mavros/local_position/pose')
        self.out_topic_name = rospy.get_param("~out_topic_name", 'drone_path')
        self.append_rate = rospy.get_param("~append_rate", 5)
        self.my_pose = rospy.Subscriber(self.topic_name, PoseStamped, self.posecallback)
        self.my_path_pub = rospy.Publisher(self.out_topic_name, Path, queue_size=1)

        self.rate = rospy.Rate(self.append_rate)
        self.my_path = Path()
        self.my_path.header.frame_id = self.frame_id
        self.check = 0

    def posecallback(self, msg):
        self.pose = msg
        self.check = 1

''' main '''
path_pub_class = path_pub()

if __name__ == '__main__':
    while 1:
        try:
            if path_pub_class.check == 1:
                path_pub_class.pose.header.frame_id = path_pub_class.frame_id
                path_pub_class.my_path.poses.append(path_pub_class.pose)
                path_pub_class.my_path.header.stamp = rospy.Time.now()
                path_pub_class.my_path_pub.publish(path_pub_class.my_path)

                path_pub_class.rate.sleep()
        except (rospy.ROSInterruptException, SystemExit, KeyboardInterrupt) :
            sys.exit(0)
        # except:
        #     print("exception")
        #     pass