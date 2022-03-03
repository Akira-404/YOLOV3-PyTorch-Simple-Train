#!/bin/bash

root="/home/ubuntu/PycharmProjects/YOLOV3-PyTorch-Simple-Train/"

person_ID=`ps -ef| grep person|grep -v "$0"|grep -v grep | awk '{print $2}'`

echo "======"
if [ -z "$person_ID" ];then
  echo "server person is not found."
  echo "nohup server_person."
  #nohup python $root"server_person.py" > $root"logs/person/person_nohup.log" 2>&1 &
  nohup python $root"server_person.py" > $root"logs/person/person_nohup.log" &
#  python /home/ubuntu/PycharmProjects/YOLOV3-PyTorch-Simple-Train/server_person.py
  sleep 5
  echo "server_person running..."
else
  echo "person is found:"$person_ID
fi

echo "======"

#helmet_ID=`ps -ef| grep helmet|grep -v "$0"|grep -v grep | awk '{print $2}'`
#
#if [ -z "$person_ID" ];then
#  echo "server person is not found."
#  echo "nohup server_person."
#  nohup python $root"server_helmet.py" > $root"logs/helmet/helmet_nohup.log" 2>&1 &
##  python /home/ubuntu/PycharmProjects/YOLOV3-PyTorch-Simple-Train/server_helmet.py
#  sleep 5
#  echo "server_person running..."
#else
#  echo "person is found:"$helmet_ID
#fi


echo "======"
