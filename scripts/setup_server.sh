#!/bin/bash

person_id=`ps -ef| grep yolov3|grep -v "$0"|grep -v grep | awk '{print $2}'`
root=~/PycharmProjects/yolov3/YOLOV3-PyTorch-Simple-Train
echo "======"
if [ -z "$person_id" ];then
  echo "Server Person is not found."
  echo "Starting Server:Person..."
    nohup $HOME/anaconda3/envs/torch/bin/python $root/server_person.py > $root/logs/setup.log 2>&1 &
  sleep 5
  echo "Server:Person is running"
else
  echo "Server:Person is found:"$person_id
fi

echo "======"

helmet_id=`ps -ef| grep helmet|grep -v "$0"|grep -v grep | awk '{print $2}'`

if [ -z "$person_id" ];then
  echo "Server:Helmet is not found."
  echo "Starting Server:Helmet..."
    nohup $HOME/anaconda3/envs/torch/bin/python $root/server_helmet.py > /dev/null 2>&1 &
  sleep 5
  echo "Server:Helmet is running"
else
  echo "Server:Helmet is found:"$helmet_id
fi


echo "======"
