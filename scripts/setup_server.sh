#!/bin/bash

person_id=`ps -ef| grep person|grep -v "$0"|grep -v grep | awk '{print $2}'`
root=~/PycharmProjects/YOLOV3-PyTorch-Simple-Train/

echo "======"
if [ -z "$person_id" ];then
  echo "Server Person is not found."
  echo "Starting Server:Person..."
#  nohup python $root"server_person.py" > $root"logs/person/person_nohup.log" 2>&1 &
    nohup python $root"server_person.py" > /dev/null 2>&1 &
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
#  nohup python $root"server_helmet.py" > $root"logs/helmet/helmet_nohup.log" 2>&1 &
    nohup python $root"server_helmet.py" > /dev/null 2>&1 &
  sleep 5
  echo "Server:Helmet is running"
else
  echo "Server:Helmet is found:"$helmet_id
fi


echo "======"
