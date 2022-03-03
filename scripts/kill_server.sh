#!/bin/bash

person_ID=`ps -ef| grep person|grep -v "$0"|grep -v grep | awk '{print $2}'`
echo person_ID:$person_ID

helmet_ID=`ps -ef| grep helmet|grep -v "$0"|grep -v grep | awk '{print $2}'`
echo helmet_ID:$helmet_ID


echo "======"
if [ -z "$person_ID" ];then
    echo "server person进程不存在"
else
    echo "kill:" $person_ID
    echo 'ubuntu'| sudo -S kill -9 $person_ID
fi

echo "======"

if [ -z "$helmet_ID" ];then
    echo "server person进程不存在"
else
    echo "kill:" $helmet_ID
    echo 'ubuntu'| sudo -S kill -9 $helmet_ID
fi
echo "======"
