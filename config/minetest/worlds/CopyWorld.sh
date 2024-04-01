#!/bin/bash

echo "hello world"

rm -r ./config/minetest/worlds/ToCopy
mkdir ./config/minetest/worlds/ToCopy

if [[ -z $@ ]]; then
   num=5
else
   num=$@
fi

echo $num

for ((i = 0 ; i < $num ; i++ ))
do
   #echo "Welcome $i times"
   cp -r ./config/minetest/worlds/World_Train ./config/minetest/worlds/ToCopy 
   mv ./config/minetest/worlds/ToCopy/World_Train ./config/minetest/worlds/ToCopy/$i
   chmod go+w ./config/minetest/worlds/ToCopy/$i

done

echo "Complete!"

