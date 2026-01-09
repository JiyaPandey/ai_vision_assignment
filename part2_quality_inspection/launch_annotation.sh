#!/bin/bash
# Script to launch LabelImg with correct configuration

cd ~/ai_vision_assignment/part2_quality_inspection/data

# Launch labelImg
labelImg images/DeepPCB/ classes.txt
