#!/bin/bash

# ImageNet -> distinct class (bee killer -> goldfish)
bash run.sh "results"

# Gray image -> hummingbird
bash run.sh "gray_results" "base_images/gray_image.png" "1"

bash run.sh "noisy_results" "base_images/noisy_image.png" "1"

# Hummingbird (image not from ImNet) -> bee killer (similar class)
bash run.sh "h_results" "base_images/hummingbird.jpg" "92"

# 92 perturbed to be 94 back to 92
bash run.sh "adv_results" "base_images/adv_image.png" "1"