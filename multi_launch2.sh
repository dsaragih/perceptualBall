#!/bin/bash

# ImageNet -> distinct class (bee killer -> goldfish)
bash run2.sh "results_2"

# Gray image -> hummingbird
bash run2.sh "gray_results_2" "base_images/gray_image.png" "1"

bash run2.sh "noisy_results_2" "base_images/noisy_image.png" "1"

# Hummingbird (image not from ImNet) -> bee killer (similar class)
bash run2.sh "h_results_2" "base_images/hummingbird.jpg" "92"

# 92 perturbed to be 94 back to 92
bash run2.sh "adv_results_2" "base_images/adv_image.png" "1"