#!/bin/bash

# ImageNet -> distinct class (bee killer -> goldfish)
bash run.sh "results"

# Gray image -> hummingbird
bash run.sh "gray_results" "gray_image.png" "1"

bash run.sh "noisy_results" "noisy_image.png" "1"

# Hummingbird (image not from ImNet) -> bee killer (similar class)
bash run.sh "h_results" "hummingbird.jpg" "92"

# 92 perturbed to be 94 back to 92
bash run.sh "adv_results" "adv_image.png" "1"