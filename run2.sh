#!/bin/bash

# Check if an argument is provided
if [ -n "$2" ]; then
    IMAGE_ARG="--image $2"
else
    IMAGE_ARG=""
fi


# Just the name e.g. "results"
if [ -n "$3" ]; then
    TARGET=$3
else
    TARGET="1" # goldfish
fi

#!/bin/bash

# Lists to iterate over
# ns_values=(0 0.1 1)
# ts_values=(500 1000 1500)
# ds_values=(10 100 500)
# ga = 1000

# ga = 10
# ns_values=(0.1)
# ts_values=(50 100 500)
# ds_values=(50 100 500)
# gs_values=(10 100 1000)

# Tuples (gs, ns, ts, ds)
tuples_list=(
    # "10 0.1 100 100"
    # "10 10 100 100" # brisque as prior
    "10 0.1 1 500" # dino only
    "1000 0.1 1 500" # dino + perp
    "10 10 1 500" # dino + brisque
    # "10 0.1 100 1" # label only
    # "1000 0.1 100 1" # label + perp
    # "10 10 100 1" # label + brisque
    # "10 10 1 1" # brisque
)

counter=1

# Loop over each tuple
for tuple in "${tuples_list[@]}"; do
    # Read the values of gs, ns, ts, and ds from the tuple
    read -r gs ns ts ds <<< "$tuple"

    echo "Running with gs=${gs}, ns=${ns}, ts=${ts}, ds=${ds}"

    OUTPUT_DIR="${1}/results_${counter}"

    python run.py \
        --k 5 \
        --epochs 50 \
        --ga $gs \
        --sa 0 \
        --ds $ds \
        --targetID $TARGET \
        --ts $ts \
        --ns $ns \
        $IMAGE_ARG \
        --out_dir $OUTPUT_DIR

    ((counter++))
done