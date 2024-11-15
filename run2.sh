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
ns_values=(0.1 1)
ts_values=(50 100 500)
ds_values=(10 50 100)

counter=1

# Loop over each combination of ns, ts, and ds
for ns in "${ns_values[@]}"; do
    for ts in "${ts_values[@]}"; do
        for ds in "${ds_values[@]}"; do
            echo "Running with ns=${ns}, ts=${ts}, ds=${ds}"
            
            OUTPUT_DIR="${1}/results_${counter}"

            python run.py \
                --k 20 \
                --epochs 5 \
                --ga 10 \
                --sa 0 \
                --ds $ds \
                --targetID $TARGET \
                --ts $ts \
                --ns $ns \
                $IMAGE_ARG \
                --out_dir $OUTPUT_DIR
            
            ((counter++))
        done
    done
done

