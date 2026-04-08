#!/bin/bash
HF_HOME=/root/.hf_home

echo '=== Waiting for S3 to finish ==='
while pgrep -f 'strategy 3' > /dev/null; do sleep 60; done
echo 'S3 done!'

echo '=== Starting S2 ==='
python3 -u run_allam_cuda.py --strategy 2 --start 0 --end 299 > allam_s2_fixed.log 2>&1
echo 'S2 done!'

echo '=== Running analysis ==='
jupyter nbconvert --to notebook --execute analyze_results_with_allam.ipynb --output analyze_results_with_allam_done.ipynb 2>&1 | tee analysis.log
echo 'ALL DONE!'
