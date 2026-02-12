#!/bin/bash

# 1. Dynamically find where the mmsegmentation library is installed
MMSEG_PATH=$(python -c "import mmseg; import os; print(os.path.dirname(mmseg.__file__))")

echo "---> Found MMSegmentation at: $MMSEG_PATH"

# 2. Define the script path (internal to the pip-installed library)
DEMO_SCRIPT="$MMSEG_PATH/.python-scripts/demo/image_demo.py"

# 3. Ensure the output directory exists
mkdir -p ./work_dirs

# 4. Run the demo using library files and the official weight URL
python "$DEMO_SCRIPT" \
    "$MMSEG_PATH/.python-scripts/demo/demo.png" \
    "$MMSEG_PATH/.python-scripts/configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py" \
    https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512_20221203_234230-7d64e5dd.pth \
    --device cuda:0 \
    --out-file ./work_dirs/library_test_result.jpg

echo "---> Test complete. Check ./work_dirs/library_test_result.jpg"