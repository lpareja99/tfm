# This imports EVERYTHING from your current config
_base_ = ['./initial_test_flowity.py'] 

# Redirect the output so it doesn't overwrite your other run
work_dir = './work_dirs/flowity_no_background'

# We assume background is class 0
val_evaluator = dict(ignore_index=0) 
test_evaluator = dict(ignore_index=0)
