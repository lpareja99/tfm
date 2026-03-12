DATA_ROOT = 'data/2026-01-19-defect_dataset'
LABEL_DIR = 'labels_cracks'
TRAIN_SPLIT = 'train.txt'
VAL_SPLIT = 'val.txt'
TEST_SPLIT = 'test.txt'

ID2LABEL = {0: "bg", 1: "cracks", 2: "cracks_alligator", 3: "cracks_severe"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

BASE_MODEL = "facebook/mask2former-swin-tiny-ade-semantic"
IMAGE_SIZE = {"height": 512, "width": 512}

WORK_DIR = "./work_dirs/basic_round_4"
