class Config:
    DEVICE = "cuda"
    BATCH_SIZE = 64
    IMG_SIZE = 240
    FOLD = 0
    NUM_CLASSES = 58
    FEATURE_EXTRACTING = False
    PRETRAINED = True
    EPOCHS = 100
    INPUT_PATH = "../input/df_input.csv"
    LR = 1.02e-4
    BEST_LOSS_MODEL = "outputs/models/best_loss_{}.bin"
    BEST_ACC_MODEL = "outputs/models/best_accsss_{}.bin"
