import os
import keras.api._v2.keras as keras
import joblib
import logging as lg
from src.utils.all_utils import get_timestamp




def create_and_save_tf_callback(CALLBACKS_DIR_path, tensorboard_log_dir_path):
    unique_name = get_timestamp("tb_logs")
    tb_running_log_dir = os.path.join(tensorboard_log_dir_path, unique_name)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir= tb_running_log_dir)

    tb_callback_filepath = os.path.join(CALLBACKS_DIR_path,"tensorboard_cb.cb")
    joblib.dump(tensorboard_callback)

    lg.info(f"tb_callback is being saved at {tb_callback_filepath}")

def create_and_save_checkpoint_callback(CALLBACKS_DIR_path,CHECKPOINT_DIR_path):
    pass