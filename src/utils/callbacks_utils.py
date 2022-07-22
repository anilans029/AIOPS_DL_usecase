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
    joblib.dump(tensorboard_callback, tb_callback_filepath)

    lg.info(f"tb_callback is being saved at {tb_callback_filepath}")

def create_and_save_checkpoint_callback(CALLBACKS_DIR_path,CHECKPOINT_DIR_path):
    checkpoint_file_path = os.path.join(CHECKPOINT_DIR_path, 'ckpt_model.h5')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file_path,
        save_best_only=True,
        verbose = 1)
    
    ckpt_callback_filepath = os.path.join(CALLBACKS_DIR_path, "checkpoint_cb.cb")
    joblib.dump(checkpoint_callback, ckpt_callback_filepath)
    lg.info(f"tensorboard callback is being saved at {ckpt_callback_filepath}")


def get_callbacks(callback_dir_path):
    callbacks_path = [os.path.join(callback_dir_path, binfile) for binfile in os.listdir(callback_dir_path) if binfile.endswith(".cb")]

    callbacks = [
        joblib.load(path) for path in callbacks_path
    ]
    lg.info(f"callbacks are loaded from the path {callback_dir_path}")
    return callbacks