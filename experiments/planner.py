import inspect
import os
import sys
from time import time

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.planner import _plot, PlanningNetworkMP, Loss, unpack_data

from argparse import ArgumentParser

from tqdm import tqdm

from utils.execution import ExperimentHandler, LoadFromFile

tf.random.set_seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, data)
            pbar.update(batch_size)

def main(args):
    # 1. Get datasets
    train_ds, train_size = scenarios.planning_dataset(args.scenario_path)
    val_ds, val_size = scenarios.planning_dataset(args.scenario_path.replace("train", "val"))

    val_ds = val_ds \
        .batch(args.batch_size) \
        .prefetch(args.batch_size)

    # 2. Define model
    N = 3
    model = PlanningNetworkMP(N)
    loss = Loss(N)

    # 3. Optimization

    optimizer = tf.keras.optimizers.Adam(args.eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        # workaround for tf problems with shuffling
        dataset_epoch = train_ds.shuffle(10000)
        #dataset_epoch = train_ds.shuffle(train_size)
        dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)

        # 5.1. Training Loop
        experiment_handler.log_training()
        acc = []
        for i, data in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape(persistent=True) as tape:
                output, pts, p = model(data, None, training=True)
                model_loss, invalid_loss, curvature_loss, overshoot_loss, total_curvature_loss, x_path, y_path, th_path, curvature = loss(output, data)
            grads = tape.gradient(model_loss, model.trainable_variables)
            #for g in grads:
                #print(tf.reduce_max(tf.abs(g)))
            #grads = [tf.clip_by_value(g, -1., 1) for g in grads]
            #print(model_loss)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 5.1.3 Calculate statistics
            t = tf.reduce_mean(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
            s = tf.reduce_mean(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))
            acc.append(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))

            # 5.1.4 Save logs for particular interval
            with tf.summary.record_if(train_step % args.log_interval == 0):
                tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
                tf.summary.scalar('metrics/invalid_loss', tf.reduce_mean(invalid_loss), step=train_step)
                tf.summary.scalar('metrics/curvature_loss', tf.reduce_mean(curvature_loss), step=train_step)
                tf.summary.scalar('metrics/total_curvature_loss', tf.reduce_mean(total_curvature_loss), step=train_step)
                tf.summary.scalar('metrics/collision_free_paths', t, step=train_step)
                tf.summary.scalar('metrics/ideal_paths', s, step=train_step)

            # 5.1.5 Update meta variables
            if train_step % 100 == 0:
                _plot(x_path, y_path, th_path, data, train_step, output)
            #_plot(x_path, y_path, th_path, data, train_step, output)
            #if train_step > 100: assert False
            train_step += 1
        epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))

        # 5.1.6 Take statistics over epoch
        with tf.summary.record_if(True):
            tf.summary.scalar('epoch/good_paths', epoch_accuracy, step=epoch)

        #if epoch_accuracy > best_accuracy:
        #    experiment_handler.save_best()
        #    best_accuracy = epoch_accuracy
        #experiment_handler.save_last()
        #continue

        # 5.2. Validation Loop
        experiment_handler.log_validation()
        acc = []
        for i, data in _ds('Validation', val_ds, val_size, epoch, args.batch_size):
            # 5.2.1 Make inference of the model for validation and calculate losses
            output, pts, p = model(data, None, training=False)
            model_loss, invalid_loss, curvature_loss, overshoot_loss, total_curvature_loss, x_path, y_path, th_path, curvature = loss(output, data)

            t = tf.reduce_mean(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
            s = tf.reduce_mean(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))
            acc.append(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))

            # 5.2.3 Print logs for particular interval
            with tf.summary.record_if(val_step % args.log_interval == 0):
                tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
                tf.summary.scalar('metrics/invalid_loss', tf.reduce_mean(invalid_loss), step=val_step)
                tf.summary.scalar('metrics/curvature_loss', tf.reduce_mean(curvature_loss), step=val_step)
                tf.summary.scalar('metrics/total_curvature_loss', tf.reduce_mean(total_curvature_loss), step=val_step)
                tf.summary.scalar('metrics/collision_free_paths', t, step=val_step)
                tf.summary.scalar('metrics/ideal_paths', s, step=val_step)

            # 5.2.4 Update meta variables
            val_step += 1

        epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))

        # 5.2.5 Take statistics over epoch
        with tf.summary.record_if(True):
            tf.summary.scalar('epoch/good_paths', epoch_accuracy, step=epoch)

        # 5.3 Save last and best
        if epoch_accuracy > best_accuracy:
            experiment_handler.save_best()
            best_accuracy = epoch_accuracy
        #experiment_handler.save_last()

        experiment_handler.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config-file', action=LoadFromFile, type=open)
    parser.add_argument('--scenario-path', type=str)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--out-name', type=str)
    parser.add_argument('--eta', type=float, default=5e-4)
    args, _ = parser.parse_known_args()
    main(args)
