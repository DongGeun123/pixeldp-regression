# Copyright (C) 2018 Vaggelis Atlidakis <vatlidak@cs.columbia.edu> et
# Mathias Lecuyer <mathias@cs.columbia.edu>
#
# Script to eval attack results on PixelDP
#
import tensorflow as tf
import numpy as np
import math
import json
import os, sys, time
from multiprocessing import Pool

import models
import models.params
import attacks.utils
import datasets
from models.utils import robustness

import attacks.params
from attacks import train_attack
from flags import FLAGS

max_batch_size = {
    'madry':            250,
    'pixeldp_resnet':   250,
    'pixeldp_cnn':      1000,
    'inception_model':  160
}
def evaluate_one(dataset, model_class, model_params, attack_class,
                 attack_params, dir_name=None, compute_robustness=True,
                 dev='/cpu:0'):

    gpu = int(dev.split(":")[1]) + FLAGS.min_gpu_number
    gpu = gpu % 16  # for 16 GPUs exps
    dev = "{}:{}".format(dev.split(":")[0], gpu)

    print("Evaluating attack on dev:{}".format(dev), "\n", attack_params)
    with tf.device(dev):
        if dir_name == None:
            dir_name = FLAGS.models_dir

        model_dir  = os.path.join(dir_name, models.params.name_from_params(model_class, model_params))
        attack_dir = os.path.join(model_dir, 'attack_results',
                attacks.params.name_from_params(attack_params))

        # if results are in place, don't redo
        result_path = os.path.join(attack_dir, "eval_data.json")
        if os.path.exists(result_path):
            print("Path: {} exists -- skipping!!!".format(result_path))
            return

        if dataset == None:
            dataset = FLAGS.dataset

        tot_batch_size_atk = train_attack.max_batch_size[models.name_from_module(model_class)]
        tot_batch_size     = max_batch_size[models.name_from_module(model_class)]
        # Some book keeping to maximize the GPU usage depending on the attack
        # requirement.
        images_per_batch_attack = min(
                attack_params.num_examples,
                attack_class.Attack.image_num_per_batch_train(
                    tot_batch_size_atk, attack_params))
        images_per_batch_eval   = min(
                attack_params.num_examples,
                attack_class.Attack.image_num_per_batch_eval(
                    tot_batch_size, attack_params))
        batch_size = min(images_per_batch_attack, images_per_batch_eval)

        image_placeholder = tf.placeholder(tf.float32,
                [batch_size, model_params.image_size,
                 model_params.image_size, model_params.n_channels])
        label_placeholder = tf.placeholder(tf.int32,
                [batch_size, model_params.num_classes])

        model_params = models.params.update(model_params, 'batch_size',
                batch_size)
        model_params = models.params.update(model_params, 'n_draws',
                attack_params.n_draws_eval)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = str(dev.split(":")[-1])
        sess = tf.Session(config=config)

        model = model_class.Model(model_params, image_placeholder,
                label_placeholder, 'eval')
        model.build_graph()
        saver = tf.train.Saver()

        with sess:
            tf.train.start_queue_runners(sess)
            coord = tf.train.Coordinator()

            summary_writer = tf.summary.FileWriter(model_dir)
            try:
                ckpt_state = tf.train.get_checkpoint_state(model_dir)
            except tf.errors.OutOfRangeError as e:
                print('Cannot restore checkpoint: ', e)
                return
            if not (ckpt_state and ckpt_state.model_checkpoint_path):
                print('\n\n\n\t *** No model to eval yet at: {}\n\n\n'.\
                        format(model_dir))
                return
            print('Loading checkpoint ', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)

            ops = model.predictions  # results of the softmax layer

            clean_preds = []
            defense_preds = []
            counters = []

            if model.noise_scale != None:
                args = {model.noise_scale: 1.0}
            else:
                args = {}

            data = {}

            num_iter = int(math.ceil(attack_params.num_examples / images_per_batch_attack))
            intra_batch_num_iter = int(math.ceil(images_per_batch_attack / batch_size))
            loss = 0.0
            for step in range(0, num_iter):
                print("Evaluate:: Starting step {}/{}".format(step+1, num_iter))
                adv_norm = np.zeros(
                        [images_per_batch_attack, attack_params.restarts])

                for restart in range(0, attack_params.restarts):
                    print("Evaluate:: Starting restart {}/{}".format(
                        restart+1, attack_params.restarts))
                    # Naming is advbatch-1-r-1, advbatch-2-r-1, advbatch-1-r-2 ...
                    inputs, adv_inputs, labs, adv_labs = attacks.utils.load_batch(
                        attack_dir, step + 1, restart + 1)

                    if attack_params.attack_norm == 'l2':
                        norm_ord = 2
                    elif attack_params.attack_norm == 'l_inf':
                        norm_ord = np.inf
                    else:
                        raise ValueError("Attack norm not supported")

                    s = inputs.shape
                    adv_norm_restart = np.linalg.norm(
                            np.reshape(inputs, (s[0], -1)) -  \
                                    np.reshape(adv_inputs, (s[0], -1)),
                            ord=norm_ord,
                            axis=1
                    )
                    adv_norm[:,restart] = adv_norm_restart

                    for intra_batch_step in range(0, intra_batch_num_iter):
                        batch_i_start = intra_batch_step       * batch_size
                        batch_i_end   = min((intra_batch_step + 1) * batch_size,
                                images_per_batch_attack)

                        image_batch     = inputs[batch_i_start:batch_i_end]
                        adv_image_batch = adv_inputs[batch_i_start:batch_i_end]
                        label_batch     = labs[batch_i_start:batch_i_end]

                        # Handle end of batch with wrong size
                        true_batch_size = image_batch.shape[0]
                        if true_batch_size < batch_size:
                            pad_size = batch_size - true_batch_size
                            image_batch = np.pad(image_batch,
                                    [(0, pad_size), (0,0), (0,0), (0,0)],
                                    'constant')
                            adv_image_batch = np.pad(adv_image_batch,
                                    [(0, pad_size), (0,0), (0,0), (0,0)],
                                    'constant')
                            label_batch = np.pad(label_batch,
                                    [(0, pad_size), (0,0)],
                                    'constant')

                        # Predictions on the original image: only on one restart
                        if restart == 0:
                            args[image_placeholder] = image_batch
                            args[label_placeholder] = label_batch
                            softmax = sess.run(ops, args)

                        # Predictions on the adversarial image for current
                        # restart
                        args[image_placeholder] = adv_image_batch
                        args[label_placeholder] = label_batch

                        #softmax = sess.run(ops, args)

                        mse = tf.losses.mean_squared_error(labels=label_batch, predictions=softmax)
                        loss += mse.eval()
        # save tensor in json file


        sensitivity_multiplier = 1.0
        try:
            with open(model_dir + "/sensitivity_multiplier.json") as f:
                sensitivity_multiplier = float(json.loads(f.read())[0])
        except Exception:
            print("Missing Mulltiplier")
            pass
        data['sensitivity_mult_used'] = sensitivity_multiplier
        data['mse'] = loss / (num_iter * attack_params.restarts)

        # Log eval data
        with open(result_path, 'w') as f:
            f.write(json.dumps(data))

        return data

