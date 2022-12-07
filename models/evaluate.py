# Copyright 2016 The Pixeldp Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Based on https://github.com/tensorflow/models/tree/master/research/resnet

"""ResNet Train/Eval module.
"""
import time
import six
import sys
import os
import json

import datasets
import numpy as np
import models.params
from models import pixeldp_cnn, pixeldp_resnet
import tensorflow as tf

from models.utils import robustness

from flags import FLAGS

def evaluate(hps, model, dataset=None, dir_name=None, rerun=False,
             compute_robustness=True, dev='/cpu:0'):
    """Evaluate the ResNet and log prediction counters to compute
    sensitivity."""

    # Trick to start from arbitrary GPU  number
    gpu = int(dev.split(":")[1]) + FLAGS.min_gpu_number
    if gpu >= 16:
        gpu -= 16
    dev = "{}:{}".format(dev.split(":")[0], gpu)

    print("Evaluating model on dev:{}".format(dev))
    with tf.device(dev):
        if dir_name == None:
            dir_name = FLAGS.models_dir

        #dir_name = os.path.join(dir_name, models.params.name_from_params(model, hps))

        if os.path.isfile(dir_name + "/eval_data.json") and not rerun:
            print("Skip eval of:{}".format(dir_name))
            # run only new models
            return

        if dataset == None:
            dataset = FLAGS.dataset

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = str(dev.split(":")[-1])
        sess = tf.Session(config=config)

        images, labels = datasets.build_input(
            dataset,
            FLAGS.data_path,
            hps.batch_size,
            hps.image_standardization,
            'eval'
        )

        tf.train.start_queue_runners(sess)
        model = model.Model(hps, images, labels, 'eval')
        model.build_graph()


        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(dir_name)

        try:
            dir_name +='/pixeldp_resnet_attack_norm_l2_size_0.1_1_prenoise_layers_sensitivity_l2_scheme_bound_activation_postnoise'
            ckpt_state = tf.train.get_checkpoint_state(dir_name)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        # Make predictions on the dataset, keep the label distribution
        data = {
            'mean_square_error': []
        }
        total_loss = 0
        eval_data_size   = hps.eval_data_size
        eval_batch_size  = hps.batch_size
        eval_batch_count = int(eval_data_size / eval_batch_size)
        for i in six.moves.range(eval_batch_count):
            if model.noise_scale == None:
                args = {}  # For Madry and inception
            else:
                args = {model.noise_scale: 1.0}
            (loss, predictions, truth) = sess.run([
                    model.cost,
                    model.predictions,
                    model.labels, ], args)
            total_loss += loss
            print("Done: {}/{}".format(eval_batch_size*i, eval_data_size))
            # truth = np.argmax(truth, axis=1)[:hps.batch_size]
            # data['argmax_sum'] += predictions.tolist()
        # For Parseval, get true sensitivity, use to rescale the actual attack
        # bound as the nosie assumes this to be 1 but often it is not.
        # Parseval updates usually have a sensitivity higher than 1
        # despite the projection: we need to rescale when computing
        # sensitivity.
        if model.pre_noise_sensitivity() == None:
            sensitivity_multiplier = None
        else:
            sensitivity_multiplier = float(sess.run(
                model.pre_noise_sensitivity(),
                {model.noise_scale: 1.0}
            ))
        with open(dir_name + "/sensitivity_multiplier.json", 'w') as f:
            d = [sensitivity_multiplier]
            f.write(json.dumps(d))

        # Compute robustness and add it to the eval data.
        # if compute_robustness:  # This is used mostly to avoid errors on non pixeldp DNNs
        #     dp_mechs = {
        #         'l2': 'gaussian',
        #         'l1': 'laplace'
        #     }
        #     robustness_size= [robustness.robustness_size_argmax(
        #         counts=x,
        #         eta=hps.robustness_confidence_proba,
        #         dp_attack_size=hps.attack_norm_bound,
        #         dp_epsilon=hps.dp_epsilon,
        #         dp_delta=hps.dp_delta,
        #         dp_mechanism=dp_mechs[hps.sensitivity_norm]
        #         ) / sensitivity_multiplier for x in data['argmax_sum']] #argmax_sum : hps.batch_size, hps.num_classes]
        #     data['robustness_size'] = robustness_size
        #     data['mean_loss'] =mean_loss
        # # Log eval data
        # with open(dir_name + "/eval_data.json", 'w') as f:
        #     f.write(json.dumps(data))

        # Print stuff
        mean_square_error = total_loss / eval_batch_count
        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='mean_square_loss', simple_value=mean_square_error)
        data['mean_square_error'] = mean_square_error
        with open(dir_name + "/eval_data.json", 'w') as f:
            f.write(json.dumps(data))
        #summary_writer.add_summary(precision_summ, train_step)
        #  summary_writer.add_summary(summaries, train_step)
        tf.logging.info('mean_square_error: %.3f' %(mean_square_error))
        summary_writer.flush()

