# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for training Tensorflow models on the YouTube-8M dataset."""
import glob
import json
import os
import time
import numpy as np

import eval_util
import losses
import frame_level_models
import nextvlad
import video_level_models
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
import utils
from eval import evaluate
import os


FLAGS = flags.FLAGS



def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def validate_class_name(flag_value, category, modules, expected_superclass):
    """Checks that the given string matches a class of the expected type.

    Args:
      flag_value: A string naming the class to instantiate.
      category: A string used further describe the class in error messages
                (e.g. 'model', 'reader', 'loss').
      modules: A list of modules to search for the given class.
      expected_superclass: A class that the given class should inherit from.

    Raises:
      FlagsError: If the given class could not be found or if the first class
      found with that name doesn't inherit from the expected superclass.

    Returns:
      True if a class was found that matches the given constraints.
    """
    candidates = [getattr(module, flag_value, None) for module in modules]
    for candidate in candidates:
        if not candidate:
            continue
        if not issubclass(candidate, expected_superclass):
            raise flags.FlagsError("%s '%s' doesn't inherit from %s." %
                                   (category, flag_value,
                                    expected_superclass.__name__))
        return True
    raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))


def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
    """Creates the section of the graph which reads the training data.

    Args:
      reader: A class which parses the training data.
      data_pattern: A 'glob' style path to the data files.
      batch_size: How many examples to process at a time.
      num_epochs: How many passes to make over the training data. Set to 'None'
                  to run indefinitely.
      num_readers: How many I/O threads to use.

    Returns:
      A tuple containing the features tensor, labels tensor, and optionally a
      tensor containing the number of frames per video. The exact dimensions
      depend on the reader being used.

    Raises:
      IOError: If no files matching the given pattern were found.
    """
    logging.info("Using batch size of " + str(batch_size) + " for training.")

    with tf.name_scope("train_input"):
        files = gfile.Glob(data_pattern)  #其实就是训练集的路径？  data_pattern=/media/linrongc/dream/data/yt8m/2/frame/train/train*.tfrecord
        #A list of strings containing filenames that match the given pattern(s).
        #print('training_files=',files)
        #training_files= ['/home/disk3/a_zhongzhanhui/yt8m_dataset/train_all/train0848.tfrecord', '/home/disk3/a_zhongzhanhui/yt8m_dataset/train_all/train2552.tfrecord',
        if not files:
            raise IOError("Unable to find training files. data_pattern='" +
                          data_pattern + "'.")
        logging.info("Number of training files: %s.", str(len(files)))

        filename_queue = tf.train.string_input_producer(
            files, num_epochs=num_epochs, shuffle=True)  #num_epochs=5，就是说在这些tfrecord训练集上训练5遍就不再重复将他们加进队列里了
        print('filename_queue=', filename_queue)
        training_data = [
            reader.prepare_reader(filename_queue) for _ in range(num_readers)
        ]
        # print('training_data=', training_data)
        return tf.train.shuffle_batch_join( #只是把training_data这些tensor组成batch而已
            training_data,
            batch_size=batch_size,
            capacity=batch_size * 5,
            min_after_dequeue=batch_size,
            allow_smaller_final_batch=False,
            enqueue_many=True)  #Create batches by randomly shuffling tensors.

def get_latest_checkpoint():
    index_files = glob.glob(os.path.join(FLAGS.train_dir, 'model.ckpt-*.index'))

    # No files
    if not index_files:
        return None

    # Index file path with the maximum step size.
    latest_index_file = sorted(
        [(int(os.path.basename(f).split("-")[-1].split(".")[0]), f)
         for f in index_files])[-1][1]

    # Chop off .index suffix and return
    return latest_index_file[:-6]

def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def build_graph_teacher( model_input, labels_batch, num_frames,
                reader,
                model,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                ):
    #建teacher的图，十分简单，完全无需对输入进行切片

    global_step = tf.Variable(0, trainable=False, name="global_step")

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus = gpus[:FLAGS.num_gpu]
    num_gpus = len(gpus)
    if num_gpus > 0:
        logging.info("Using the following GPUs to train: " + str(gpus))
        num_towers = num_gpus
        device_string = '/gpu:%d'
    else:
        logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = '/cpu:%d'

    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step * batch_size * num_towers,
        learning_rate_decay_examples,
        learning_rate_decay,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate) #可视化在tensorboard上

    optimizer = optimizer_class(learning_rate) #就是Adam

    #这个tower就是用在多gpu并行上的
    tower_inputs = tf.split(model_input, num_towers) #一个tensor的list，每个tensor分配给一个gpu训练
    print('tower_inputs=', tower_inputs)
    tower_labels = tf.split(labels_batch, num_towers)
    tower_num_frames = tf.split(num_frames, num_towers)
    tower_gradients = []
    tower_predictions = []
    tower_representations = []
    tower_label_losses = []
    tower_reg_losses = []
    for i in range(num_towers):
        # For some reason these 'with' statements can't be combined onto the same
        # line. They have to be nested.
        with tf.device(device_string % i):
            with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
                with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus != 1 else "/gpu:0")):
                    representations,result = model.create_model(
                        tower_inputs[i],
                        num_frames=tower_num_frames[i],
                        vocab_size=reader.num_classes,
                        labels=tower_labels[i],
                        )
                    # for variable in slim.get_model_variables():
                    #     tf.summary.histogram(variable.op.name, variable)

                    #这里已经是前向传播完毕了，得到预测结果了呀。  如果要知识蒸馏的话我要在前面保存一下representation(就是下面的logit)，这里算loss

                    logits = result["logits"] #一个num_class维的特征
                    predictions = result["predictions"] #logits经过了sigmoid函数后的probability
                    #指导student训练所需要的是representation和prediction
                    tower_representations.append(representations)
                    tower_predictions.append(predictions)

                    if "loss" in result.keys():
                        label_loss = result["loss"]
                    else:
                        label_loss = label_loss_fn.calculate_loss(predictions, tower_labels[i]) #就是交叉熵
                        if "aux_predictions" in result.keys():
                            for pred in result["aux_predictions"]:
                                label_loss += label_loss_fn.calculate_loss(pred, tower_labels[i])

                    if "regularization_loss" in result.keys():
                        reg_loss = result["regularization_loss"]
                    else:
                        reg_loss = tf.constant(0.0)

                    reg_losses = tf.losses.get_regularization_losses()
                    if reg_losses:
                        reg_loss += tf.add_n(reg_losses)

                    tower_reg_losses.append(reg_loss)

                    # Adds update_ops (e.g., moving average updates in batch normalization) as
                    # a dependency to the train_op.
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    if "update_ops" in result.keys():
                        update_ops += result["update_ops"]
                    if update_ops:
                        with tf.control_dependencies(update_ops):
                            barrier = tf.no_op(name="gradient_barrier")
                            with tf.control_dependencies([barrier]):
                                label_loss = tf.identity(label_loss)

                    tower_label_losses.append(label_loss)

                    # Incorporate the L2 weight penalties etc.
                    final_loss = regularization_penalty * reg_loss + label_loss  #加入了正则化误差提高泛化能力，但这个权重设置为1这么大吗
                    gradients = optimizer.compute_gradients(final_loss,colocate_gradients_with_ops=False) #对loss计算梯度，后面用来做反向传播
                    tower_gradients.append(gradients)
    label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
    tf.summary.scalar("label_loss", label_loss)
    if regularization_penalty != 0:
        reg_loss = tf.reduce_mean(tf.stack(tower_reg_losses))
        tf.summary.scalar("reg_loss", reg_loss)
    merged_gradients = utils.combine_gradients(tower_gradients)

    if clip_gradient_norm > 0: #解决梯度爆炸问题
        with tf.name_scope('clip_grads'):
            merged_gradients = utils.clip_gradient_norms(merged_gradients, clip_gradient_norm)

    train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)

    postfix='_teacher'
    print('postfix=', postfix)

    tf.add_to_collection("global_step"+postfix, global_step)
    tf.add_to_collection("loss"+postfix, label_loss)
    tf.add_to_collection("input_batch"+postfix, model_input)
    tf.add_to_collection("num_frames"+postfix, num_frames)
    tf.add_to_collection("labels"+postfix, tf.cast(labels_batch, tf.float32))
    tf.add_to_collection("train_op"+postfix, train_op)

    tf.add_to_collection("model_input" + postfix, model_input)

    tf.add_to_collection("representations" + postfix, tf.concat(tower_representations, 0))
    tf.add_to_collection("predictions" + postfix, tf.concat(tower_predictions, 0))

    return tf.concat(tower_representations, 0),tf.concat(tower_predictions, 0)




def build_graph_student( model_input, labels_batch, num_frames,
                representations_tch,predictions_tch,
                reader,
                model,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                k_frame=300,
                ):
    #建student的图，需要传参k_frame，并且对input进行切片
    global_step = tf.Variable(0, trainable=False, name="global_step")

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus = gpus[:FLAGS.num_gpu]
    num_gpus = len(gpus)
    if num_gpus > 0:
        logging.info("Using the following GPUs to train: " + str(gpus))
        num_towers = num_gpus
        device_string = '/gpu:%d'
    else:
        logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = '/cpu:%d'

    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step * batch_size * num_towers,
        learning_rate_decay_examples,
        learning_rate_decay,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate) #可视化在tensorboard上

    optimizer = optimizer_class(learning_rate) #就是Adam

    frame_step = int(300 / k_frame)
    model_input = model_input[:, 0:300:frame_step, :]

    #这个tower就是用在多gpu并行上的
    tower_inputs = tf.split(model_input, num_towers) #一个tensor的list，每个tensor分配给一个gpu训练
    print('tower_inputs=', tower_inputs)
    tower_labels = tf.split(labels_batch, num_towers)
    tower_num_frames = tf.split(num_frames, num_towers)

    label_loss_tch= label_loss_fn.calculate_loss(predictions_tch, labels_batch)  # 就是交叉熵


    tower_representations_tch=tf.split(representations_tch, num_towers)
    tower_predictions_tch = tf.split(predictions_tch, num_towers)

    tower_gradients = []
    tower_predictions = []
    tower_representations=[]
    tower_label_losses = []
    tower_rep_losses = []
    tower_pred_losses = []
    tower_reg_losses = []


    for i in range(num_towers):
        # For some reason these 'with' statements can't be combined onto the same
        # line. They have to be nested.
        with tf.device(device_string % i):
            with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
                with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus != 1 else "/gpu:0")):
                    representations,result = model.create_model(
                        tower_inputs[i],
                        num_frames=tower_num_frames[i],
                        vocab_size=reader.num_classes,
                        labels=tower_labels[i],
                        k_frame=k_frame)
                    # for variable in slim.get_model_variables():
                    #     tf.summary.histogram(variable.op.name, variable)

                    #这里已经是前向传播完毕了，得到预测结果了呀。  如果要知识蒸馏的话我要在前面保存一下representation(就是下面的logit)，这里算loss

                    logits = result["logits"] #一个num_class维的特征
                    predictions = result["predictions"] #logits经过了sigmoid函数后的probability

                    tower_representations.append(representations)
                    tower_predictions.append(predictions)

                    if "loss" in result.keys():
                        label_loss = result["loss"]
                    else:
                        label_loss = label_loss_fn.calculate_loss(predictions, tower_labels[i])  # 就是交叉熵
                        if "aux_predictions" in result.keys():
                            for pred in result["aux_predictions"]:
                                label_loss += label_loss_fn.calculate_loss(pred, tower_labels[i])

                    if "regularization_loss" in result.keys():
                        reg_loss = result["regularization_loss"]
                    else:
                        reg_loss = tf.constant(0.0)

                    reg_losses = tf.losses.get_regularization_losses()
                    if reg_losses:
                        reg_loss += tf.add_n(reg_losses)

                    tower_reg_losses.append(reg_loss)

                    # Adds update_ops (e.g., moving average updates in batch normalization) as
                    # a dependency to the train_op.
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    if "update_ops" in result.keys():
                        update_ops += result["update_ops"]
                    if update_ops:
                        with tf.control_dependencies(update_ops):
                            barrier = tf.no_op(name="gradient_barrier")
                            with tf.control_dependencies([barrier]):
                                label_loss = tf.identity(label_loss)

                    tower_label_losses.append(label_loss)


                    #以下两个loss都是取平方差
                    #论文上没有加tf.reduce_mean
                    rep_loss= tf.reduce_sum(tf.square(tower_representations_tch[i]-representations))
                    pred_loss = tf.reduce_sum(tf.square(tower_predictions_tch[i]-predictions))

                    # pred_loss=tf.reduce_sum(predictions * tf.log(predictions) - predictions * tf.log(tower_predictions_tch[i]))

                    tower_rep_losses.append(rep_loss)
                    tower_pred_losses.append(pred_loss)

                    # Incorporate the L2 weight penalties etc.
                    # final_loss = label_loss  + rep_loss + pred_loss
                    # final_loss= 0.8*label_loss  + 0.2* pred_loss +0.01* rep_loss
                    # final_loss =  label_loss
                    # final_loss = 0.7 * label_loss + 0.3* pred_loss
                    # final_loss = 0.8 * label_loss + 0.2 * pred_loss
                    # final_loss = 0.9 * label_loss + 0.1 * pred_loss
                    final_loss = label_loss + 0.01 * pred_loss +0.001*rep_loss
                    # final_loss = label_loss + 0.01 * pred_loss + 0.0001 * rep_loss
                    student_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='student')

                    gradients = optimizer.compute_gradients(final_loss,var_list=student_variables,colocate_gradients_with_ops=False) #对loss计算梯度，后面用来做反向传播
                    tower_gradients.append(gradients)

    label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
    rep_loss=tf.reduce_mean(tf.stack(tower_rep_losses))
    pred_loss = tf.reduce_mean(tf.stack(tower_pred_losses))

    tf.summary.scalar("label_loss", label_loss)
    if regularization_penalty != 0:
        reg_loss = tf.reduce_mean(tf.stack(tower_reg_losses))
        tf.summary.scalar("reg_loss", reg_loss)
    merged_gradients = utils.combine_gradients(tower_gradients)

    if clip_gradient_norm > 0: #解决梯度爆炸问题
        with tf.name_scope('clip_grads'):
            merged_gradients = utils.clip_gradient_norms(merged_gradients, clip_gradient_norm)

    train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)

    postfix='_student'
    print('postfix=', postfix)

    tf.add_to_collection("global_step"+postfix, global_step)
    tf.add_to_collection("loss"+postfix, label_loss)
    tf.add_to_collection("predictions"+postfix, tf.concat(tower_predictions, 0))
    tf.add_to_collection("input_batch"+postfix, model_input)
    tf.add_to_collection("num_frames"+postfix, num_frames)
    tf.add_to_collection("labels"+postfix, tf.cast(labels_batch, tf.float32))
    tf.add_to_collection("train_op"+postfix, train_op)

    tf.add_to_collection("model_input" + postfix, model_input)

    tf.add_to_collection("rep_loss"+postfix, rep_loss)
    tf.add_to_collection("pred_loss" + postfix, pred_loss)
    tf.add_to_collection("label_loss_tch" + postfix, label_loss_tch)

    tf.add_to_collection("representations_tch" + postfix, representations_tch)


def GetInputDataBatch(reader,batch_size,num_readers,num_epochs,train_data_pattern,num_towers):

    unused_video_id, model_input_raw, labels_batch, num_frames = (
        get_input_data_tensors(
            reader,
            train_data_pattern,
            batch_size=batch_size * num_towers,
            num_readers=num_readers,
            num_epochs=num_epochs))

    print('unused_video_id', unused_video_id)
    print('model_input_raw', model_input_raw)
    print('labels_batch', labels_batch)
    print('num_frames', num_frames)

    # 数据集格式看https://research.google.com/youtube8m/download.html
    # 因为用了两张显卡，而且batch_size=80，所以是160
    # unused_video_id
    # Tensor("train_input/shuffle_batch_join:0", shape=(160,), dtype=string) video的id， (160,)代表一个含有160个元素的一维数组
    # model_input_raw
    # Tensor("train_input/shuffle_batch_join:1", shape=(160, 300, 1152), dtype=float32)  300秒， 帧特征1024+音频特征128=1152
    # labels_batch
    # Tensor("train_input/shuffle_batch_join:2", shape=(160, 3862), dtype=bool)  3862个标签，这是用one-hot代表video的label
    # num_frames
    # Tensor("train_input/shuffle_batch_join:3", shape=(160,), dtype=int32)  帧的数量

    tf.summary.histogram("model/input_raw", model_input_raw)  # 显示直方图信息
    # feature_dim = len(model_input_raw.get_shape()) - 1

    # 这些操作，还pca什么的，莫非是跟NextVLAD有关？事实上就是NextVLAD的操作。
    offset = np.array([4. / 512] * 1024 + [0] * 128)
    offset = tf.constant(offset, dtype=tf.float32)

    eigen_val = tf.constant(np.sqrt(np.load("yt8m_pca/eigenvals.npy")[:1024, 0]), dtype=tf.float32)
    model_input = tf.multiply(model_input_raw - offset, tf.pad(eigen_val + 1e-4, [[0, 128]], constant_values=1.))
    # model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

    print('offset=', offset)
    print('eigen_val=', eigen_val)
    print('model_input=', model_input)


    return model_input,labels_batch,num_frames



class Trainer(object):
    """A Trainer to train a Tensorflow graph."""

    def __init__(self, cluster, task, train_dir, reader, model_exporter,
                 log_device_placement=True, max_steps=None,
                 export_model_steps=10000,model_type="student"):
        """"Creates a Trainer.

        Args:
          cluster: A tf.train.ClusterSpec if the execution is distributed.
            None otherwise.
          task: A TaskSpec describing the job type and the task index.
        """

        self.cluster = cluster
        self.task = task
        self.is_master = (task.type == "master" and task.index == 0)
        self.train_dir = train_dir

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) #allow_growth=True
        self.config = tf.ConfigProto(
            gpu_options=gpu_options,allow_soft_placement=True, log_device_placement=log_device_placement)


        self.reader = reader
        self.model_exporter = model_exporter
        self.max_steps = max_steps
        self.max_steps_reached = False
        self.export_model_steps = export_model_steps
        self.last_model_export_step = 0
        self.model_type=model_type
    #     if self.is_master and self.task.index > 0:
    #       raise StandardError("%s: Only one replica of master expected",
    #                           task_as_string(self.task))

    def run(self, start_new_model=False):
        """Performs training on the currently defined Tensorflow graph.

        Returns:
          A tuple of the training Hit@1 and the training PERR.
        """
        if self.is_master and start_new_model:  #只要模型不改动，都不需要start_new_model，就读取之前保存好的就ok了
            self.remove_training_directory(self.train_dir)

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        model_flags_dict = { #当前模型的参数，用来对比json文件中的模型参数是否一致，或者用来保存模型json文件
            "model": FLAGS.model,
            "feature_sizes": FLAGS.feature_sizes,
            "feature_names": FLAGS.feature_names,
            "frame_features": FLAGS.frame_features, #frame_features=True
            "label_loss": FLAGS.label_loss,
        }
        flags_json_path = os.path.join(FLAGS.train_dir, "model_flags.json") #train_dir=nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic

        if os.path.exists(flags_json_path):  #看是否已经存在保存好的json模型，如果存在且参数跟当前参数不同，则报错，不存在则保存当前模型json文件
            existing_flags = json.load(open(flags_json_path))
            if existing_flags != model_flags_dict:
                logging.error("Model flags do not match existing file %s. Please "
                              "delete the file, change --train_dir, or pass flag "
                              "--start_new_model",
                              flags_json_path)
                logging.error("Ran model with flags: %s", str(model_flags_dict))
                logging.error("Previously ran with flags: %s", str(existing_flags))
                exit(1)
        else:
            # Write the file. 写模型的json文件
            with open(flags_json_path, "w") as fout:
                fout.write(json.dumps(model_flags_dict))

        target, device_fn = self.start_server_if_distributed() #不知道是做什么的，或许是并行的？

        if self.model_type=="teacher":
            model=nextvlad.NeXtVLADModel_teacher()
            self.pure_train(model, device_fn, target, start_new_model)
        # elif self.model_type=="student":
        #     model=nextvlad.NeXtVLADModel_student()
        #     self.pure_train(model, device_fn, target, start_new_model)
        elif self.model_type=="KD":
            self.Knowledge_Distillation(device_fn,target,start_new_model)
        else:
            print('model_type is ',self.model_type,'  Invalid!!!')
            print('仅支持单独训练student或进行知识蒸馏训练student，若要训练student，去看项目KD')




    def Knowledge_Distillation(self,device_fn,target,start_new_model):
        print('Knowledge_Distillation')
        teacher_model=nextvlad.NeXtVLADModel_teacher()
        student_model = nextvlad.NeXtVLADModel_student()

        with tf.Graph().as_default() as graph:  # 画图
            with tf.device(device_fn):
                #读取输入数据-------------------------------------------------
                label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
                optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

                local_device_protos = device_lib.list_local_devices()
                gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
                gpus = gpus[:FLAGS.num_gpu]
                num_gpus = len(gpus)

                model_input, labels_batch, num_frames = GetInputDataBatch(self.reader,
                                                                          batch_size=FLAGS.batch_size,
                                                                          num_readers=FLAGS.num_readers,
                                                                          num_epochs=FLAGS.num_epochs,
                                                                          train_data_pattern=FLAGS.train_data_pattern,
                                                                          num_towers=num_gpus,)
                # 先建个teacher的图-------------------------------------------------
                with tf.variable_scope('teacher', reuse=False):
                    # self.build_model(model, self.reader) #其实就是建图
                    # model_input_placerholder_tch = tf.placeholder(dtype=tf.float32, shape=[None, 300 , 1152], name='MI00')
                    # labels_batch_placerholder_tch = tf.placeholder(dtype=tf.bool, shape=[None, 3862], name='LB00')
                    # num_frames_placerholder_tch = tf.placeholder(dtype=tf.int32, shape=[None, ], name='NF00')

                    rep_tch,pred_tch=build_graph_teacher(model_input, labels_batch, num_frames,
                                reader=self.reader,
                                model=teacher_model,
                                optimizer_class=optimizer_class,
                                clip_gradient_norm=FLAGS.clip_gradient_norm,
                                label_loss_fn=label_loss_fn,
                                base_learning_rate=FLAGS.base_learning_rate,
                                learning_rate_decay=FLAGS.learning_rate_decay,
                                learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                                regularization_penalty=FLAGS.regularization_penalty,
                                batch_size=FLAGS.batch_size)

                postfix = '_teacher'
                global_step_teacher = tf.get_collection("global_step"+postfix)[0]
                loss_teacher = tf.get_collection("loss"+postfix)[0]
                labels_teacher = tf.get_collection("labels"+postfix)[0]
                train_op_teacher = tf.get_collection("train_op"+postfix)[0]
                model_input_teacher=tf.get_collection("model_input"+postfix)[0]

                representations_teacher = tf.get_collection("representations" + postfix)[0]
                predictions_teacher = tf.get_collection("predictions" + postfix)[0]

                teacher_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='teacher')
                teacher_saver = tf.train.Saver(var_list=teacher_variables,max_to_keep=5, keep_checkpoint_every_n_hours=1)  # teacher_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1.) #如果没有向 tf.train.Saver() 传递任何参数，则 Saver 会处理图中的所有变量。


                #再建个student的图--------------------------------------------------
                with tf.variable_scope('student', reuse=False):

                    # representations_tch_placeholder=tf.placeholder(dtype=tf.float32,shape=[None,2048],name='REPTCH')
                    # predictions_tch_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 3862],name='PREDTCH')

                    # model_input_placerholder_stu=tf.placeholder(dtype=tf.float32, shape=[None,300/FLAGS.k_frame,1152],name='MI11')
                    # labels_batch_placerholder_stu = tf.placeholder(dtype=tf.bool, shape=[None, 3862],name='LB11')
                    # num_frames_placerholder_stu = tf.placeholder(dtype=tf.int32, shape=[None, ],name='NF11')

                    build_graph_student(model_input, labels_batch, num_frames,
                                rep_tch, pred_tch,
                                reader=self.reader,
                                model=student_model,
                                optimizer_class=optimizer_class,
                                clip_gradient_norm=FLAGS.clip_gradient_norm,
                                label_loss_fn=label_loss_fn,
                                base_learning_rate=FLAGS.base_learning_rate,
                                learning_rate_decay=FLAGS.learning_rate_decay,
                                learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                                regularization_penalty=FLAGS.regularization_penalty,
                                batch_size=FLAGS.batch_size,
                                k_frame=FLAGS.k_frame)

                postfix = '_student'
                global_step_student = tf.get_collection("global_step" + postfix)[0]
                loss_student = tf.get_collection("loss" + postfix)[0]
                predictions_student = tf.get_collection("predictions" + postfix)[0]
                labels_student = tf.get_collection("labels" + postfix)[0]
                train_op_student = tf.get_collection("train_op" + postfix)[0]
                model_input_student = tf.get_collection("model_input" + postfix)[0]

                rep_loss=tf.get_collection("rep_loss" + postfix)[0]
                pred_loss = tf.get_collection("pred_loss" + postfix)[0]
                label_loss_tch = tf.get_collection("label_loss_tch" + postfix)[0]

                representations_tch_stu = tf.get_collection("representations_tch" + postfix)[0]


                student_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='student')
                student_saver = tf.train.Saver(var_list=student_variables,max_to_keep=5, keep_checkpoint_every_n_hours=1)  # student_saver = tf.train.Saver()

                # print("See variables in student")
                # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='student'):
                #     print(i)


                init_op = tf.global_variables_initializer()
                sv = tf.train.Supervisor(# 辅助训练，多管闲事 https: // www.tensorflow.org / api_docs / python / tf / train / Supervisor
                    graph,
                    logdir=self.train_dir,
                    init_op=init_op,
                    is_chief=self.is_master,
                    global_step=global_step_student,
                    save_model_secs=60 * 60,
                    save_summaries_secs=120,
                    saver=None)

                #开始知识蒸馏----------------------------------------------
                logging.info("%s: Starting managed session.", task_as_string(self.task))
                with sv.managed_session(target, config=self.config) as sess:
                    try:
                        # 先加载teacher的checkpoint
                        logging.info("%s: Check if teacher works .", task_as_string(self.task))
                        teacher_checkpoint_name = 'nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic_teacher/model.ckpt-95830'
                        print('teacher_checkpoint_name=', teacher_checkpoint_name)
                        teacher_saver.restore(sess, teacher_checkpoint_name)
                        # 检查这个teacher的效果
                        # EvalGAP = evaluate(teacher_model, teacher_checkpoint_name,300)
                        # print('Teacher_EvalGAP=', EvalGAP)

                        if FLAGS.start_new_model == False:
                            # 再加载student的checkpoint
                            print('load student checkpoint')
                            student_checkpoint_name=get_latest_checkpoint()
                            # student_checkpoint_name = 'nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic_KD_k_6/model.ckpt-110'
                            print('student_checkpoint_name=', student_checkpoint_name)
                            student_saver.restore(sess, student_checkpoint_name)

                        logging.info("%s: Entering training loop.", task_as_string(self.task))
                        while (not sv.should_stop()) and (not self.max_steps_reached):
                            batch_start_time = time.time()

                            _, global_step_stu_val, loss_stu_val, predictions_stu_val, labels_stu_val,\
                            rep_loss_val,pred_loss_val,label_loss_tch_val= sess.run(
                                [train_op_student, global_step_student, loss_student, predictions_student, labels_student,
                                 rep_loss,pred_loss,label_loss_tch])

                            seconds_per_batch = time.time() - batch_start_time  # 上面已经训练完一个batch了
                            examples_per_second = labels_stu_val.shape[0] / seconds_per_batch
                            if self.max_steps and self.max_steps <= global_step_stu_val:  # 训练完一个batch就是走了一个step
                                self.max_steps_reached = True

                            if self.is_master and global_step_stu_val % 10 == 0 and self.train_dir:

                                hit_at_one = eval_util.calculate_hit_at_one(predictions_stu_val,labels_stu_val)  # 一个评估指标H@1
                                perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_stu_val, labels_stu_val)
                                gap = eval_util.calculate_gap(predictions_stu_val, labels_stu_val)

                                logging.info(
                                    "Student training step " + str(global_step_stu_val) + " | Loss: " + ("%.2f" % loss_stu_val) +
                                    # " | rep_Loss: " + ("%.2f" % rep_loss_stu_val) +" | pred_Loss: " + ("%.2f" % pred_loss_stu_val) +
                                    " Examples/sec: " + ("%.2f" % examples_per_second) + " | Hit@1: " +
                                    ("%.2f" % hit_at_one) + " PERR: " + ("%.2f" % perr) +
                                    " GAP: " + ("%.2f" % gap)
                                    )

                                # Exporting the model every x steps
                                time_to_export = ((self.last_model_export_step == 0) or
                                                  (global_step_stu_val - self.last_model_export_step >= self.export_model_steps))

                                if self.is_master and time_to_export:
                                    student_saver.save(sess, sv.save_path, global_step_stu_val)
                                    latest_checkpoint = get_latest_checkpoint()
                                    print('latest_checkpoint=',latest_checkpoint)
                                    EvalGAP = evaluate(student_model, latest_checkpoint,FLAGS.k_frame)

                                    with open( 'record_'+FLAGS.train_dir+'.txt', "a") as fout:
                                        fout.write('step=' + str(global_step_stu_val) + ' trainGAP=' + str(
                                            gap) + ' valGAP=' + str(EvalGAP) + '\n')
                                    self.last_model_export_step = global_step_stu_val

                            else:
                                logging.info(
                                    "Student training step " + str(global_step_stu_val) + " | Loss: " + (
                                                "%.2f" % loss_stu_val) +
                                    " | rep_Loss: " + ("%.2f" % rep_loss_val) + " | pred_Loss: " + (
                                                "%.2f" % pred_loss_val) +
                                    " Examples/sec: " + ("%.2f" % examples_per_second)
                                    + '  Teaher_label_loss_tch_stu：'+str(label_loss_tch_val) )
                    except tf.errors.OutOfRangeError:
                        logging.info("%s: Done training -- epoch limit reached.",
                                     task_as_string(self.task))

                    logging.info("%s: Exited training loop.", task_as_string(self.task))
                    sv.Stop()




    def pure_train(self,model,device_fn,target,start_new_model,):
        print('pure_train_'+FLAGS.model_type)
        print('model=', model)  # 现在model=NextVLADModel_teacher or _student

        scope_name = 'teacher'
        # if model==nextvlad.NeXtVLADModel_student():
        #     scope_name='student'

        with tf.Graph().as_default() as graph:  # 画图
            with tf.device(device_fn):
                label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
                optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

                local_device_protos = device_lib.list_local_devices()
                gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
                gpus = gpus[:FLAGS.num_gpu]
                num_gpus = len(gpus)

                model_input, labels_batch, num_frames = GetInputDataBatch(self.reader,
                                                                          batch_size=FLAGS.batch_size,
                                                                          num_readers=FLAGS.num_readers,
                                                                          num_epochs=FLAGS.num_epochs,
                                                                          train_data_pattern=FLAGS.train_data_pattern,
                                                                          num_towers=num_gpus,
                                                                          )
                with tf.variable_scope(scope_name, reuse=False):
                    build_graph_teacher(model_input, labels_batch, num_frames,
                                reader=self.reader,
                                model=model,
                                optimizer_class=optimizer_class,
                                clip_gradient_norm=FLAGS.clip_gradient_norm,
                                label_loss_fn=label_loss_fn,
                                base_learning_rate=FLAGS.base_learning_rate,
                                learning_rate_decay=FLAGS.learning_rate_decay,
                                learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                                regularization_penalty=FLAGS.regularization_penalty,
                                batch_size=FLAGS.batch_size)


                saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1.)
                # 取出图中几个后面要用到的结点，方便处理
                postfix='_teacher'
                global_step = tf.get_collection("global_step"+postfix)[0]
                loss = tf.get_collection("loss"+postfix)[0]
                predictions = tf.get_collection("predictions"+postfix)[0]
                labels = tf.get_collection("labels"+postfix)[0]
                train_op = tf.get_collection("train_op"+postfix)[0]
                init_op = tf.global_variables_initializer()

            # stats_graph(graph)#真的是这么算FLOPs的吗？


        sv = tf.train.Supervisor(  # 辅助训练，多管闲事 https: // www.tensorflow.org / api_docs / python / tf / train / Supervisor
            graph,
            logdir=self.train_dir,
            init_op=init_op,
            is_chief=self.is_master,
            global_step=global_step,
            save_model_secs=60 * 60,
            save_summaries_secs=120,
            saver=None)

        logging.info("%s: Starting managed session.", task_as_string(self.task))
        with sv.managed_session(target, config=self.config) as sess:
            try:
                if start_new_model==False:
                    checkpoint_name=tf.train.latest_checkpoint(FLAGS.train_dir)
                    # checkpoint_name= 'nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic_teacher/model.ckpt-17210'#如果我不想load最后一个ckeckpoint的话，可以直接给出meta_filename吧。
                    print('checkpoint_name=', checkpoint_name)
                    saver.restore(sess,checkpoint_name)
                    EvalGAP=evaluate(model, checkpoint_name,FLAGS.k_frame)


                logging.info("%s: Entering training loop.", task_as_string(self.task))
                while (not sv.should_stop()) and (not self.max_steps_reached):
                    batch_start_time = time.time()
                    _, global_step_val, loss_val, predictions_val, labels_val = sess.run([train_op, global_step, loss, predictions, labels])

                    seconds_per_batch = time.time() - batch_start_time  # 上面已经训练完一个batch了
                    examples_per_second = labels_val.shape[0] / seconds_per_batch

                    if self.max_steps and self.max_steps <= global_step_val:  # 训练完一个batch就是走了一个step
                        self.max_steps_reached = True

                    if self.is_master and global_step_val % 10 == 0 and self.train_dir:
                        eval_start_time = time.time()

                        hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)  # 一个评估指标H@1
                        perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val, labels_val)
                        gap = eval_util.calculate_gap(predictions_val, labels_val)

                        eval_end_time = time.time()
                        eval_time = eval_end_time - eval_start_time

                        logging.info("training step " + str(global_step_val) + " | Loss: " + ("%.2f" % loss_val) +
                                     " Examples/sec: " + ("%.2f" % examples_per_second) + " | Hit@1: " +
                                     ("%.2f" % hit_at_one) + " PERR: " + ("%.2f" % perr) +
                                     " GAP: " + ("%.2f" % gap))

                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_Hit@1", hit_at_one),
                            global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_Perr", perr), global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_GAP", gap), global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("global_step/Examples/Second",
                                              examples_per_second), global_step_val)
                        sv.summary_writer.flush()

                        # Exporting the model every x steps
                        time_to_export = ((self.last_model_export_step == 0) or
                                          (global_step_val - self.last_model_export_step >= self.export_model_steps))

                        if self.is_master and time_to_export:
                            saver.save(sess, sv.save_path, global_step_val)
                            latest_checkpoint = get_latest_checkpoint()
                            EvalGAP = evaluate(model,latest_checkpoint,FLAGS.k_frame)
                            with open('record_' + FLAGS.train_dir + '.txt', "a") as fout:
                                fout.write('step=' + str(global_step_val) + ' trainGAP=' + str(
                                    gap) + ' valGAP=' + str(EvalGAP) + '\n')
                            self.last_model_export_step = global_step_val


                    else:
                        logging.info("training step " + str(global_step_val) + " | Loss: " +
                                     ("%.2f" % loss_val) + " Examples/sec: " + ("%.2f" % examples_per_second))
            except tf.errors.OutOfRangeError:
                logging.info("%s: Done training -- epoch limit reached.",
                             task_as_string(self.task))

            logging.info("%s: Exited training loop.", task_as_string(self.task))
            sv.Stop()


    def start_server_if_distributed(self):
        """Starts a server if the execution is distributed."""

        if self.cluster:
            logging.info("%s: Starting trainer within cluster %s.",
                         task_as_string(self.task), self.cluster.as_dict())
            server = start_server(self.cluster, self.task)
            target = server.target
            device_fn = tf.train.replica_device_setter(
                ps_device="/job:ps",
                worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
                cluster=self.cluster)
        else:
            target = ""
            device_fn = ""
        return (target, device_fn)

    def remove_training_directory(self, train_dir):
        """Removes the training directory."""
        try:
            logging.info(
                "%s: Removing existing train directory.",
                task_as_string(self.task))
            gfile.DeleteRecursively(train_dir)
        except:
            logging.error(
                "%s: Failed to delete directory " + train_dir +
                " when starting a new model. Please delete it manually and" +
                " try again.", task_as_string(self.task))


def get_reader():
    # Convert feature_names and feature_sizes to lists of values.
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(FLAGS.feature_names, FLAGS.feature_sizes)
    #得到的feature names=['rgb','audio'] feature_sizes=[1024,128]，就是说视频图像的特征维度是1024，音频是128

    if FLAGS.frame_features: #frame_features=True，我就是要用帧的特征
        reader = readers.YT8MFrameFeatureReader(feature_names=feature_names, feature_sizes=feature_sizes)
    else:
        reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names, feature_sizes=feature_sizes)

    return reader


class ParameterServer(object):
    """A parameter server to serve variables in a distributed execution."""

    def __init__(self, cluster, task):
        """Creates a ParameterServer.

        Args:
          cluster: A tf.train.ClusterSpec if the execution is distributed.
            None otherwise.
          task: A TaskSpec describing the job type and the task index.
        """

        self.cluster = cluster
        self.task = task

    def run(self):
        """Starts the parameter server."""

        logging.info("%s: Starting parameter server within cluster %s.",
                     task_as_string(self.task), self.cluster.as_dict())
        server = start_server(self.cluster, self.task)
        server.join()


def start_server(cluster, task):
    """Creates a Server.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    if not task.type:
        raise ValueError("%s: The task type must be specified." %
                         task_as_string(task))
    if task.index is None:
        raise ValueError("%s: The task index must be specified." %
                         task_as_string(task))

    # Create and start a server.
    return tf.train.Server(
        tf.train.ClusterSpec(cluster),
        protocol="grpc",
        job_name=task.type,
        task_index=task.index)


def task_as_string(task):
    return "/job:%s/task:%s" % (task.type, task.index)


def main(unused_argv):
    if FLAGS.model_type=="teacher":
        FLAGS.train_dir = FLAGS.train_dir + '_teacher'
    elif FLAGS.model_type=="student":
        FLAGS.train_dir = FLAGS.train_dir +'_student'+ '_k_' + str(FLAGS.k_frame)
    else:
        FLAGS.train_dir = FLAGS.train_dir + '_KD' + '_k_' + str(FLAGS.k_frame)

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_gpu

    # Load the environment.
    env = json.loads(os.environ.get("TF_CONFIG", "{}"))

    # Load the cluster data from the environment.
    cluster_data = env.get("cluster", None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

    # Load the task data from the environment.
    task_data = env.get("task", None) or {"type": "master", "index": 0}
    task = type("TaskSpec", (object,), task_data)

    # Logging the version.
    logging.set_verbosity(tf.logging.INFO)
    logging.info("%s: Tensorflow version: %s.",
                 task_as_string(task), tf.__version__)
    print('task.type=',task.type)    #master、worker是一种并行的方式，https://www.cnblogs.com/yueshutong/p/9695411.html
    # Dispatch to a master, a worker, or a parameter server.
    if not cluster or task.type == "master" or task.type == "worker":

        reader = get_reader() # 这个类用来读取frame_feature数据
        Trainer(cluster, task, FLAGS.train_dir, reader, None,FLAGS.log_device_placement, FLAGS.max_steps,FLAGS.export_model_steps,model_type=FLAGS.model_type)\
            .run(start_new_model=FLAGS.start_new_model) #不仅定义了Trainer，还调用了它的run函数

    elif task.type == "ps":
        ParameterServer(cluster, task).run()
    else:
        raise ValueError("%s: Invalid task_type: %s." %
                         (task_as_string(task), task.type))

def GetEvalParameter():
    flags.DEFINE_string(
        "eval_data_pattern", "/home/disk3/a_zhongzhanhui/yt8m_dataset/validate_368M/validate*.tfrecord",
        "File glob defining the evaluation dataset in tensorflow.SequenceExample "
        "format. The SequenceExamples are expected to have an 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_boolean("run_once", True, "Whether to run eval only once.")
    flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

if __name__ == "__main__":
    GetEvalParameter()
    # Dataset flags.
    flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                        "The directory to save the model files in.")
    flags.DEFINE_string(
        "train_data_pattern", "",
        "File glob for the training dataset. If the files refer to Frame Level "
        "features (i.e. tensorflow.SequenceExample), then set --reader_type "
        "format. The (Sequence)Examples are expected to have 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                                                     "to use for training.")
    flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

    # Model flags.
    flags.DEFINE_bool(
        "frame_features", False,
        "If set, then --train_data_pattern must be frame-level features. "
        "Otherwise, --train_data_pattern must be aggregated video-level "
        "features. The model must also be set appropriately (i.e. to read 3D "
        "batches VS 4D batches.")
    flags.DEFINE_string(
        "model", "LogisticModel",
        "Which architecture to use for the model. Models are defined "
        "in models.py.")
    flags.DEFINE_bool(
        "start_new_model", False,
        "If set, this will not resume from a checkpoint and will instead create a"
        " new model instance.")

    # Training flags.
    flags.DEFINE_integer("num_gpu", 1,
                         "The maximum number of GPU devices to use for training. "
                         "Flag only applies if GPUs are installed")
    flags.DEFINE_integer("batch_size", 1024,
                         "How many examples to process per batch for training.")
    flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                        "Which loss function to use for training the model.")
    flags.DEFINE_float(
        "regularization_penalty", 1.0,
        "How much weight to give to the regularization loss (the label loss has "
        "a weight of 1).")
    flags.DEFINE_float("base_learning_rate", 0.01,
                       "Which learning rate to start with.")
    flags.DEFINE_float("learning_rate_decay", 0.95,
                       "Learning rate decay factor to be applied every "
                       "learning_rate_decay_examples.")
    flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                       "Multiply current learning rate by learning_rate_decay "
                       "every learning_rate_decay_examples.")
    flags.DEFINE_integer("num_epochs", 5,
                         "How many passes to make over the dataset before "
                         "halting training.")
    flags.DEFINE_integer("max_steps", None,
                         "The maximum number of iterations of the training loop.")
    flags.DEFINE_integer("export_model_steps", 100,
                         "The period, in number of steps, with which the model "
                         "is exported for batch prediction.")

    # Other flags.
    flags.DEFINE_integer("num_readers", 8,
                         "How many threads to use for reading input files.")
    flags.DEFINE_string("optimizer", "AdamOptimizer",
                        "What optimizer class to use.")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
    flags.DEFINE_bool(
        "log_device_placement", False,
        "Whether to write the device on which every op will run into the "
        "logs on startup.")

    flags.DEFINE_string("visible_gpu", "0",
                        "visible gpu")
    flags.DEFINE_integer("k_frame", 300,
                         "value of k")
    flags.DEFINE_string("model_type", "student",
                         "teacher or student or KD")
    app.run() #首先加载flags的参数项，然后执行main()函数，其中参数使用tf.app.flags.FLAGS定义的。


