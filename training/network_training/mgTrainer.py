#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from _warnings import warn

import torch.backends.cudnn as cudnn
from time import time

from torch.cuda.amp import GradScaler, autocast
from tqdm import trange

from nnunet.utilities.nd_softmax import softmax_helper, identity

from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

from nnunet.network_architecture.model_gensis import generate_pair

import shutil
import traceback
from collections import OrderedDict
from multiprocessing import Pool
from time import sleep
from typing import Tuple, List

import matplotlib
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from torch import nn
from torch.optim import lr_scheduler

import nnunet
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.postprocessing.connected_components import determine_postprocessing
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_default_augmentation, get_patch_size
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset

from nnunet.training.network_training.network_trainer import NetworkTrainer
from monai import transforms

import SimpleITK as sitk

matplotlib.use("agg")


class MgTrainer(NetworkTrainer):
    def __init__(self, plans_file, fold, output_folder=None,
                 dataset_directory=None, batch_dice=True, stage=None, unpack_data=True,
                 deterministic=True, fp16=False):
        """
        :param deterministic:
        :param fold: can be either [0 ... 5) for cross-validation, 'all' to train on all available training data or
        None if you wish to load some checkpoint and do inference only
        :param plans_file: the pkl file generated by preprocessing. This file will determine all design choices
        :param subfolder_with_preprocessed_data: must be a subfolder of dataset_directory (just the name of the folder,
        not the entire path). This is where the preprocessed data lies that will be used for network training. We made
        this explicitly available so that differently preprocessed data can coexist and the user can choose what to use.
        Can be None if you are doing inference only.
        :param output_folder: where to store parameters, plot progress and to the validation
        :param dataset_directory: the parent directory in which the preprocessed Task data is stored. This is required
        because the split information is stored in this directory. For running prediction only this input is not
        required and may be set to None
        :param batch_dice: compute dice loss for each sample and average over all samples in the batch or pretend the
        batch is a pseudo volume?
        :param stage: The plans file may contain several stages (used for lowres / highres / pyramid). Stage must be
        specified for training:
        if stage 1 exists then stage 1 is the high resolution stage, otherwise it's 0
        :param unpack_data: if False, npz preprocessed data will not be unpacked to npy. This consumes less space but
        is considerably slower! Running unpack_data=False with 2d should never be done!

        IMPORTANT: If you inherit from nnUNetTrainer and the init args change then you need to redefine self.init_args
        in your init accordingly. Otherwise checkpoints won't load properly!
        """
        super(MgTrainer, self).__init__(deterministic, fp16)
        self.max_num_epochs = 400

        self.unpack_data = unpack_data
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16)
        # set through arguments from init
        self.stage = stage
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file

        self.output_folder = output_folder

        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder
        self.fold = fold

        self.plans = None

        # if we are running inference only then the self.dataset_directory is set (due to checkpoint loading) but it
        # irrelevant
        if self.dataset_directory is not None and isdir(self.dataset_directory):
            self.gt_niftis_folder = join(self.dataset_directory, "gt_segmentations")
        else:
            self.gt_niftis_folder = None

        self.folder_with_preprocessed_data = None

        # set in self.initialize()

        self.dl_tr = self.dl_val = None
        self.num_input_channels = self.num_classes = self.net_pool_per_axis = self.patch_size = self.batch_size = \
            self.threeD = self.base_num_features = self.intensity_properties = self.normalization_schemes = \
            self.net_num_pool_op_kernel_sizes = self.net_conv_kernel_sizes = None  # loaded automatically from plans_file
        self.basic_generator_patch_size = self.data_aug_params = self.transpose_forward = self.transpose_backward = None

        # self.batch_dice = batch_dice
        # self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.loss = nn.MSELoss()

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.classes = self.do_dummy_2D_aug = self.use_mask_for_norm = self.only_keep_largest_connected_component = \
            self.min_region_size_per_class = self.min_size_per_class = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        self.update_fold(fold)
        self.pad_all_sides = None

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.33

        self.conv_per_stage = None
        self.regions_class_order = None

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """
        self.print_to_log_file('-------------------------------------------------')

        self.print_to_log_file('I am using the following configuration:')

        self.print_to_log_file('max epoch:      | ', self.max_num_epochs)
        self.print_to_log_file('output fold:    | ', self.output_folder)
        self.print_to_log_file('mask type:     | ', 'Model Gensis')

        self.print_to_log_file('-------------------------------------------------\n')

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)

        self.setup_DA_params()

        if training:
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)

            self.dl_tr, self.dl_val = self.get_basic_generators()
            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")
            self.tr_gen, self.val_gen = get_default_augmentation(self.dl_tr, self.dl_val,
                                                                 self.data_aug_params[
                                                                     'patch_size_for_spatialtransform'],
                                                                 self.data_aug_params)
            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                   also_print_to_console=False)
            self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                   also_print_to_console=False)
        else:
            pass
        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True

    def initialize_network(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """
        # self.print_to_log_file(self.net_num_pool_op_kernel_sizes)
        # self.print_to_log_file(self.net_conv_kernel_sizes)

        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, 4, net_numpool,
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        self.network.inference_apply_nonlin = identity

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                           patience=self.lr_scheduler_patience,
                                                           verbose=True, threshold=self.lr_scheduler_eps,
                                                           threshold_mode="abs")

    def update_fold(self, fold):
        """
        used to swap between folds for inference (ensemble of models from cross-validation)
        DO NOT USE DURING TRAINING AS THIS WILL NOT UPDATE THE DATASET SPLIT AND THE DATA AUGMENTATION GENERATORS
        :param fold:
        :return:
        """
        if fold is not None:
            if isinstance(fold, str):
                assert fold == "all", "if self.fold is a string then it must be \'all\'"
                if self.output_folder.endswith("%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "%s" % str(fold))
            else:
                if self.output_folder.endswith("fold_%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "fold_%s" % str(fold))
            self.fold = fold

    def setup_DA_params(self):
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

    def plot_network_architecture(self):
        try:
            from batchgenerators.utilities.file_and_folder_operations import join
            import hiddenlayer as hl
            if torch.cuda.is_available():
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                                   transforms=None)
            else:
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)),
                                   transforms=None)
            g.save(join(self.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            self.print_to_log_file("Unable to plot network architecture:")
            self.print_to_log_file(e)

            self.print_to_log_file("\nprinting the network instead:\n")
            self.print_to_log_file(self.network)
            self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil

        shutil.copy(self.plans_file, join(self.output_folder_base, "plans.pkl"))

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                #self.print_to_log_file("saving best epoch checkpoint...")
                if self.save_best_checkpoint:
                    self.save_model_naive(join(self.output_folder, "model_best.pth"))
                    self.save_checkpoint(join(self.output_folder, "model_best.model"))
                    self.print_to_log_file('Saving best model')

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                #self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                #self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    #self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    #self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                pass
                #self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            self.print_to_log_file("WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            self.print_to_log_file("WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def preprocess_patient(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        from nnunet.training.model_restore import recursive_find_python_class
        preprocessor_name = self.plans.get('preprocessor_name')
        if preprocessor_name is None:
            if self.threeD:
                preprocessor_name = "GenericPreprocessor"
            else:
                preprocessor_name = "PreprocessorFor2D"

        print("using preprocessor", preprocessor_name)
        preprocessor_class = recursive_find_python_class([join(nnunet.__path__[0], "preprocessing")],
                                                         preprocessor_name,
                                                         current_module="nnunet.preprocessing")
        assert preprocessor_class is not None, "Could not find preprocessor %s in nnunet.preprocessing" % \
                                               preprocessor_name
        preprocessor = preprocessor_class(self.normalization_schemes, self.use_mask_for_norm,
                                          self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'])
        return d, s, properties

    def save_model_naive(self, filename):
        self.print_to_log_file('save_model_naive to :', filename)
        torch.save(self.network.state_dict(), filename)

    def save_checkpoint(self, fname, save_optimizer=True):
        super(MgTrainer, self).save_checkpoint(fname, save_optimizer)
        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans

        write_pickle(info, fname + ".pkl")

    def on_epoch_end(self):
        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training

    def run_training(self):
        self.save_debug_information()

        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        # self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

            if self.epoch % 100 == 0 and self.epoch != 0:
                self.save_model_naive(filename=join(self.output_folder, f"{int(self.epoch)}.pth"))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_model_naive(join(self.output_folder, "model_final.pth"))

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']

        # if self.synthesize is not None:
        #     data = synthesize_input(data, self.mask_size, self.mask_ratio)
        # else:
        #     data = data

        data, target = generate_pair(data.numpy(), batch_size=2)

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data

                l = self.loss(output, target)

                if do_backprop:
                    self.amp_grad_scaler.scale(l).backward()
                    self.amp_grad_scaler.step(self.optimizer)
                    self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data

            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)
        del target

        return l.detach().cpu().numpy()

    def validate(self, validation_folder_name: str = 'validation_raw', overwrite=True):
        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training

        self.network.eval()
        self.network.to('cuda')

        output_folder = join(self.output_folder, validation_folder_name)

        crop = transforms.CenterSpatialCrop(roi_size=(128, 128, 128))

        for k in self.dataset_tr.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]

            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))):
                data = np.load(self.dataset[k]['data_file'])['data']

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0
                data = data[:-1]

                C, h, w, z = data.shape

                padding = np.maximum((h, w, z), (128, 128, 128))

                spatial_pad = transforms.SpatialPad(spatial_size=padding)
                data = crop(spatial_pad(data))

                data = data.unsqueeze(0)

                data, gt = generate_pair(data.numpy(), 1)
                data = torch.from_numpy(data)
                gt = torch.from_numpy(gt).squeeze(0)

                out = self.network(data.to('cuda')).detach().cpu().squeeze(0).numpy()

                t1 = out[0]
                t1_ce = out[1]
                t2 = out[2]
                flair = out[-1]

                t1 = sitk.GetImageFromArray(t1)
                t1_ce = sitk.GetImageFromArray(t1_ce)
                t2 = sitk.GetImageFromArray(t2)
                flair = sitk.GetImageFromArray(flair)

                maybe_mkdir_p(os.path.join(output_folder, fname))
                sitk.WriteImage(t1, os.path.join(output_folder, fname, "t1.nii.gz"))
                sitk.WriteImage(t1_ce, os.path.join(output_folder, fname, "t1_ce.nii.gz"))
                sitk.WriteImage(t2, os.path.join(output_folder, fname, "t2.nii.gz"))
                sitk.WriteImage(flair, os.path.join(output_folder, fname, "flair.nii.gz"))

                t1 = gt[0]
                t1_ce = gt[1]
                t2 = gt[2]
                flair = gt[-1]

                t1 = sitk.GetImageFromArray(t1)
                t1_ce = sitk.GetImageFromArray(t1_ce)
                t2 = sitk.GetImageFromArray(t2)
                flair = sitk.GetImageFromArray(flair)

                maybe_mkdir_p(os.path.join(self.output_folder_base, "gt_niftis", fname))
                sitk.WriteImage(t1, os.path.join(self.output_folder_base, "gt_niftis", fname, "t1.nii.gz"))
                sitk.WriteImage(t1_ce, os.path.join(self.output_folder_base, "gt_niftis", fname, "t1_ce.nii.gz"))
                sitk.WriteImage(t2, os.path.join(self.output_folder_base, "gt_niftis", fname, "t2.nii.gz"))
                sitk.WriteImage(flair, os.path.join(self.output_folder_base, "gt_niftis", fname, "flair.nii.gz"))

                data = data.squeeze(0)

                t1 = data[0]
                t1_ce = data[1]
                t2 = data[2]
                flair = data[-1]

                t1 = sitk.GetImageFromArray(t1)
                t1_ce = sitk.GetImageFromArray(t1_ce)
                t2 = sitk.GetImageFromArray(t2)
                flair = sitk.GetImageFromArray(flair)

                maybe_mkdir_p(os.path.join(self.output_folder_base, "gen_pair", fname))
                sitk.WriteImage(t1, os.path.join(self.output_folder_base, "gen_pair", fname, "t1.nii.gz"))
                sitk.WriteImage(t1_ce, os.path.join(self.output_folder_base, "gen_pair", fname, "t1_ce.nii.gz"))
                sitk.WriteImage(t2, os.path.join(self.output_folder_base, "gen_pair", fname, "t2.nii.gz"))
                sitk.WriteImage(flair, os.path.join(self.output_folder_base, "gen_pair", fname, "flair.nii.gz"))

        self.network.train(current_mode)
