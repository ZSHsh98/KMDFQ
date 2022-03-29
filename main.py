import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import DataParallel
# option file should be modified according to your expriment
from options import Option
import numpy as np
from dataloader import DataLoader
from trainer import Trainer
import random
import utils as utils
from quantization_utils.quant_modules import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from conditional_batchnorm import CategoricalConditionalBatchNorm2d
from torch.utils.tensorboard import SummaryWriter

class Generator(nn.Module):
	def __init__(self, input_dim=None, options=None, conf_path=None):
		super(Generator, self).__init__()
		self.settings = options or Option(conf_path)
		self.label_emb = nn.Embedding(self.settings.nClasses, input_dim)
		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

		self.conv_blocks0 = nn.Sequential(
			nn.BatchNorm2d(128),
		)

		self.conv_blocks1 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.conv_blocks2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
			nn.Tanh(),
			nn.BatchNorm2d(self.settings.channels, affine=False)
		)

	def forward(self, z, labels):
		gen_input = torch.mul(self.label_emb(labels), z)					#	torch.Size([200, 100])
		out = self.l1(gen_input)											#	torch.Size([200, 8192]) 
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)	#	torch.Size([200, 128, 8, 8])
		img = self.conv_blocks0(out)										#	torch.Size([200, 128, 8, 8])
		img = nn.functional.interpolate(img, scale_factor=2)				#	torch.Size([200, 128, 16, 16])
		img = self.conv_blocks1(img)										#	torch.Size([200, 128, 16, 16])
		img = nn.functional.interpolate(img, scale_factor=2)				#	torch.Size([200, 128, 32, 32])
		img = self.conv_blocks2(img)										#	torch.Size([200, 3, 32, 32])
		return img

class Generator_imagenet_resnet18(nn.Module):
	def __init__(self, input_dim=None, options=None, conf_path=None):

		super(Generator_imagenet_resnet18, self).__init__()
		self.settings = options or Option(conf_path)

		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		return img

class Generator_imagenet_vgg(nn.Module):
	def __init__(self, input_dim=None, options=None, conf_path=None):

		super(Generator_imagenet_vgg, self).__init__()

		self.settings = options or Option(conf_path)

		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		return img


class Generator_imagenet_inceptionv3(nn.Module):
	def __init__(self, input_dim=None, options=None, conf_path=None):

		super(Generator_imagenet_inceptionv3, self).__init__()

		self.settings = options or Option(conf_path)

		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		img = nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
		return img

class ExperimentDesign:
	def __init__(self,  model=None, model_teacher=None, options=None, conf_path=None):
		self.settings = options or Option(conf_path)
		
		self.train_loader = None
		self.test_loader = None
		self.model = model
		self.model_teacher = model_teacher
		
		self.optimizer_state = None
		self.trainer = None
		self.start_epoch = 0
		self.test_input = None

		self.unfreeze_Flag = True
		

		
		self.settings.set_save_path()
		self.logger = self.set_logger()
		self.settings.paramscheck(self.logger)

		self.prepare()
	
	def set_logger(self):
		logger = logging.getLogger('baseline')
		file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
		console_formatter = logging.Formatter('%(message)s')
		# file log
		file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
		file_handler.setFormatter(file_formatter)
		
		# console log
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(console_formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		
		logger.setLevel(logging.INFO)
		return logger

	def prepare(self):
		self._set_gpu()
		self._set_dataloader()
		self._set_model()
		self._set_gan_model()
		self._replace()
		self.logger.info("#==>CE_WEIGHT: {:.3f}, BNS_WEIGHT : {:.3f}, FEATURE_WEIGHT : {:.3f},\
		warmup_epochs: {:d}".format(self.settings.CE_WEIGHT, self.settings.BNS_WEIGHT, \
			self.settings.FEATURE_WEIGHT, self.settings.warmup_epochs))
		#self.logger.info(self.model)
		self._set_trainer()
	
	def _set_gpu(self):
		random.seed(self.settings.manualSeed)
		np.random.seed(self.settings.manualSeed)
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed_all(self.settings.manualSeed)
		cudnn.benchmark = True
		cudnn.deterministic = False
		assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"

	def _set_dataloader(self):
		# create data loader
		data_loader = DataLoader(dataset=self.settings.dataset,
		                         batch_size=self.settings.batchSize,
								 imgsize = self.settings.img_size,
		                         data_path=self.settings.dataPath,
		                         n_threads=self.settings.nThreads,
		                         ten_crop=self.settings.tenCrop,
		                         logger=self.logger)
		
		self.train_loader, self.test_loader = data_loader.getloader()

	def _set_model(self):
		if self.settings.dataset in ["cifar100"]:
			self.test_input = Variable(torch.randn(1, 3, 32, 32).cuda())
			self.model = DataParallel(ptcv_get_model('resnet20_cifar100', pretrained=True)).cuda()
			self.model_teacher = DataParallel(ptcv_get_model('resnet20_cifar100', pretrained=True)).cuda()
			self.model_teacher.eval()

		elif self.settings.dataset in ["cifar10"]:
			self.test_input = Variable(torch.randn(1, 3, 32, 32).cuda())
			self.model = DataParallel(ptcv_get_model('resnet20_cifar10', pretrained=True)).cuda()
			self.model_teacher = DataParallel(ptcv_get_model('resnet20_cifar10', pretrained=True)).cuda()
			self.model_teacher.eval()

		elif self.settings.dataset in ["imagenet"]:
			self.test_input = Variable(torch.randn(1, 3, self.settings.img_size, self.settings.img_size).cuda())
			if self.settings.net == 'bn_vgg16':
				self.model = DataParallel(ptcv_get_model('bn_vgg16', pretrained=True)).cuda()
				self.model_teacher = DataParallel(ptcv_get_model('bn_vgg16', pretrained=True)).cuda()
				self.model_teacher.eval()
			elif self.settings.net == 'inceptionv3':
				self.model = DataParallel(ptcv_get_model('inceptionv3', pretrained=True)).cuda()
				self.model_teacher = DataParallel(ptcv_get_model('inceptionv3', pretrained=True)).cuda()
				self.model_teacher.eval()
			elif self.settings.net == 'resnet18':
				self.model = DataParallel(ptcv_get_model('resnet18', pretrained=True)).cuda()
				self.model_teacher = DataParallel(ptcv_get_model('resnet18', pretrained=True)).cuda()
				self.model_teacher.eval()
			elif self.settings.net == 'MobileV2':
				self.model = DataParallel(ptcv_get_model('mobilenetv2_w1', pretrained=True)).cuda()
				self.model_teacher = DataParallel(ptcv_get_model('mobilenetv2_w1', pretrained=True)).cuda()
				self.model_teacher.eval()
			elif self.settings.net == 'shufflenet':
				self.model = DataParallel(ptcv_get_model('shufflenet_g1_w1', pretrained=True)).cuda()
				self.model_teacher = DataParallel(ptcv_get_model('shufflenet_g1_w1', pretrained=True)).cuda()
				self.model_teacher.eval()
			elif self.settings.net == 'resnet50':
				self.model = DataParallel(ptcv_get_model('resnet50', pretrained=True)).cuda()
				self.model_teacher = DataParallel(ptcv_get_model('resnet50', pretrained=True)).cuda()
				self.model_teacher.eval()
			else:
				assert False, "please select the net for imagenet!"
		else:
			assert False, "unsupport data set: " + self.settings.dataset

	def _set_gan_model(self):
		_, feature = self.model(self.test_input, out_feature=True)
		#print(feature.shape[-1])

		if self.settings.dataset in ["cifar100", "cifar10"]:
			self.generator = DataParallel(Generator(self.settings.latent_dim,self.settings)).cuda()
		elif self.settings.dataset in ["imagenet"]:
			if self.settings.net == 'bn_vgg16':
				self.generator = DataParallel(Generator_imagenet_vgg(self.settings.latent_dim,self.settings)).cuda()
			elif self.settings.net == 'inceptionv3':
				self.generator = DataParallel(Generator_imagenet_inceptionv3(self.settings.latent_dim,self.settings)).cuda()
			elif self.settings.net in ['resnet18', 'MobileV2', 'shufflenet','resnet50']:
				self.generator = DataParallel(Generator_imagenet_resnet18(self.settings.latent_dim,self.settings)).cuda()
		else:
			assert False, "invalid data set"
	
	def _set_trainer(self):
		# set lr master
		lr_master_S = utils.LRPolicy(self.settings.lr_S,
		                           self.settings.nEpochs,
		                           self.settings.lrPolicy_S)
		lr_master_G = utils.LRPolicy(self.settings.lr_G,
									 self.settings.nEpochs,
									 self.settings.lrPolicy_G)
		lr_master_D = utils.LRPolicy(self.settings.lr_D,
									 self.settings.nEpochs,
									 self.settings.lrPolicy_D)

		params_dict_S = {
			'step': self.settings.step_S,
			'decay_rate': self.settings.decayRate_S
		}

		params_dict_G = {
			'step': self.settings.step_G,
			'decay_rate': self.settings.decayRate_G
		}

		params_dict_D = {
			'step': self.settings.step_D,
			'decay_rate': self.settings.decayRate_D
		}
		
		lr_master_S.set_params(params_dict=params_dict_S)
		lr_master_G.set_params(params_dict=params_dict_G)
		lr_master_D.set_params(params_dict=params_dict_D)


		# set trainer
		self.trainer = Trainer(
			model=self.model,
			model_teacher=self.model_teacher,
			generator = self.generator,
			train_loader=self.train_loader,
			test_loader=self.test_loader,
			lr_master_S=lr_master_S,
			lr_master_G=lr_master_G,
			lr_master_D=lr_master_D,
			settings=self.settings,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			run_count=self.start_epoch)

	def quantize_model(self,model):
		"""
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""
		
		weight_bit = self.settings.qw
		act_bit = self.settings.qa
		
		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		
		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
		
		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model(m))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model(mod))
			return q_model
	
	def _replace(self):
		self.model = self.quantize_model(self.model).cuda()
	
	def freeze_model(self,model):
		"""
		freeze the activation range
		"""
		if type(model) == QuantAct:
			model.fix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.freeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.freeze_model(mod)
			return model
	
	def unfreeze_model(self,model):
		"""
		unfreeze the activation range
		"""
		if type(model) == QuantAct:
			model.unfix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.unfreeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.unfreeze_model(mod)
			return model

	def run(self):
		best_top1 = 100
		best_top5 = 100
		start_time = time.time()

		test_error, test_loss, test5_error = self.trainer.test_teacher(0)
		self.logger.info("Teacher network accuracy %.4f%%"	% (100.00 - test_error)	)
		writer = SummaryWriter(self.settings.save_path)

		
		try:
			for epoch in range(self.start_epoch, self.settings.nEpochs):
				self.epoch = epoch
				self.start_epoch = 0

			
				if epoch < 4:
					print ("\n self.unfreeze_model(self.model)\n")
					self.unfreeze_model(self.model)

				self.trainer.train(epoch=epoch, writer=writer)
				#self.trainer.save_G_model("last")
				#self.trainer.save_S_model("last")

				self.freeze_model(self.model)

				if self.settings.dataset in ["cifar100", "cifar10", "imagenet"]:
					test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
				else:
					assert False, "invalid data set"


				if best_top1 >= test_error:
					best_top1 = test_error
					best_top5 = test5_error
					self.trainer.save_G_model("best")
					#self.trainer.save_S_model("best")
				
				self.logger.info("#==>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}".format(best_top1, best_top5))
				self.logger.info("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1,
				                                                                                       100 - best_top5))

		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			traceback.print_exc()
		

		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		return best_top1, best_top5




def main():
	parser = argparse.ArgumentParser(description='Baseline')
	parser.add_argument('--conf_path', type=str, default='./cifar100_resnet20.hocon',
						help='input the path of config file')
	parser.add_argument('--seed', type=int, default=97,metavar='seed',
						help='set the seed')
	parser.add_argument('--id', type=int, default=100,
	                    help='Experiment ID')
	parser.add_argument('--CE_WEIGHT', type=float, default=1,
						help='Set the CE_WEIGHT')
	parser.add_argument('--BNS_WEIGHT', type=float, default=0.1,
						help='Set the BNS_WEIGHT')


	parser.add_argument('--qw', type=int, default=4)
	parser.add_argument('--qa', type=int, default=4)
	
	parser.add_argument('--FEATURE_WEIGHT', type=float, default=1,
						help='Set the FEATURE_WEIGHT')
	parser.add_argument('--warmup_epochs', type=int, default=20,
						help='Set the warmup_epochs')					
	parser.add_argument('--visible_devices', type=str, default='0',
						help='Set the gpu')	
	parser.add_argument('--selenet', type=str, default='',
						help='Set the model_type: bn_vgg16|inceptionv3|resnet18|shufflenet')	
	args = parser.parse_args()
	
	option = Option(args.conf_path)
	option.manualSeed = args.seed
	option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id)
	option.qw = args.qw
	option.qa = args.qa
	
	option.CE_WEIGHT = args.CE_WEIGHT
	option.BNS_WEIGHT = args.BNS_WEIGHT

	option.FEATURE_WEIGHT = args.FEATURE_WEIGHT

	option.warmup_epochs = args.warmup_epochs
	option.visible_devices = args.visible_devices
	option.net = args.selenet

	os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = option.visible_devices
	

	experiment = ExperimentDesign(options=option)
	experiment.run()


if __name__ == '__main__':
	main()
