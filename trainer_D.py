"""
basic trainer
"""
import time
import os
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.utils import save_image 

__all__ = ["Trainer"]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G, lr_master_D,
	             train_loader, test_loader, settings, logger, tensorboard_logger=None,
	             opt_type="SGD", optimizer_state=None, run_count=0):
		"""
		init trainer
		"""
		
		self.settings = settings
		self.model = model
		self.model_teacher = model_teacher
		self.generator = generator
		
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.tensorboard_logger = tensorboard_logger
		self.criterion = nn.CrossEntropyLoss().cuda()
		self.bce_logits = nn.BCEWithLogitsLoss().cuda()
		self.MSE_loss = nn.MSELoss().cuda()
		self.lr_master_S = lr_master_S
		self.lr_master_G = lr_master_G
		self.lr_master_D = lr_master_D
		self.opt_type = opt_type
		if opt_type == "SGD":
			self.optimizer_S = torch.optim.SGD(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				momentum=self.settings.momentum,
				weight_decay=self.settings.weightDecay,
				nesterov=True,
			)
		elif opt_type == "RMSProp":
			self.optimizer_S = torch.optim.RMSprop(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1.0,
				weight_decay=self.settings.weightDecay,
				momentum=self.settings.momentum,
				alpha=self.settings.momentum
			)
		elif opt_type == "Adam":
			self.optimizer_S = torch.optim.Adam(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1e-5,
				weight_decay=self.settings.weightDecay
			)
		else:
			assert False, "invalid type: %d" % opt_type
		if optimizer_state is not None:
			self.optimizer_S.load_state_dict(optimizer_state)\

		self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
											betas=(self.settings.b1, self.settings.b2))
		self.optimizer_D = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_D,
											betas=(self.settings.b1, self.settings.b2))

		self.logger = logger
		self.run_count = run_count
		self.scalar_info = {}
		self.mean_list = []
		self.var_list = []
		self.teacher_running_mean = []
		self.teacher_running_var = []
		self.save_BN_mean = []
		self.save_BN_var = []

		
		self.sample_interval = 400
		self.fix_G = False

		self.CE_WEIGHT = self.settings.CE_WEIGHT	
		self.BNS_WEIGHT = self.settings.BNS_WEIGHT	
		
		self.FEATURE_WEIGHT = self.settings.FEATURE_WEIGHT
		self.warmup_epochs = self.settings.warmup_epochs
		os.makedirs(self.settings.save_path + "/images", exist_ok=True)
		self.image_path = self.settings.save_path + "/images"
	
	def update_lr(self, epoch):
		"""
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
		lr_S = self.lr_master_S.get_lr(epoch)
		lr_G = self.lr_master_G.get_lr(epoch)
		lr_D = self.lr_master_D.get_lr(epoch)
		# update learning rate of model optimizer
		for param_group in self.optimizer_S.param_groups:
			param_group['lr'] = lr_S

		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr_G
		
		for param_group in self.optimizer_D.param_groups:
			param_group['lr'] = lr_D
	
	def loss_fn_kd(self, output, labels, teacher_outputs):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha

		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities! See Issue #2
		"""

		criterion_d = nn.CrossEntropyLoss().cuda()
		kdloss = nn.KLDivLoss().cuda()

		alpha = self.settings.alpha
		T = self.settings.temperature
		a = F.log_softmax(output / T, dim=1)
		b = F.softmax(teacher_outputs / T, dim=1)
		c = (alpha * T * T)
		#d = criterion_d(output, labels)

		KD_loss = kdloss(a,b)*c #+ d
		return KD_loss
	
	def forward(self, images, teacher_outputs, labels=None):
		"""
		forward propagation
		"""
		# forward and backward and optimize
		output,output_1 = self.model(images, True)
		if labels is not None:
			loss = nn.MSELoss(output_1, teacher_outputs)

			# loss = self.loss_fn_kd(output_1, labels, teacher_outputs)
			return output, loss
		else:
			return output, None
	
	def backward_G(self, loss_G):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		loss_G.backward()
		self.optimizer_G.step()

	def backward_S(self, loss_S):
		"""
		backward propagation
		"""
		self.optimizer_S.zero_grad()
		loss_S.backward()
		self.optimizer_S.step()
	
	def backward_D(self, loss_D):
		"""
		backward propagation
		"""
		self.optimizer_D.zero_grad()
		loss_D.backward(retain_graph=True)
		self.optimizer_D.step()

	def backward(self, loss):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		self.optimizer_S.zero_grad()
		self.optimizer_s.zero_grad()
		loss.backward()
		self.optimizer_G.step()
		self.optimizer_S.step()
		self.optimizer_D.step()

	def hook_fn_forward(self,module, input, output):
		input = input[0]
		mean = input.mean([0, 2, 3])
		# use biased var in train
		var = input.var([0, 2, 3], unbiased=False)

		self.mean_list.append(mean)
		self.var_list.append(var)
		self.teacher_running_mean.append(module.running_mean)
		self.teacher_running_var.append(module.running_var)

	def hook_fn_forward_saveBN(self,module, input, output):
		self.save_BN_mean.append(module.running_mean.cpu())
		self.save_BN_var.append(module.running_var.cpu())

	def save_G_model(self, name):
		"""
		save the model
		"""
		torch.save(self.generator.state_dict(), os.path.join(self.settings.save_path, name + "_generator.pth"))

	def save_S_model(self, name):
		"""
		save the model
		"""
		torch.save(self.model.state_dict(), os.path.join(self.settings.save_path, name + "_model.pth"))


	def train(self, epoch, writer):
		"""
		training
		"""
		
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		fp_acc = utils.AverageMeter()
		
		#iters = len(self.train_loader)

		self.update_lr(epoch)

		self.model.eval()
		self.model_teacher.eval()
		self.generator.train()
		
		start_time = time.time()
		end_time = start_time
		iters = 200
		if epoch==0:
			
			for m in self.model_teacher.modules():
				if isinstance(m, nn.BatchNorm2d):
					m.register_forward_hook(self.hook_fn_forward)
		
		for i in range(iters):

			
			noise = Variable(torch.randn(self.settings.batchSize, self.settings.latent_dim)).cuda()

			# Get labels ranging from 0 to n_classes for n rows
			labels = Variable(torch.randint(0, self.settings.nClasses, (self.settings.batchSize,))).cuda()
			noise = noise.contiguous()
			labels = labels.contiguous()

			# ---------------------
			#  Train Generator
			# ---------------------

			fake_images = self.generator(noise, labels)
			
			# One hot loss
			self.mean_list.clear()
			self.var_list.clear()
			self.teacher_running_mean.clear()
			self.teacher_running_var.clear()

			output_teacher_batch, _ = self.model_teacher(fake_images, out_feature=True)
			loss_one_hot = self.criterion(output_teacher_batch, labels)

			# BN statistic loss
			BNS_loss = torch.zeros(1).cuda()

	
			for num in range(len(self.mean_list)):
				BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
					self.var_list[num], self.teacher_running_var[num])

			BNS_loss = BNS_loss / len(self.mean_list)


			loss_G = self.CE_WEIGHT * loss_one_hot + self.BNS_WEIGHT * BNS_loss

			self.backward_G(loss_G)
						


			# ---------------------
			#  Train Quantized Model
			# ---------------------

			output,_= self.model(fake_images.detach(), True)
			loss_C_S = self.criterion(output, labels)
			

			# loss_fea_S = self.loss_fn_kd(output, labels, output_teacher_batch.detach())
			loss_fea_S = nn.MSELoss()(output_teacher_batch.detach(), output)
			loss_S = loss_C_S  + self.FEATURE_WEIGHT * loss_fea_S  
			
			if epoch>= self.settings.warmup_epochs:
				self.backward_S(loss_S)


			single_error, single_loss, single5_error = utils.compute_singlecrop(
				outputs=output, labels=labels,
				loss=loss_S, top5_flag=True, mean_flag=True)
			
			top1_error.update(single_error, fake_images.size(0))
			top1_loss.update(single_loss, fake_images.size(0))
			top5_error.update(single5_error, fake_images.size(0))
			end_time = time.time()
			
			gt = labels.data.cpu().numpy()
			s_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)

			fp_acc.update(s_acc)
			if i==0:
				save_image(fake_images.data[:12], self.image_path + "/stand{}.png".format( epoch), nrow=6, normalize=True)
			if epoch >-1:
				self.scalar_info['accuracy every epoch'] = 100 * s_acc
				self.scalar_info['loss_fea_G'] = loss_G
				self.scalar_info['One-hot loss every epoch'] = loss_one_hot
				self.scalar_info['BNS_loss loss every epoch'] = BNS_loss
				self.scalar_info['C_S loss every epoch'] = loss_C_S	
				self.scalar_info['fea_S loss every epoch'] = loss_fea_S			
				self.scalar_info['S loss every epoch'] = loss_S

				self.scalar_info['training_top1error'] = top1_error.avg
				self.scalar_info['training_top5error'] = top5_error.avg
				self.scalar_info['training_loss'] = top1_loss.avg
		
				#if self.tensorboard_logger is not None:
				for tag, value in list(self.scalar_info.items()):
						writer.add_scalar(tag, value, i + epoch *len(self.train_loader))


		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [one_hot loss: %f] [BNS loss: %f] [G loss: %f] [C_S loss: %f] [fea_S loss: %f] [S loss: %f]  "
			% (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, loss_one_hot.item(), BNS_loss.item(), loss_G.item(), loss_C_S.item(), loss_fea_S.item(),  loss_S.item())
		)
				

		return top1_error.avg, top1_loss.avg, top5_error.avg

	def test(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				
				labels = labels.cuda()
				images = images.cuda()
				output = self.model(images)

				loss = torch.ones(1)
				self.mean_list.clear()
				self.var_list.clear()

				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))
				
				end_time = time.time()
		
		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
		)
		
		self.scalar_info['testing_top1error'] = top1_error.avg
		self.scalar_info['testing_top5error'] = top5_error.avg
		self.scalar_info['testing_loss'] = top1_loss.avg
		if self.tensorboard_logger is not None:
			for tag, value in self.scalar_info.items():
				self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
			self.scalar_info = {}
		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg

	def test_teacher(self, epoch):
		"""
		testing
		"""
		
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()

		self.model_teacher.eval()

		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):


				start_time = time.time()
				data_time = start_time - end_time

				labels = labels.cuda()
				if self.settings.tenCrop:
					image_size = images.size()
					images = images.view(
						image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
					images_tuple = images.split(image_size[0])
					output = None
					for img in images_tuple:
						if self.settings.nGPU == 1:
							img = img.cuda()
						img_var = Variable(img, volatile=True)
						temp_output, _ = self.forward(img_var)
						if output is None:
							output = temp_output.data
						else:
							output = torch.cat((output, temp_output.data))
					single_error, single_loss, single5_error = utils.compute_tencrop(
						outputs=output, labels=labels)
				else:
					if self.settings.nGPU == 1:
						images = images.cuda()

					output = self.model_teacher(images)

					loss = torch.ones(1)
					self.mean_list.clear()
					self.var_list.clear()

					single_error, single_loss, single5_error = utils.compute_singlecrop(
						outputs=output, loss=loss,
						labels=labels, top5_flag=True, mean_flag=True)
				#
				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))

				end_time = time.time()
				iter_time = end_time - start_time

		print(
				"Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
		)

		self.run_count += 1



		return top1_error.avg, top1_loss.avg, top5_error.avg
