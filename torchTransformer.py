import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class _ReplaceFunc(object):
    """!
    This Function replace torch functions with self-define Function.
    Inorder to get the imformation of torch model layer infomration.
    """
    def __init__(self, ori_func, replace_func, **kwargs):
        self.torch_func = ori_func
        self.replace_func = replace_func

    def __call__(self, *args, **kwargs):
        out = self.replace_func(self.torch_func, *args, **kwargs)
        return out

class Log(object):
	def __init__(self):
		self.graph = OrderedDict()
		self.bottoms = OrderedDict()
		self.output_shape = OrderedDict()
		self.cur_tensor = None
		self.cur_id = None
	
	# for general layer (should has only one input?)
	def putLayer(self, layer):		
		# force use different address id ( prevent use same defined layer more than once, eg: bottleneck in torchvision)
		layer_id = id(copy.deepcopy(layer))
		self.graph[layer_id] = layer
		self.bottoms[layer_id] = [self.cur_id]		
		self.cur_id = layer_id
	
	def getGraph(self):
		return self.graph
	
	def getBottoms(self):
		return self.bottoms
	
	def getOutShapes(self):
		return self.output_shape
	
	def getTensor(self):
		return self.cur_tensor
	
	def setTensor(self, tensor):
		self.cur_tensor = tensor
		if tensor is not None:
			self.output_shape[self.cur_id] = self.cur_tensor.size()
		else:
			self.output_shape[self.cur_id] = None
	
	def __add__(self, other):
		#print("add")
		# merge other branch		
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		self.output_shape.update(other.output_shape)
		layer_name = "add_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.output_shape[layer_name] = self.cur_tensor.size()
		self.cur_id = layer_name
		# save memory
		del other		
		
		return self		
	
	def __iadd__(self, other):
		#print("iadd")		
		# merge other branch		
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		self.output_shape.update(other.output_shape)
		layer_name = "iadd_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.output_shape[layer_name] = self.cur_tensor.size()
		self.cur_id = layer_name
		# save memory
		del other		
		return self
	
	def __sub__(self, other):
		#print("sub")
		# merge other branch
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		self.output_shape.update(other.output_shape)
		layer_name = "sub_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.output_shape[layer_name] = self.cur_tensor.size()
		self.cur_id = layer_name
		# save memory
		del other
		return self
	
	def __isub__(self, other):
		#print("isub")
		# merge other branch
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		self.output_shape.update(other.output_shape)
		layer_name = "sub_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.output_shape[layer_name] = self.cur_tensor.size()
		self.cur_id = layer_name
		# save memory
		del other
		return self
	
	def __mul__(self, other):
		#print("mul")
		# merge other branch
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		self.output_shape.update(other.output_shape)
		layer_name = "mul_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.output_shape[layer_name] = self.cur_tensor.size()
		self.cur_id = layer_name
		# save memory
		del other
		return self
	
	def __imul__(self, other):
		#print("imul")
		# merge other branch
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		self.output_shape.update(other.output_shape)
		layer_name = "mul_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.output_shape[layer_name] = self.cur_tensor.size()
		self.cur_id = layer_name
		# save memory
		del other
		return self

class UnitLayer(nn.Module):
	def __init__(self, ori_layer):
		super(UnitLayer, self).__init__()
		self.origin_layer = ori_layer
		
	def setOrigin(self, ori_layer):
		self.origin_layer = ori_layer

	# general layer should has only one input?
	def forward(self, log, *args):
		# prevent overwrite log for other forward flow
		cur_log = copy.deepcopy(log)
		cur_log.putLayer(self.origin_layer)
		log_tensor = log.getTensor()
		
		#print("------------------------------------------------")
		#print(self.origin_layer)
		# set as leaf for copy
		out_tensor = self.origin_layer(log_tensor).clone().detach()		
		cur_log.setTensor(out_tensor)
		
		#cur_log.putLayer(self.origin_layer)
		
		return cur_log



class TorchTransformer(nn.Module):
	def __init__(self):
		super(TorchTransformer, self).__init__()

		self._module_graph = OrderedDict()
		self._register_dict = OrderedDict()
		
		self.log = Log()
		
		self._raw_cat = None
		self._raw_split = None
		self._raw_max = None
		self._raw_flatten = None

	# register class to trans
	def register(self, origin_class, target_class):
		pass
	
	def _build_graph(self, model, input_tensor = None):
		self.log.setTensor(input_tensor)
		# if graph is empty, buld
		if not self._module_graph:
			print("set unit")
			# increase memory??
			tmp_model = self._trans_unit(copy.deepcopy(model))
			# tmp_model = self._trans_unit (model)			
			
			print("Log")			
			# set trans function
			self._raw_flatten = torch.flatten
			torch.flatten = _ReplaceFunc(self._raw_flatten, self._trans_flatten)
			self.log = tmp_model.forward(self.log)
	
	def summary(self, model = None, input_tensor = None):
		input_tensor = torch.randn([1, 3, 224, 224])		
		model_graph = self.log.getGraph()
		# if graph empty
		if not model_graph:
			if model is None:
				raise ValueError("Please input model to summary")
			else:
				self._build_graph(model, input_tensor)
		
		# get dicts
		model_graph = self.log.getGraph()
		bottoms_graph = self.log.getBottoms()
		output_shape_graph = self.log.getOutShapes()
		totoal_trainable_params = 0
		total_params = 0
		# loop graph
		print("##########################################################################################")
		line_title = "{:>5}| {:<15} | {:<15} {:<25} {:<15}".format("Index","Layer (type)", "Bottoms","Output Shape", "Param #")
		print(line_title)
		print("---------------------------------------------------------------------------")	
		
		
		for layer_index, key in enumerate(model_graph):	
			#print(model_graph[key])
			#print(bottoms_graph[key])
			# bottom is data
			if bottoms_graph[key][0] == None:				
				# data input
				layer_type = "Data"
				bottoms = ""
				output_shape = ""
				param_num = ""
				data_layer = "{:>5}| {:<15} | {:<15} {:<25} {:<15}".format(layer_index, layer_type, "", "", "0")
				print(data_layer)
				print("---------------------------------------------------------------------------")
				# first layer				
				layer = model_graph[key]
				layer_type = layer.__class__.__name__
				if layer_type == "str":
					layer_type = key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index + 1)
				bottoms = ""
				output_shape = "[{}]".format(tuple(output_shape_graph[key]))
				param_weight_num = 0				
				if hasattr(layer, "weight") and hasattr(layer.weight, "size"):
					param_weight_num += torch.prod(torch.LongTensor(list(layer.weight.size())))
					if layer.weight.requires_grad:
						totoal_trainable_params += param_weight_num
				if hasattr(layer, "bias") and hasattr(layer.weight, "bias"):
					param_weight_num += torch.prod(torch.LongTensor(list(layer.bias.size())))				
					if layer.bias.requires_grad:
						totoal_trainable_params += param_weight_num
				
				total_params += param_weight_num
				new_layer = "{:5}| {:<15} | {:<15} {:<25} {:<15}".format(layer_index+1, layer_type, "Data", output_shape, param_weight_num)
				print(new_layer)
				pass
			else:				
				layer = model_graph[key]
				layer_type = layer.__class__.__name__
				
				if layer_type == "str":
					layer_type = key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index + 1)
				
				bottoms = []
				for b_key in bottoms_graph[key]:
					bottom = model_graph[b_key].__class__.__name__
					if bottom == "str":
						bottom = b_key
					else:
						bottom = bottom + "_{}".format(layer_index + 1)
					
					bottoms.append(bottom)
				#bottoms = ["{}_{}".format(model_graph[b_key].__class__.__name__, list(model_graph.keys()).index(b_key)+1) for b_key in bottoms_graph[key]]				
				if key in output_shape_graph:
					output_shape = "[{}]".format(tuple(output_shape_graph[key]))
				else:
					output_shape = "Error"
				param_weight_num = 0				
				if hasattr(layer, "weight") and hasattr(layer.weight, "size"):
					param_weight_num += torch.prod(torch.LongTensor(list(layer.weight.size())))
					if layer.weight.requires_grad:
						totoal_trainable_params += param_weight_num
				if hasattr(layer, "bias") and hasattr(layer.weight, "bias"):
					param_weight_num += torch.prod(torch.LongTensor(list(layer.bias.size())))				
					if layer.bias.requires_grad:
						totoal_trainable_params += param_weight_num			
				total_params += param_weight_num
				for idx, b in enumerate(bottoms):					
					# if more than one bottom, only print bottom
					if idx == 0:
						#print("00000")
						new_layer = "{:>5}| {:<15} | {:<15} {:<25} {:<15}".format(layer_index+1, layer_type, b, output_shape, param_weight_num)				
					else:
						new_layer = "{:>5}| {:<15} | {:<15} {:<25} {:<15}".format("", "", b, "", "")
					print(new_layer)
			print("---------------------------------------------------------------------------")
		
		
		# total information
		print("==================================================================================")
		print("Total Trainable params: {} ".format(totoal_trainable_params))
		print("Total Non-Trainable params: {} ".format(total_params - totoal_trainable_params))
		print("Total params: {} ".format(total_params))

	def _trans_unit(self, model):		
		for module_name in model._modules:			
			# has children
			if len(model._modules[module_name]._modules) > 0:
				self._trans_unit(model._modules[module_name])
			else:				
				unitlayer = UnitLayer(getattr(model, module_name))				
				setattr(model, module_name, unitlayer)			

		return model
	
	def trans_layers(self, model):
		if len(self._register_dict) == 0:
			print("No layer to swap")
			print("Please use register( {origin_layer}, {target_layer} ) to register layer")
			return model
		else:
			for module_name in model._modules:			
				# has children
				if len(model._modules[module_name]._modules) > 0:
					self.trans_layers(model._modules[module_name])
				else:				
					if (getattr(model, module_name)) in self._register_dict.keys():
						# need to add swap process
						# should think if there is any input arg
						pass	
				
			
	# torch.flatten()
	def _trans_flatten(self, raw_func, log, start_dim = 0, end_dim = -1):
		print("------------------------------------------------")	
		
		log = copy.deepcopy(log)
		
		# input should be log		
		print("flatten")
		
		layer_name = "torchFlatten_{}".format(len(log.getGraph()))
		log.graph[layer_name] = layer_name
		log.bottoms[layer_name] = [log.cur_id]
		log.cur_id = layer_name			
		
		log_tensor = log.getTensor()
		
		# set as leaf for copy					
		out_tensor = raw_func(log_tensor, start_dim = start_dim, end_dim = end_dim).clone().detach()		
		
		log.setTensor(out_tensor)
		
		return log
					
	# torch.max()
	def _trans_max(self, input):
		# input should be log
		print("flatten")		
		layer_name = "torchMax_{}".format(len(input.graph))
		input.graph[layer_name] = layer_name
		input.bottoms[layer_name] = [input.cur_id]
		input.cur_id = layer_name
		return input
					

