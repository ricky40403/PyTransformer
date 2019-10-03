import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class Log(object):
	def __init__(self):
		self.graph = OrderedDict()
		self.bottoms = OrderedDict()
		self.output_shape = OrderedDict()
		self.cur_tensor = None
		self.cur_id = None
	
	# for general layer (should has only one input?)
	def putLayer(self, layer):		
		layer_id = id(layer)		
		self.graph[layer_id] = layer
		self.bottoms[layer_id] = [self.cur_id]
		self.cur_id = layer_id
	
	def getGraph(self):
		return self.graph
	
	def getBottoms(self):
		return self.bottoms
	
	def getTensor(self):
		return self.cur_tensor
	
	def setTensor(self, tensor):
		self.cur_tensor = cur_tensor
	
	def __add__(self, other):
		print("add")
		# merge other branch		
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		layer_name = "add_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.cur_id = layer_name
		# save memory
		del other
		return self		
	
	def __iadd__(self, other):
		print("iadd")		
		# merge other branch		
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		layer_name = "add_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.cur_id = layer_name
		# save memory
		del other
		return self
	
	def __sub__(self, other):
		print("sub")
		# merge other branch
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		layer_name = "sub_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.cur_id = layer_name
		# save memory
		del other
		return self
	
	def __isub__(self, other):
		print("isub")
		# merge other branch
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		layer_name = "sub_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.cur_id = layer_name
		# save memory
		del other
		return self
	
	def __mul__(self, other):
		print("mul")
		# merge other branch
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		layer_name = "mul_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
		self.cur_id = layer_name
		# save memory
		del other
		return self
	
	def __imul__(self, other):
		print("imul")
		# merge other branch
		self.graph.update(other.graph)
		self.bottoms.update(other.bottoms)
		layer_name = "mul_{}".format(len(self.graph))
		self.graph[layer_name] = layer_name
		self.bottoms[layer_name] = [self.cur_id, other.cur_id]
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
		log.putLayer(self.origin_layer)	
			
		return copy.deepcopy(log)



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
	
	def _build_graph(self, model):
		# if graph is empty, buld
		if not self._module_graph:
			print("set unit")
			# increase memory??
			tmp_model = self._trans_unit(copy.deepcopy(model))
			# tmp_model = self._trans_unit (model)			
			
			print("Log")			
			# set trans function
			self._raw_flatten = torch.flatten
			torch.flatten = self._trans_flatten
			self.log = tmp_model.forward(self.log)
	
	def summary(self, model = None):
		model_graph = self.log.getGraph()
		# if graph empty
		if not model_graph:
			if model is None:
				raise ValueError("Please input model to summary")
			else:
				self._build_graph(model)
		
		# get graph again
		model_graph = self.log.getGraph()
		bottoms_graph = self.log.getBottoms()
		# loop graph
		print("##########################################################################################")
		line_title = "{:>5}| {:<15} | {:<15} {:>25} {:>15}".format("Index","Layer (type)", "Bottoms","Output Shape", "Param #")
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
				data_layer = "{:>5}| {:<15} | {:<15} {:>25} {:>15}".format(layer_index, layer_type, "","Output Shape", "Param #")
				print(data_layer)
				print("---------------------------------------------------------------------------")
				layer = model_graph[key]
				layer_type = layer.__class__.__name__
				if layer_type == "str":
					layer_type = key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index + 1)


				bottoms = ""
				output_shape = ""
				param_num = ""
				new_layer = "{:5}| {:<15} | {:<15} {:>25} {:>15}".format(layer_index+1, layer_type, "Data","Output Shape", "Param #")
				print(new_layer)
				pass
			else:				
				layer = model_graph[key]
				layer_type = layer.__class__.__name__
				if layer_type == "str":
					layer_type = key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index + 1)
				#print(layer_type)				
				bottoms = ["{}_{}".format(model_graph[b_key].__class__.__name__, list(model_graph.keys()).index(b_key)+1) for b_key in bottoms_graph[key]]				
				output_shape = ""
				param_num = ""
				for idx, b in enumerate(bottoms):					
					# if more than one bottom, only print bottom
					if idx == 0:
						#print("00000")
						new_layer = "{:>5}| {:<15} | {:<15} {:>25} {:>15}".format(layer_index+1, layer_type, b,"Output Shape", "Param #")				
					else:
						new_layer = "{:>5}| {:<15} | {:<15} {:>25} {:>15}".format("", "", b, "", "")
					print(new_layer)
			print("---------------------------------------------------------------------------")
				

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
	def _trans_flatten(self, input, start_dim = 0, end_dim = -1):
		# input should be log
		print("flatten")		
		layer_name = "torchFlatten_{}".format(len(input.graph))
		input.graph[layer_name] = layer_name
		input.bottoms[layer_name] = [input.cur_id]
		input.cur_id = layer_name
		return input
					
	# torch.max()
	def _trans_max(self, input):
		# input should be log
		print("flatten")		
		layer_name = "torchMax_{}".format(len(input.graph))
		input.graph[layer_name] = layer_name
		input.bottoms[layer_name] = [input.cur_id]
		input.cur_id = layer_name
		return input
					

