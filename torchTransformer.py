import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class Log(object):
	def __init__(self):
		self.graph = OrderedDict()
		self.bottoms = OrderedDict()
		self.cur_id = None
	
	# for general layer (should has only one input?)
	def putLayer(self, layer):
		layer_id = id(layer)		
		self.graph[layer_id] = layer
		self.bottoms[layer_id] = [self.cur_id]
		self.cur_id = layer_id
		
	
	def __add__(self, other):
		print("add")
		# merge other branch
		self.graph.update(other.graph)
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
			log = Log()
			# set trans function
			self._raw_flatten = torch.flatten
			torch.flatten = self._trans_flatten
			tmp_model.forward(log)
			

	def _trans_unit(self, model):		
		for module_name in model._modules:			
			# has children
			if len(model._modules[module_name]._modules) > 0:
				self._trans_unit(model._modules[module_name])
			else:							
				unitlayer = UnitLayer(getattr(model, module_name))				
				setattr(model, module_name, unitlayer)			

		return model				
	
	def _trans_flatten(self, input, start_dim = 0, end_dim = -1):
		# input should be log
		print("flatten")		
		layer_name = "faltten_{}".format(len(self.graph))
		input.graph[layer_name] = layer_name
		input.bottoms[layer_name] = [input.cur_id]
		input.cur_id = layer_name
		return input
