import copy
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect 

from graphviz import Digraph
import pydot

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
		# # def wrapper(*args, **kwargs):
		# # 	out = self.replace_func(self.torch_func, *args, **kwargs)
		# # 	return out
		# # return wrapper
		# # # has args
		# if len(args) > 1:
		# 	# print(args)
		# 	# print(kwargs)
		# 	if len(kwargs) > 0:
		# 		out = self.replace_func(self.torch_func, args, kwargs)
		# 	else:
		# 		out = self.replace_func(self.torch_func, args)
		# # has only input log
		# elif len(args) == 1:
		# 	args = args[0]			
		# 	if len(kwargs) > 0:
		# 		out = self.replace_func(self.torch_func, args, kwargs)
		# 	else:
		# 		out = self.replace_func(self.torch_func, args)
		# else:
		# 	if len(kwargs) > 0:
		# 		out = self.replace_func(self.torch_func, kwargs)
		# 	else:
		# 		out = self.replace_func(self.torch_func)
		
		# return out

class Log(object):
	def __init__(self):
		self.graph = OrderedDict()
		self.bottoms = OrderedDict()
		self.output_shape = OrderedDict()
		self.cur_tensor = None
		self.cur_id = None
		self.tmp_list = None
		self.log_init()

	def __len__(self):
		# one log
		return 1

	def reset(self):
		self.graph = OrderedDict()
		self.bottoms = OrderedDict()
		self.output_shape = OrderedDict()
		self.cur_tensor = None
		self.cur_id = None
		self.tmp_list = []
		self.log_init()
	
	  
	# add data input layer to log
	def log_init(self):
		layer_id = "Data"
		self.graph[layer_id] = layer_id
		self.bottoms[layer_id] = None
		self.output_shape[layer_id] = ""
		self.cur_id = layer_id
		self.tmp_list = []

  
	# for general layer (should has only one input?)
	def putLayer(self, layer):		
		# force use different address id ( prevent use same defined layer more than once, eg: bottleneck in torchvision)
		# tmp_layer = copy.deepcopy(layer)
		layer_id = id(layer)
		self.tmp_list.append(layer)
		layer_id = id(self.tmp_list[-1])
		if layer_id in self.graph:
			tmp_layer = copy.deepcopy(layer)
			self.tmp_list.append(tmp_layer)
			# layer_id = id(self.tmp_list[-1])
			layer_id = id(tmp_layer)

		# if layer_id in self.graph:
		# 	print("FK....................................")

		self.graph[layer_id] = layer
		self.bottoms[layer_id] = [self.cur_id]
		self.cur_id = layer_id
		# del layer, tmp_layer, layer_id
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
	
	
	# handle tensor operation(eg: tensor.view)
	def __getattr__(self, name):		
		if name == "__deepcopy__" or name == "__setstate__":
			return object.__getattribute__(self, name)			
		# if get data => get cur_tensor.data
		elif name == "data":
			return self.cur_tensor.data		
		
		elif hasattr(self.cur_tensor, name):			
			def wrapper(*args, **kwargs):				
				func = self.cur_tensor.__getattribute__(name)
				out_tensor = func(*args, **kwargs)
				# print(type(out_tensor))
				# print(out_tensor.size())
				# print(len(out_tensor))
				# if multi output:
				# if len(out_tensor) > 1:
				if not isinstance(out_tensor, torch.Tensor):
					out_logs = []
					for t in out_tensor:
						out_log = copy.deepcopy(self)
						out_log.setTensor(t)						
						out_logs.append(out_log)
						
					# print("multi Tensor return : {}".format(out_logs))
					return out_logs
				else:						
					# print("///////////////////////////////////////////////////")
					self.cur_tensor = out_tensor
					self.output_shape[self.cur_id] = out_tensor.size() 
					
					# print("Tensor return : {}".format(out_tensor.size()))
					return self
			# print(wrapper)
			return wrapper
			
			# return self


		else:
			return object.__getattribute__(self, name)			
		
	
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


	def size(self, dim=None):
		return self.cur_tensor.size(dim) if dim is not None else self.cur_tensor.size()



class UnitLayer(nn.Module):
	def __init__(self, ori_layer):
		super(UnitLayer, self).__init__()
		self.origin_layer = ori_layer
		

	def setOrigin(self, ori_layer):
		self.origin_layer = ori_layer


	# general layer should has only one input?
	def forward(self, log, *args):
		# print("1233312")
		# print(log)
		# print("Helloooooooooooooooooooooooooooooooooooooooooo")
		# print(self.origin_layer)

		# prevent overwrite log for other forward flow
		cur_log = copy.deepcopy(log)
		# print(cur_log)
		cur_log.putLayer(self.origin_layer)
		
		# print(log.cur_tensor)
		log_tensor = log.getTensor()
		# print(log.cur_id)
		# print(log_tensor.size())
		# print("------------------------------------------------")
		# print(len(cur_log.graph))
		# print(cur_log.cur_id)
		# print("Helloooooooooooooooooooooooooooooooooooooooooo")	
		# print(self.origin_layer)
		# print(cur_log.getG())
		# print(log_tensor.size())
		# print(type(log_tensor))
		# print("------------------------------------------------")
		# print(self.origin_layer)
		# set as leaf for copy
		out_tensor = self.origin_layer(log_tensor).clone().detach()		
		cur_log.setTensor(out_tensor)

		#cur_log.putLayer(self.origin_layer)
		# print("UniLayer Return : {}".format(cur_log.cur_tensor.size()))
		# print("UniLayer Return : {}".format(cur_log))
		# print("///////////////////////////////////////////////////////")
		return cur_log


class TorchTransformer(nn.Module):
	def __init__(self):
		super(TorchTransformer, self).__init__()

		# self._module_graph = OrderedDict()
		self._register_dict = OrderedDict()

		self.log = Log()
		
		self._raw_TrochFuncs = OrderedDict()
		self._raw_TrochFunctionals = OrderedDict()
		self._raw_cat = None		
		self._raw_max = None
		self._raw_flatten = None
		self._raw_split = None
		self._raw_transpose = None


	# register class to trans
	def register(self, origin_class, target_class):
		print("register", origin_class, target_class)
		self._register_dict[origin_class] = target_class
		pass
	
	def _build_graph(self, model, input_tensor = None):
		# reset log
		self.log = Log()		
		# add Data input
		self.log.setTensor(input_tensor)		


		tmp_model = self._trans_unit(copy.deepcopy(model))
		
		# accept_functions = ["flatten", "split", "relu_", "cat", "adaptive_avg_pool2d"]
		# for f in dir(torch):
		# 	if isinstance(getattr(torch, f) ,types.BuiltinFunctionType):
		# 		print(f)
		# sys.exit()
		for f in dir(torch):
			# if private function, pass
			if f.startswith("_"):
				continue
			if isinstance(getattr(torch, f) ,types.BuiltinMethodType) or isinstance(getattr(torch, f) ,types.BuiltinFunctionType):			
				self._raw_TrochFuncs[f] = getattr(torch, f)
				setattr(torch, f, _ReplaceFunc(getattr(torch,f), self._torchFunctions))
    
		# accept_functionals = ["adaptive_avg_pool2d", "linear"]
		# for f in dir(F):
		# 	if isinstance(getattr(F, f) ,types.FunctionType):
		# 		print(f)
		# sys.exit()
		for f in dir(F):
			if f.startswith("_"):
				continue
			# if isinstance(getattr(F, f) ,types.FunctionType):
			# 	print(f)
			# print(type(getattr(F, '_list_with_default')))
			# print(isinstance(getattr(F, 'adaptive_avg_pool2d') ,types.FunctionType))
			# sys.exit()
			if isinstance(getattr(F, f) ,types.BuiltinMethodType) or isinstance(getattr(F, f) ,types.BuiltinFunctionType) or isinstance(getattr(F, f) ,types.FunctionType):
				# if private function, pass
				
				# if isinstance(getattr(F, f), types.FunctionType):
				self._raw_TrochFunctionals[f] = getattr(F, f)
				setattr(F, f, _ReplaceFunc(getattr(F,f), self._torchFunctionals))
				
		
		# # set trans function
		# self._raw_cat = torch.cat
		# torch.cat = _ReplaceFunc(self._raw_cat, self._trans_cat)			
		# self._raw_max = torch.max
		# torch.max = _ReplaceFunc(self._raw_max, self._trans_max)
		# self._raw_flatten = torch.flatten
		# torch.flatten = _ReplaceFunc(self._raw_flatten, self._trans_flatten)
		# self._raw_split = torch.split
		# torch.split = _ReplaceFunc(self._raw_split, self._trans_split)
		# self._raw_transpose = torch.transpose
		# torch.transpose = _ReplaceFunc(self._raw_transpose, self._trans_transpose)

		# print(tmp_model)
		# sys.exit()
		# forward to generate log
		self.log = tmp_model.forward(self.log)
		# print(self.log)

		# reset back 
		for f in self._raw_TrochFuncs:
			setattr(torch, f, self._raw_TrochFuncs[f])
   
		for f in self._raw_TrochFunctionals:
			setattr(F, f, self._raw_TrochFunctionals[f])
   
		# torch.cat = self._raw_cat
		# torch.max = self._raw_max
		# torch.flatten = self._raw_flatten
		# torch.split = self._raw_split
		del tmp_model
	
	def summary(self, model = None, input_tensor = None):
		input_tensor = torch.randn([1, 3, 224, 224])
		# model_graph = self.log.getGraph()
		# if graph empty		
		# if model is None:
		# 	# check if use self modules
		# 	if len(self._modules) > 0:
		# 		self._build_graph(self, input_tensor)	
		# 	else:
		# 		raise ValueError("Please input model to summary")
		# else:
		# 	self._build_graph(model, input_tensor)
		self._build_graph(model, input_tensor)
   
		# get dicts and variables
		model_graph = self.log.getGraph()
		bottoms_graph = self.log.getBottoms()
		output_shape_graph = self.log.getOutShapes()
		# store top names for bottoms
		topNames = OrderedDict()		
		totoal_trainable_params = 0
		total_params = 0
		# loop graph
		print("##########################################################################################")
		line_title = "{:>5}| {:<15} | {:<15} {:<25} {:<15}".format("Index","Layer (type)", "Bottoms","Output Shape", "Param #")
		print(line_title)
		print("---------------------------------------------------------------------------")	
		# for key in model_graph:
		# 	print(model_graph[key])
		
		for layer_index, key in enumerate(model_graph):	
			
			# data layer
			if bottoms_graph[key] is None:
				# Layer information
				layer = model_graph[key]
				layer_type = layer.__class__.__name__
				if layer_type == "str":
					layer_type = key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index)
				
				topNames[key] = layer_type

				# Layer Output shape
				output_shape = "[{}]".format(tuple(output_shape_graph[key]))
				
				# Layer Params
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
				
				new_layer = "{:5}| {:<15} | {:<15} {:<25} {:<15}".format(layer_index+1, layer_type, "", output_shape, param_weight_num)
				print(new_layer)
				
			else:
				# Layer Information
				layer = model_graph[key]
				layer_type = layer.__class__.__name__
				
				# add, sub, mul...,etc. (custom string)
				if layer_type == "str":
					# the key should be XXX_{idx_prevent_duplicate}
					tmp_key = key.split("_")
					tmp_key[-1] = "_{}".format(layer_index)	
					tmp_key = "".join(tmp_key)
					layer_type = tmp_key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index)

				topNames[key] = layer_type
				# print(key)
				# print(layer_type," ", bottoms_graph[key])
    			# print(bottoms_graph[key])
    
				# Layer Bottoms
				bottoms = []
				for b_key in bottoms_graph[key]:
					bottom = topNames[b_key]				
					bottoms.append(bottom)
				
				# Layer Output Shape
				if key in output_shape_graph:
					output_shape = "[{}]".format(tuple(output_shape_graph[key]))
				else:
					output_shape = "None"
				
				# Layer Params
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
				
				# Print (one bottom a line)
				for idx, b in enumerate(bottoms):					
					# if more than one bottom, only print bottom
					if idx == 0:						
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
  
		# del model_graph, bottoms_graph, output_shape_graph, topNames
		return model

	def visualize(self, model = None, input_tensor = None, save_name = None, graph_size = 30):
		input_tensor = torch.randn([1, 3, 224, 224])
		# model_graph = self.log.getGraph()
		
		# if graph empty		
		if model is None:
			# check if use self modules
			if len(self._modules) > 0:
				self._build_graph(self, input_tensor)	
			else:
				raise ValueError("Please input model to visualize")
		else:
			self._build_graph(model, input_tensor)
		
		# graph 
		node_attr = dict(style='filled',
						 shape='box',
						 align='left',
						 fontsize='30',
						 ranksep='0.1',
						 height='0.2')
		
		dot = Digraph(node_attr=node_attr, graph_attr=dict(size="{},{}".format(graph_size, graph_size)))	

		# get dicts and variables
		model_graph = self.log.getGraph()
		# print(model_graph)
		bottoms_graph = self.log.getBottoms()
		# print(bottoms_graph)
		for layer_index, key in enumerate(model_graph):
			# Input Data layer
			if bottoms_graph[key] is None:
				layer = model_graph[key]
				layer_type = layer.__class__.__name__				
				# add, sub, mul...,etc. (custom string)
				if layer_type == "str":
					layer_type = key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index)
					
				dot.node(str(key), layer_type, fillcolor='orange')			
			else:
				# Layer Information
				layer = model_graph[key]
				layer_type = layer.__class__.__name__				
				# add, sub, mul...,etc. (custom string)
				if layer_type == "str":
					# the key should be XXX_{idx_prevent_duplicate}
					tmp_key = key.split("_")
					tmp_key[-1] = "_{}".format(layer_index)	
					tmp_key = "".join(tmp_key)
					layer_type = tmp_key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index)
				
				# layer_type = layer_type
				# print("Layer: {}".format(layer_type))
				# print("Key: {}".format(key))
				dot.node(str(key), layer_type, fillcolor='orange')				
				# link bottoms
				# print("Bottoms: ")
				for bot_key in bottoms_graph[key]:
					# print(bot_key)
					dot.edge(str(bot_key), str(key))				
		
		# return graph
		if save_name is not None:
			(graph,) = pydot.graph_from_dot_data(dot.source)
			graph.write_png(save_name + ".png" )
		return dot
		
	def _trans_unit(self, model):
		# print("TRNS_UNIT")
		for module_name in model._modules:
			# has children
			if len(model._modules[module_name]._modules) > 0:
				self._trans_unit(model._modules[module_name])
			else:				
				unitlayer = UnitLayer(getattr(model, module_name))
				setattr(model, module_name, unitlayer)

		return model
	

	def trans_layers(self, model):
		# print("trans layer")
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
					if type(getattr(model, module_name)) in self._register_dict:
						# use inspect.signature to know args and kwargs of __init__
						_sig = inspect.signature(type(getattr(model, module_name)))
						_kwargs = {}
						for key in _sig.parameters:
							if _sig.parameters[key].default == inspect.Parameter.empty: #args 
								# assign args
								# default values should be handled more properly, unknown data type might be an issue
								if 'kernel' in key:
									# _sig.parameters[key].replace(default=inspect.Parameter.empty, annotation=3)
									value = 3
								elif 'channel' in key:
									# _sig.parameters[key].replace(default=inspect.Parameter.empty, annotation=32)
									value = 32
								else:
									# _sig.parameters[key].replace(default=inspect.Parameter.empty, annotation=None)
									value = None
						
								_kwargs[key] = value

						_attr_dict = getattr(model, module_name).__dict__
						_layer_new = self._register_dict[type(getattr(model, module_name))](**_kwargs) # only give positional args
						_layer_new.__dict__.update(_attr_dict)

						setattr(model, module_name, _layer_new)
	
	

	# def _trans_cat(self, raw_func, logs, dim=0, out=None):
	# 	# input should be log		
	# 	# copy log to prevent overwrite
	# 	cur_log = copy.deepcopy(logs[0])
	# 	cur_bottom = [cur_log.cur_id]
	# 	tensors = [cur_log.getTensor()]
	# 	if len(logs) > 1:			
	# 		# cur_graph = cur_log.getGraph()
	# 		for log in logs[1:]:
	# 			# tmp_graph = log.getGraph()
	# 			cur_log.graph.update(log.graph)
	# 			cur_log.bottoms.update(log.bottoms)
	# 			cur_log.output_shape.update(log.output_shape)
	# 			cur_bottom.append(log.cur_id)
	# 			tensors.append(log.cur_tensor)
	
	# 	# Layer information		
	# 	layer_name = "torchCat_{}".format(len(cur_log.graph))
	# 	cur_log.graph[layer_name] = layer_name
	# 	cur_log.bottoms[layer_name] = cur_bottom
	# 	cur_log.cur_id = layer_name		
		
	# 	# fot output shape
	# 	# handle tensor operation
	# 	# log_tensor = cur_log.getTensor()		
		
	# 	# set as leaf for copy					
	# 	out_tensor = raw_func(tuple(tensors), dim=dim, out=out).clone().detach()		
	# 	cur_log.setTensor(out_tensor)
		
	# 	return cur_log
	
	# def _trans_transpose(self, raw_func, log, dim0, dim1):
	# 	log = copy.deepcopy(log)
	# 	layer_name = "torchTranspose_{}".format(len(log.getGraph()))
	# 	log.graph[layer_name] = layer_name
	# 	log.bottoms[layer_name] = [log.cur_id]
	# 	log.cur_id = layer_name
	# 	log_tensor = log.getTensor()		
	# 	# set as leaf for copy					
	# 	out_tensor = raw_func(log_tensor, dim0, dim1).clone().detach()		
	# 	log.setTensor(out_tensor)
		
	# 	return log
  
  
	# # torch.flatten()
	# def _trans_flatten(self, raw_func, log, start_dim = 0, end_dim = -1):		
	# 	# input should be log		
	# 	# copy log to prevent overwrite
	# 	log = copy.deepcopy(log)
		
	# 	# Layer information		
	# 	layer_name = "torchFlatten_{}".format(len(log.getGraph()))
	# 	log.graph[layer_name] = layer_name
	# 	log.bottoms[layer_name] = [log.cur_id]
	# 	log.cur_id = layer_name			
		
	# 	# fot output shape
	# 	# handle tensor operation
	# 	log_tensor = log.getTensor()		
	# 	# set as leaf for copy					
	# 	out_tensor = raw_func(log_tensor, start_dim = start_dim, end_dim = end_dim).clone().detach()		
	# 	log.setTensor(out_tensor)
		
	# 	return log
					
	# # torch.max()
	# def _trans_max(self, raw_func, log):	
	# 	# input should be log		
	# 	# copy log to prevent overwrite
	# 	log = copy.deepcopy(log)	
		
	# 	# Layer information		
	# 	layer_name = "torchMax_{}".format(len(log.getGraph))
	# 	log.graph[layer_name] = layer_name
	# 	log.bottoms[layer_name] = [log.cur_id]
	# 	log.cur_id = layer_name		
		
	# 	# fot output shape
	# 	# handle tensor operation
	# 	log_tensor = log.getTensor()		
	# 	# set as leaf for copy					
	# 	out_tensor = raw_func(log_tensor).clone().detach()		
	# 	log.setTensor(out_tensor)
		
	# 	return log
	
	# # torch.split()
	# def _trans_split(self, raw_func, log, split_size_or_sections, dim=0):	
	# 	# input should be log		
	# 	# copy log to prevent overwrite
	# 	log = copy.deepcopy(log)	
		
	# 	# Layer information		
	# 	layer_name = "torchSplit_{}".format(len(log.getGraph))
	# 	log.graph[layer_name] = layer_name
	# 	log.bottoms[layer_name] = [log.cur_id]
	# 	log.cur_id = layer_name		
		
	# 	# fot output shape
	# 	# handle tensor operation
	# 	log_tensor = log.getTensor()		
	# 	# set as leaf for copy					
	# 	out_tensor = raw_func(log_tensor, split_size_or_sections, dim = dim).clone().detach()		
	# 	log.setTensor(out_tensor)
		
	# 	return log

	def _torchFunctions(self, raw_func, *args, **kwargs):
		# print("Torch function")
		function_name = raw_func.__name__		
		# print(raw_func.__name__)


		# torch function may has no input
		# so check first
		
		if len(args) > 0:
			logs = args[0]
			cur_args = args[1:]
		elif len(kwargs) > 0:
			
			return raw_func(**kwargs)
		else:			
			return raw_func()

		# check is user used or in torch function call
		is_tensor_in = False
		# tensor input		
		if (len(logs) > 1) and (type(logs[0]) == torch.Tensor):
			cur_inputs = logs
			is_tensor_in = True
			return raw_func(*args, **kwargs)
					
		elif (len(logs) ==1) and (type(logs) == torch.Tensor):
			
			cur_inputs = logs	
			is_tensor_in = True			
			return raw_func(*args, **kwargs)
		
		# log input
		else:			
			# multi inputs
			bottoms = []
			cur_inputs = []				
			if len(logs) > 1:				
				cur_log = logs[0]
				for log in logs:					
					cur_inputs.append(log.cur_tensor)
					bottoms.append(log.cur_id)
					# update informations
					cur_log.graph.update(log.graph)					
					cur_log.bottoms.update(log.bottoms)
					cur_log.output_shape.update(log.output_shape)
				cur_inputs = tuple(cur_inputs)
			# one input
			else:
				cur_log = logs
				cur_inputs = cur_log.cur_tensor
				bottoms.append(cur_log.cur_id)
				
		# replace logs to tensor as function inputs to get output tensor
		args = list(args)
		args[0] = cur_inputs
		args = tuple(args)
		# send into origin functions
		out_tensor = raw_func(*args, **kwargs).clone().detach()
		
		# if function call, just return out tensor
		if is_tensor_in:
			return out_tensor

		# most multi input change to one output
		# most multi output has one input
		# if shape change
		# store theese types of  opreation as a layer
		if (len(logs) > 1) or (len(out_tensor) > 1) or (logs.cur_tensor.size() != out_tensor.size()):
			layer_name = "torch.{}_{}".format(function_name, len(cur_log.graph))
			cur_log.graph[layer_name] = layer_name
			cur_log.bottoms[layer_name] = bottoms
			cur_log.cur_id = layer_name
		
		# multi output
		if not isinstance(out_tensor , torch.Tensor):
			# print("multi output")				
			out_logs = []
			for t in out_tensor:				
				out_log = copy.deepcopy(cur_log)
				out_log.setTensor(t)			
				out_logs.append(out_log)
			
			# sometimes will has (out, ) and this lens is >1
			if len(out_logs) == 1:
				out_logs = out_logs[0]
				# print("Torch Return : {}".format(out_logs.cur_tensor.size()))
			
			# print("Torch Return : {}".format(out_logs.cur_tensor.size()))
			return out_logs

		else:			
			# print("single output")
			# print("Torch Return : {}".format(out_tensor.size()))
			cur_log.setTensor(out_tensor)   			
			return cur_log
		

	# torch.functionals
	def _torchFunctionals(self, raw_func, *args, **kwargs):	
     
		# print("Functional")
		function_name = raw_func.__name__		
		# print(raw_func.__name__)		

		# functional has input expect affine_grid
		if function_name == "affine_grid":
			pass
		else:
			logs = args[0]
			cur_args = args[1:]
		
		# check is user used or in torch function call
		is_tensor_in = False
		# tensor input
		if (len(logs) > 1) and (type(logs[0]) == torch.Tensor):
			# print(logs[0].size(), logs[1].size())
			cur_inputs = logs
			is_tensor_in = True
			out = raw_func(*args, **kwargs)
			# print("Functional return : {}".format(out.size()))
			return raw_func(*args, **kwargs)

		elif (len(logs) ==1) and (type(logs) == torch.Tensor):					
			cur_inputs = logs	
			is_tensor_in = True			
			out = raw_func(*args, **kwargs)
			# print("Functional return : {}".format(out.size()))
			return raw_func(*args, **kwargs)
		
		# log input
		else:			
			# multi inputs
			bottoms = []
			cur_inputs = []				
			if len(logs) > 1:				
				cur_log = logs[0]
				for log in logs:					
					cur_inputs.append(log.cur_tensor)
					bottoms.append(log.cur_id)
					# update informations
					cur_log.graph.update(log.graph)					
					cur_log.bottoms.update(log.bottoms)
					cur_log.output_shape.update(log.output_shape)
				cur_inputs = tuple(cur_inputs)
			# one input
			else:
				cur_log = logs
				cur_inputs = cur_log.cur_tensor
				bottoms.append(cur_log.cur_id)

		
			
		# replace logs to tensor as function inputs to get output tensor
		args = list(args)
		args[0] = cur_inputs
		args = tuple(args)
		# send into origin functions
		out_tensor = raw_func(*args, **kwargs).clone().detach()
		
		# if function call, just return out tensor
		if is_tensor_in:
			return out_tensor

		# if log input and is function type, store as an layer
		if isinstance(raw_func, types.FunctionType):			
			# use multiple address as name to prevent duplicate address
			layer_name = "F.{}_{}{}{}".format(function_name, id(out_tensor), id(args), id(kwargs))
			# replace with new address if still duplicate
			if layer_name in cur_log.graph:
				tmp_list = []
				tmp_list.append(out_tensor)
				tmp_tensor = copy.deepcopy(tmp_list[-1])
				# should not duplicate again?
				layer_name = "F.{}_{}{}{}".format(function_name, id(tmp_tensor), id(args), id(kwargs))				
			
			cur_log.graph[layer_name] = layer_name				
			cur_log.bottoms[layer_name] = bottoms
			cur_log.cur_id = layer_name			
		
		# if multi-output
		# if len(out_tensor) > 1:
		if not isinstance(out_tensor, torch.Tensor):
			out_logs = []
			for t in out_tensor:				
				out_log = copy.deepcopy(cur_log)
				out_log.setTensor(t)			
				out_logs.append(out_log)	
			
			return out_logs
		else:			
			cur_log.setTensor(out_tensor)
			return cur_log


