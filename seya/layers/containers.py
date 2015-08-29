from __future__ import absolute_import
from __future__ import print_function

import theano.tensor as T
from theano import scan
from keras.layers.core import Layer, Merge
from keras.utils.theano_utils import ndim_tensor, alloc_zeros_matrix

from ..utils import apply_model


class Recursive(Layer):
    '''
        Implement a NN graph with arbitrary layer connections,
        arbitrary number of inputs and arbitrary number of outputs.

        Note: Graph can only be used as a layer
        (connect, input, get_input, get_output)
        when it has exactly one input and one output.

        inherited from Layer:
            - get_params
            - get_output_mask
            - supports_masked_input
            - get_weights
            - set_weights
    '''
    def __init__(self):
        self.namespace = set()  # strings
        self.nodes = {}  # layer-like
        self.inputs = {}  # layer-like
        self.input_order = []  # strings
        self.states = {}  # theano.tensors
        self.state_order = []  # strings
        self.initial_states = []
        self.outputs = {}  # layer-like
        self.output_order = []  # strings
        self.input_config = []  # dicts
        self.state_config = []  # dicts
        self.output_config = []  # dicts
        self.node_config = []  # dicts

        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []

        self.states_map = {}

    @property
    def nb_input(self):
        return len(self.inputs)

    @property
    def nb_output(self):
        return len(self.outputs)

    def set_previous(self, layer, connection_map={}):
        if self.nb_input != layer.nb_output:
            raise Exception('Cannot connect layers: input count does not match output count.')
        if self.nb_input == 1:
            self.inputs[self.input_order[0]].set_previous(layer)
        else:
            if not connection_map:
                raise Exception('Cannot attach multi-input layer: no connection_map provided.')
            for k, v in connection_map.items():
                if k in self.inputs and v in layer.outputs:
                    self.inputs[k].set_previous(layer.outputs[v])
                else:
                    raise Exception('Invalid connection map.')

    def get_input(self, train=False):
        if len(self.inputs) == len(self.outputs) == 1:
            return self.inputs[self.input_order[0]].get_input(train)
        else:
            return dict([(k, v.get_input(train)) for k, v in self.inputs.items()])

    def get_states(self):
        return dict([(k, v) for k, v in self.states.items()])

    @property
    def input(self):
        return self.get_input()

    @property
    def state(self):
        return self.get_states()

    def get_output(self, train=False):
        outputs = self._get_output()
        if len(self.inputs) == len(outputs) == 1:
            return outputs
        else:
            return dict([(k, o) for k, o in zip(self.outputs.keys(), outputs)])

    def add_input(self, name, ndim=2, dtype='float'):
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self.namespace.add(name)
        self.input_order.append(name)
        layer = Layer()  # empty layer
        if dtype == 'float':
            layer.input = ndim_tensor(ndim)
        else:
            if ndim == 2:
                layer.input = T.imatrix()
            else:
                raise Exception('Type "int" can only be used with ndim==2 (Embedding).')
        layer.input.name = name
        self.inputs[name] = layer
        self.input_config.append({'name': name, 'ndim': ndim, 'dtype': dtype})

    def add_state(self, name, dim):
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        self.namespace.add(name)
        self.state_order.append(name)
        batch_size = self.input.values()[0].shape[0]
        self.states[name] = alloc_zeros_matrix(batch_size, dim)
        self.state_config.append({'name': name, 'dim': dim})

    def add_node(self, layer, name, input=None, inputs=[], merge_mode='concat',
                 return_state=None, create_output=False):
        if return_state is None:
            self.initial_states.append(None)
        else:
            self.initial_states.append(self.states[return_state])

        if hasattr(layer, 'set_name'):
            layer.set_name(name)
        if name in self.namespace:
            raise Exception('Duplicate node identifier: ' + name)
        if input:
            if input not in self.namespace:
                raise Exception('Unknown node/input identifier: ' + input)
            if input in self.nodes:
                layer.set_previous(self.nodes[input])
            elif input in self.inputs:
                layer.set_previous(self.inputs[input])
            layer.input_names = [input, ]
        if inputs:
            to_merge = []
            for n in inputs:
                if n in self.nodes:
                    to_merge.append(self.nodes[n])
                elif n in self.inputs:
                    to_merge.append(self.inputs[n])
                elif n in self.states:
                    to_merge.append(self.states[n])
                else:
                    raise Exception('Unknown identifier: ' + n)
            #merge = Merge(to_merge, mode=merge_mode)
            #layer.set_previous(merge)
            layer.input_names = inputs

        self.namespace.add(name)
        self.nodes[name] = layer
        self.node_config.append({'name': name,
                                 'input': input,
                                 'inputs': inputs,
                                 'merge_mode': merge_mode})
        layer.init_updates()
        params, regularizers, constraints, updates = layer.get_params()
        self.params += params
        self.regularizers += regularizers
        self.constraints += constraints
        self.updates += updates

        if create_output:
            self.add_output(name, input=name)
            self.nodes[name].is_output = True

    def get_constants(self):
        return []

    def _step(self, *args):
        local_outputs = {}
        for node in self.nodes:
            local_inputs = []
            for inp in node.input_names:
                idx = self.input_order.index(inp)
                local_inputs.append(args[idx])
                local_inputs.appned(local_outputs[inp])
            for st in node.state_names:
                idx = self.states_order.index(st) + len(self.input_order)
                local_inputs.append(args[idx])
            inputs = T.concatenate(local_inputs, axis=-1)
            local_outputs[node] = self.nodes._get_output(inputs)

        return local_outputs.items()

    def _get_output(self, train=False):
        outputs, updates = scan(self._step,
                                sequences=self.get_input(),
                                outputs_info=self.initial_states,
                                non_sequences=self.parmas + self.get_constants(),
                                truncate_gradient=self.truncate_gradient
                                )
        return outputs

    def add_output(self, name, input=None, inputs=[], merge_mode='concat'):
        if name in self.output_order:
            raise Exception('Duplicate output identifier: ' + name)
        if input:
            if input not in self.namespace:
                raise Exception('Unknown node/input identifier: ' + input)
            if input in self.nodes:
                self.outputs[name] = self.nodes[input]
            elif input in self.inputs:
                self.outputs[name] = self.inputs[input]
        if inputs:
            to_merge = []
            for n in inputs:
                if n not in self.nodes:
                    raise Exception('Unknown identifier: ' + n)
                to_merge.append(self.nodes[n])
            merge = Merge(to_merge, mode=merge_mode)
            self.outputs[name] = merge

        self.output_order.append(name)
        self.output_config.append({'name': name,
                                   'input': input,
                                   'inputs': inputs,
                                   'merge_mode': merge_mode})

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_config": self.input_config,
                "node_config": self.node_config,
                "output_config": self.output_config,
                "input_order": self.input_order,
                "output_order": self.output_order,
                "nodes": dict([(c["name"], self.nodes[c["name"]].get_config()) for c in self.node_config])}
