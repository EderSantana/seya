import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse

floatX = theano.config.floatX

from keras.layers.recurrent import Recurrent, GRU, LSTM
from keras import backend as K

from seya.utils import rnn_states
tol = 1e-4

def _update_neural_stack(self, V_tm1, s_tm1, d_t, u_t, v_t, time,stack=True):
    
    ############################################################
    #Equation 1
    def print_name_shape(name,x):
        return T.cast( K.sum(theano.printing.Print(name)(x.shape)) * 0,"float32")
    
    V_t = V_tm1 + print_name_shape("V_tm1",V_tm1) + \
                  print_name_shape("s_tm1",s_tm1) + \
                  print_name_shape("d_t",d_t) +\
                  print_name_shape("u_t",u_t) +\
                  print_name_shape("v_t",v_t)
                  
    V_t = T.set_subtensor(V_t[::,time,::],v_t)
        
    ############################################################
    #equation 2
    if stack:
        s_op = T.cumsum(s_tm1[::,1:time][::,::-1],axis=1) #Size t-2
        s_op = s_op[::,::-1]
    #padding
        input_shape = s_op.shape
        output_shape = (input_shape[0],
                        input_shape[1] + 1)
        
        output = T.zeros(output_shape)
        s_op =  T.set_subtensor(output[:, :input_shape[1]], s_op)
    else:
        s_op = T.cumsum(s_tm1[::,:time-1],axis=1) #Size t-2
    #padding
        input_shape = s_op.shape
        output_shape = (input_shape[0],
                        input_shape[1] + 1)
        
        output = T.zeros(output_shape)
        s_op =  T.set_subtensor(output[:, 1:input_shape[1]+1], s_op)
                
    s_op = u_t.dimshuffle(0,"x") - s_op
    
    s_op = T.maximum(s_op,0)
    
    
    #ifelse to deal with time == 0
    #m = T.max()
    #ifelse(T.ge(time,1),time,T.cast(1,"int32"))
 
    s_op = s_tm1[::,:time]-s_op
    
    s_op = T.maximum(s_op,0)
    
    s_t = s_tm1
    
    
    s_t = T.set_subtensor(s_t[::,:time], s_op)
        
    s_t = T.set_subtensor(s_t[::,time], d_t)
    
    
    
    ############################################################
    #equation 3


    if stack:
        s_op = T.cumsum(s_t[::,1:time+1][::,::-1],axis=1) #Size t-1
        s_op = s_op[::,::-1]
        #left padding
        input_shape = s_op.shape
        output_shape = (input_shape[0],
                        input_shape[1] + 1)
        
        output = T.zeros(output_shape)
        s_op =  T.set_subtensor(output[:, :input_shape[1]], s_op)
    else:
        s_op = T.cumsum(s_t[::,:time],axis=1) #Size t-1
        #left padding
        input_shape = s_op.shape
        output_shape = (input_shape[0],
                        input_shape[1] + 1)
        
        output = T.zeros(output_shape)
        s_op =  T.set_subtensor(output[:,1:1+input_shape[1]], s_op)
    
    # Max operation
    s_op = 1 - s_op
    s_op = T.maximum(s_op,0)
            
    #Min operation
    s_op = T.minimum(s_t[::,:time+1],s_op)

    
    r_t = T.sum(s_op[::,:time+1].dimshuffle(0,1,"x")*V_t[::,:time+1,::],axis=1)
    
    return V_t, s_t,r_t

class Stack(Recurrent):
    """ Neural Turing Machines

    Non obvious parameter:
    ----------------------
    shift_range: int, number of available shifts, ex. if 3, avilable shifts are
                 (-1, 0, 1)
    n_slots: number of memory locations
    m_length: memory length at each location

    Known issues:
    -------------
    Theano may complain when n_slots == 1.

    """
    def __init__(self, output_dim, n_slots, m_length,
                 inner_rnn='lstm',rnn_size=64, stack=True,
                 init='glorot_uniform', inner_init='orthogonal',
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.n_slots = n_slots + 1  # because we start at time 1
        self.m_length = m_length
        self.init = init
        self.inner_init = inner_init
        self.inner_rnn = inner_rnn
        self.rnn_size = rnn_size
        self.stack = stack

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(Stack, self).__init__(**kwargs)

    def build(self):
        input_leng, input_dim = self.input_shape[1:]
        self.input = T.tensor3()

        if self.inner_rnn == 'gru':
            self.rnn = GRU(
                activation='relu',
                input_dim=input_dim+self.m_length,
                input_length=input_leng,
                output_dim=self.output_dim, init=self.init,
                inner_init=self.inner_init)
        elif self.inner_rnn == 'lstm':
            self.rnn = LSTM(
                input_dim=input_dim+self.m_length,
                input_length=input_leng,
                output_dim=self.rnn_size, init=self.init,
                forget_bias_init='zero',
                inner_init=self.inner_init)
        else:
            raise ValueError('this inner_rnn is not implemented yet.')

        self.rnn.build()


        self.init_h = K.zeros((self.rnn_size))

        self.W_d = self.rnn.init((self.rnn_size,1))
        self.W_u = self.rnn.init((self.rnn_size,1))

        self.W_v = self.rnn.init((self.rnn_size,self.m_length))
        self.W_o = self.rnn.init((self.rnn_size,self.output_dim))

        self.b_d = K.zeros((1,),name="b_d")
        self.b_u = K.zeros((1,),name="b_u")
        self.b_v = K.zeros((self.m_length,))
        self.b_o = K.zeros((self.output_dim,))

        
        self.trainable_weights = self.rnn.trainable_weights + [
           self.W_d, self.b_d,
            self.W_v, self.b_v,
            self.W_u,  self.b_u,
            self.W_o, self.b_o, self.init_h]

        if self.inner_rnn == 'lstm':
            self.init_c = K.zeros((self.rnn_size))
            self.trainable_weights = self.trainable_weights + [self.init_c, ]
        #self.trainable_weights =[self.W_d]
       

    def get_initial_states(self, X):
        
        
        batch_size = X.shape[0]
        
        init_r = K.zeros((self.m_length)).dimshuffle('x',0).repeat(batch_size,axis=0)
        init_V = K.zeros((self.n_slots,self.m_length)).dimshuffle('x',0,1).repeat(batch_size,axis=0)
        init_S = K.zeros((self.n_slots)).dimshuffle('x',0).repeat(batch_size,axis=0)
        init_h = self.init_h.dimshuffle(('x', 0)).repeat(batch_size, axis=0)

        itime = K.zeros((1,),dtype=np.int32)
        

        if self.inner_rnn == 'lstm':
            init_c = self.init_c.dimshuffle(('x', 0)).repeat(batch_size, axis=0)
            return [init_r , init_V,init_S,itime,init_h,init_c]
      
    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.output_dim
        else:
            return input_shape[0], self.output_dim

    def get_full_output(self, train=False):
        """
        This method is for research and visualization purposes. Use it as
        X = model.get_input()  # full model
        Y = ntm.get_output()    # this layer
        F = theano.function([X], Y, allow_input_downcast=True)
        [memory, read_address, write_address, rnn_state] = F(x)

        if inner_rnn == "lstm" use it as
        [memory, read_address, write_address, rnn_cell, rnn_state] = F(x)

        """
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        assert K.ndim(X) == 3
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitely the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')

        mask = self.get_output_mask(train)
        if mask:
            # apply mask
            X *= K.cast(K.expand_dims(mask), X.dtype)
            masking = True
        else:
            masking = False

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        states = rnn_states(self.step, X, initial_states,
                            go_backwards=self.go_backwards,
                            masking=masking)
        return states

    def step(self, x, states):
        
        r_tm1, V_tm1,s_tm1,time = states[:4]
        h_tm1 = states[4:]
        
        def print_name_shape(name,x):
            return T.cast( K.sum(theano.printing.Print(name)(x.shape)) * 0,"float32")
        
        
        r_tm1 = r_tm1 +  print_name_shape("out\nr_tm1",r_tm1) + \
                          print_name_shape("V_tm1",V_tm1) + \
                          print_name_shape("s_tm1",s_tm1) + \
                          print_name_shape("x",x) + \
                          print_name_shape("h_tm1_0",h_tm1[0]) + \
                          print_name_shape("h_tm1_1",h_tm1[1]) 
                         
        
        op_t, h_t = self._update_controller( T.concatenate([x, r_tm1], axis=-1),
                                             h_tm1)
              
       # op_t = op_t  + print_name_shape("W_d",self.W_d.get_value()) 
        op_t = op_t + print_name_shape("afterop_t",op_t)
        #op_t = op_t[:,0,:]
        ao = K.dot(op_t, self.W_d)  
        ao = ao +print_name_shape("ao",ao)
        d_t = K.sigmoid( ao + self.b_d)  + print_name_shape("afterop2_t",op_t)
        u_t = K.sigmoid(K.dot(op_t, self.W_u) + self.b_u)+ print_name_shape("d_t",op_t)
        v_t = K.tanh(K.dot(op_t, self.W_v) + self.b_v) + print_name_shape("u_t",u_t)
        o_t = K.tanh(K.dot(op_t, self.W_o) + self.b_o) + print_name_shape("v_t",v_t)
        
        o_t = o_t + print_name_shape("afterbulk_t",o_t)
        
        time = time + 1
        V_t, s_t, r_t = _update_neural_stack(self, V_tm1, s_tm1, d_t[::,0], 
                                             u_t[::,0], v_t,time[0],stack=self.stack)
        
        #V_t, s_t, r_t = V_tm1,s_tm1,T.sum(V_tm1,axis = 1)
        V_t  = V_t + print_name_shape("o_t",o_t) + \
                          print_name_shape("r_t",r_t) + \
                          print_name_shape("V_t",V_t) +\
                          print_name_shape("s_t",s_t) 
                        # T.cast( theano.printing.Print("time")(time[0]),"float32")
        #time = T.set_subtensor(time[0],time[0] +)
        
        
       
        return o_t, [r_t, V_t, s_t, time] + h_t



        
    
    def _update_controller(self, inp , h_tm1):
        """We have to update the inner RNN inside the NTM, this
        is the function to do it. Pretty much copy+pasta from Keras
        """
    
        def print_name_shape(name,x,shape=True):
            if shape:
                return T.cast( K.sum(theano.printing.Print(name)(x.shape)) * 0,"float32")
            else:
                return theano.printing.Print(name)(x)
                
        
        
        #1 is for gru, 2 is for lstm
        if len(h_tm1) in [1,2]:
            if hasattr(self.rnn,"get_constants"):
                BW,BU = self.rnn.get_constants(inp)
                h_tm1 += (BW,BU)
        # update state
                
        op_t, h = self.rnn.step(inp + print_name_shape("inp",inp), h_tm1)
    
        
        return op_t + print_name_shape("opt",op_t) +print_name_shape("h",h[0])  +print_name_shape("h",h[1])\
                , h