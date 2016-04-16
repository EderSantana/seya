import numpy as np

import theano
import theano.tensor as T

floatX = theano.config.floatX

from keras.layers.recurrent import Recurrent, GRU, LSTM
from keras import backend as K

tol = 1e-4

def _update_controller(self, inp , h_tm1):
    """We have to update the inner RNN inside the NTM, this
    is the function to do it. Pretty much copy+pasta from Keras
    """

    #1 is for gru, 2 is for lstm
    if len(h_tm1) in [1,2]:
        if hasattr(self.rnn,"get_constants"):
            BW,BU = self.rnn.get_constants(inp)
            h_tm1 += (BW,BU)
    # update state
            
    op_t, h = self.rnn.step(inp, h_tm1)
     
    return op_t  , h

def _update_neural_stack(self, V_tm1, s_tm1, d_t, u_t, v_t, time,stack=True):
    
    ############################################################
    #Equation 1
  
    V_t = V_tm1
                  
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
    """ Stack and queue network
    
    
    output_dim = output dimension
    n_slots = number of memory slot
    m_length = dimention of the memory
    rnn_size = output length of the memory controler
    inner_rnn = "lstm" only lstm is supported 
    stack = True to create neural stack or False to create neural queue
    
    
    from Learning to Transduce with Unbounded Memory
    [[http://arxiv.org/pdf/1506.02516.pdf]]
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
        if inner_rnn != "lstm":
            print "Only lstm is supported"
            raise
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

    def step(self, x, states):
        
        r_tm1, V_tm1,s_tm1,time = states[:4]
        h_tm1 = states[4:]
 
        
        
        r_tm1 = r_tm1
        
        op_t, h_t = _update_controller(self, T.concatenate([x, r_tm1], axis=-1),
                                             h_tm1)
              
       # op_t = op_t  + print_name_shape("W_d",self.W_d.get_value()) 
        op_t = op_t
        #op_t = op_t[:,0,:]
        d_t = K.sigmoid( K.dot(op_t, self.W_d)  + self.b_d)  
        u_t = K.sigmoid(K.dot(op_t, self.W_u) + self.b_u)
        v_t = K.tanh(K.dot(op_t, self.W_v) + self.b_v)
        o_t = K.tanh(K.dot(op_t, self.W_o) + self.b_o) 
        
        
        time = time + 1
        V_t, s_t, r_t = _update_neural_stack(self, V_tm1, s_tm1, d_t[::,0], 
                                             u_t[::,0], v_t,time[0],stack=self.stack)
        

       
        return o_t, [r_t, V_t, s_t, time] + h_t

    
