from inits import *
import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()#兼容1.x版本

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        self.test = []

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, length, placeholders, tag, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support_'+tag]
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.tag = tag
        self.length = length
        # FIX: Store output_dim to explicitly set shape later
        self.output_dim = output_dim

        with tf.variable_scope(self.name+ '_' + self.tag + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
                self.vars['bias_' + str(i)] = tf.zeros(shape=(self.length, 1), name='bias_' + str(i))

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        
        supports = list()
        for i in range(len(self.support)):
            x_dropped = tf.nn.dropout(x, 1-self.dropout)

            if not self.featureless:
                pre_sup = dot(x_dropped, self.vars['weights_' + str(i)])
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup)
            support = support + self.vars['bias_' + str(i)]
            supports.append(support)

        output = tf.add_n(supports)
        
        # FIX: Explicitly set the shape of the output tensor to fix inference issue.
        output.set_shape([self.length, self.output_dim])

        return self.act(output)

class RatLayer():
    def __init__(self, user, item, act=tf.nn.relu):
        self.user = user
        self.item = item
        self.act = act

    def __call__(self):
        rate_matrix = tf.matmul(self.user,tf.transpose(self.item))
        return self.act(rate_matrix)


class RateLayer():
    def __init__(self, user, item, user_dim, item_dim, ac=tf.nn.relu):
        self.user = user
        self.item = item
        self.name = 'RateLayer'
        self.ac = ac
        self.vars = {}
        with tf.name_scope(self.name + '_vars'):
            self.vars['user_latent'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.latent_dim), user_dim],
                                                                       stddev=1.0), name='user_latent_matrix')
            self.vars['item_latent'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.latent_dim), item_dim],
                                                                       stddev=1.0), name='item_latent_matrix')
            self.vars['user_specific'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.output_dim), item_dim],
                                                                         stddev=0.1), name='user_specific')
            self.vars['item_specific'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.output_dim), user_dim],
                                                                         stddev=0.1), name='item_specific')
            self.vars['user_bias'] = tf.zeros(shape=[user_dim,1],name='user_bias')
            self.vars['item_bias'] = tf.zeros(shape=[item_dim,1], name='item_bias')
            self.vars['alpha1'] = tf.Variable(initial_value=1.0, name='alpha1')
            self.vars['alpha2'] = tf.Variable(initial_value=1.0, name='alpha2')

    def __call__(self):
        rate_matrix1 = tf.matmul(tf.transpose(self.vars['user_latent']),self.vars['item_latent'])
        u_matrix = self.vars['alpha1']*(tf.matmul(self.user, self.vars['user_specific'])+self.vars['user_bias'])
        i_matrix = self.vars['alpha2']*(tf.transpose(tf.matmul(self.item,
                                                               self.vars['item_specific'])+self.vars['item_bias']))
        rate_matrix2 = rate_matrix1+u_matrix+i_matrix
        return rate_matrix2

class GraphAttentionLayer(Layer):
    """图注意力层，支持在每种元路径模式内部进行GAT操作"""
    def __init__(self, input_dim, output_dim, length, placeholders, tag, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support_'+tag]
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.tag = tag
        self.length = length
        self.input_dim = input_dim
        self.output_dim = output_dim

        with tf.variable_scope(self.name + '_' + self.tag + '_vars'):
            # 为每种元路径模式创建独立的注意力权重
            self.vars['attention_weights'] = {}
            self.vars['attention_vector'] = {}
            for i in range(len(self.support)):
                self.vars['attention_weights'][i] = glorot([input_dim, output_dim],
                                                      name=f'attention_weights_{i}')
                self.vars['attention_vector'][i] = glorot([2 * output_dim, 1],
                                                     name=f'attention_vector_{i}')
            if bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)

        # 为每种元路径模式计算独立的注意力
        path_outputs = []
        for i in range(len(self.support)):
            # 使用当前元路径模式的注意力权重
            transformed_features = dot(x, self.vars['attention_weights'][i])
            
            # 计算当前元路径模式下的注意力分数
            neighbor_features = dot(self.support[i], transformed_features)
            attention_input = tf.concat([transformed_features, neighbor_features], axis=1)
            attention_score = dot(attention_input, self.vars['attention_vector'][i])
            attention_score = tf.nn.leaky_relu(attention_score)
            
            # 应用注意力权重
            attention_weights = tf.nn.softmax(attention_score, axis=1)
            output = dot(self.support[i], transformed_features)
            output = output * attention_weights
            
            path_outputs.append(output)

        # 合并所有元路径模式的输出
        output = tf.add_n(path_outputs)
        
        # 设置输出形状
        output.set_shape([self.length, self.output_dim])
        
        if self.bias:
            output += self.vars['bias']
            
        return self.act(output)

class SimpleAttLayer():
    def __init__(self, attention_size, tag, time_major=False):
        self.attention_size = attention_size
        self.time_major = time_major
        self.tag = tag
        self.vars = {}

    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)

        if self.time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value

        with tf.variable_scope('v_'+self.tag):
            w_omega = tf.get_variable(initializer=tf.random_normal([64, self.attention_size],
                                                                   stddev=0.1), name='w_omega')
            self.vars['w_omega'] = w_omega
            b_omega = tf.get_variable(initializer=tf.random_normal([self.attention_size], stddev=0.1), name='b_omega')
            self.vars['b_omega'] = b_omega
            u_omega = tf.get_variable(initializer=tf.random_normal([self.attention_size], stddev=0.1), name='u_omega')
            self.vars['u_omega'] = u_omega
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        self.alphas = vu

        output = tf.reduce_sum(inputs*tf.expand_dims(alphas, -1), 0)

        return output
