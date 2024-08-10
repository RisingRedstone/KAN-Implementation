@tf.function
def expand_dims_with_index_values(self,indices):
    expanded_dims = tf.shape(indices)
    extra_dim = tf.gather_nd(tf.shape(expanded_dims), [0])
    result_dims = tf.concat([expanded_dims, [extra_dim]], 0)
    num_dims = tf.gather_nd(tf.shape(expanded_dims), [0])
    index = tf.constant(np.zeros(expanded_dims.shape[0]), dtype = np.int32)
    
    result = tf.constant(np.zeros( result_dims.eval(session=tf.compat.v1.Session()) ), dtype = np.int32)
    i = tf.constant(0)
    j = tf.constant(0)
    while(tf.gather_nd(index, [0]) < tf.gather_nd(expanded_dims, [0])):
        i = 0
        temp0 = tf.tensor_scatter_nd_update(index, [[num_dims-1]], [tf.gather_nd(indices, index)])
        result = tf.tensor_scatter_nd_update(result, [index], [temp0])

        i = tf.gather_nd(tf.shape(index), [0])-1
        j = 1
        while ((i > 0) & (j >= 1)):
            index = tf.tensor_scatter_nd_add(index, [[i]], [1])
            j = 0
            if(tf.gather_nd(index, [i]) >= tf.gather_nd(expanded_dims, [i])):
                index = tf.tensor_scatter_nd_update(index, [[i]], [0])
                j = 1
            i-=1
        
        if(j == 1):
            index = tf.tensor_scatter_nd_add(index, [[0]], [1])
    return result

@tf.function
def dot_matmul(self, A, B):
    expanded_dims_A = tf.shape(A).eval(session = tf.compat.v1.Session())
    expanded_dims_B = tf.shape(B).eval(session = tf.compat.v1.Session())
    prune_dims_A = expanded_dims_A[:-1]
    prune_dims_B = expanded_dims_B[:-1]
    num_dims = expanded_dims_A.shape[0]
    A0 = tf.reshape(tf.gather(A, [0], axis = (num_dims-1) ), prune_dims_A)
    B0 = tf.reshape(tf.gather(B, [0], axis = (num_dims-1) ), prune_dims_B)
    Y = tf.matmul(A0, B0)
    i = tf.constant(1)
    while(i < expanded_dims_A[-1]):
        A0 = tf.reshape(tf.gather(A, [i], axis = (num_dims-1) ), prune_dims_A)
        B0 = tf.reshape(tf.gather(B, [i], axis = (num_dims-1) ), prune_dims_B)
        Y = Y + tf.matmul(A0, B0)
        i+=1
    return Y
