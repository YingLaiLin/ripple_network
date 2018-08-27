

import tensorflow as tf
import pandas as pd
import utils
import config
import networkx as nx
import numpy as np

# 申请内存过多。  how to fix ？

network = utils.read_network(config.network_path, config.interest_relations)

num_nodes = config.num_nodes
num_relations = len(config.interest_relations) 
embedding_size = config.embedding_size
network_weights = nx.to_numpy_matrix(network, nodelist=np.arange(1,num_nodes+1), weight='label')
regularizer = tf.contrib.layers.l2_regularizer(config.lambda1)
interest_relations = range(len(config.interest_relations))


def get_next_hop_matrix(adjacent_matrix, weight_matrix):
    next_hop_adjacent_matrix = tf.matmul(adjacent_matrix, adjacent_matrix)
    return tf.multiply(next_hop_adjacent_matrix, weight_matrix)

with tf.name_scope('input_layer'):
    user = tf.placeholder(dtype=tf.int32, shape=(num_nodes, num_nodes), name='user')
    item = tf.placeholder(dtype=tf.int32, shape=(1),name='item')
    label = tf.placeholder(dtype=tf.int32, shape=[1, 1], name='label')
    weight_matrix = tf.constant(dtype=tf.int32, value=network_weights)

with tf.name_scope('parameters'):
    entity_embeddings = tf.get_variable("entity_embeddings", shape=[num_nodes, embedding_size])
    relation_embeddings = tf.get_variable("relation_embeddings", shape=[num_relations, embedding_size, embedding_size])
    item_embeddings = tf.get_variable("item_embeddings", shape=[num_nodes, embedding_size])



# contruct ripple layers


adj = user
u = None
for layer_num in range(1, config.hops + 1):

    with tf.name_scope('ripple_layer_%d' % layer_num):
        #TODO 这里邻接矩阵都可以先算。
        next_hop_matrix = get_next_hop_matrix(adj, weight_matrix) 
        adj = next_hop_matrix
        next_hop_matrix = tf.cast(next_hop_matrix, tf.bool)
        non_zero_items = tf.where(next_hop_matrix)
        # 获取 head, tail, relation 对应的值, 用于提取对应的 embedding 结果
        # heads = tf.gather(non_zero_items, 0, axis=1) #TODO fix it
        # tails = tf.gather(non_zero_items, 1, axis=1)
        heads = non_zero_items[:,0]
        tails = non_zero_items[:,1]
        non_zero_items_indices = []
        
        
        # 找到对应的关系
        relations = tf.map_fn(lambda x: weight_matrix[x[0],x[1]], tf.stack([heads, tails], axis=1),dtype=tf.int32)
        r = tf.nn.embedding_lookup(relation_embeddings, relations)
        head_embeddings = tf.nn.embedding_lookup(entity_embeddings, heads)
        head_embeddings = tf.reshape(head_embeddings, shape=[-1, 1,embedding_size])
        v = tf.nn.embedding_lookup(item_embeddings, item)
        v_tile = tf.tile(tf.expand_dims(v,0), [tf.size(heads),1,1])   # 复制 v，方便计算
        p = tf.nn.softmax(tf.matmul(tf.matmul(v_tile, r), tf.transpose(head_embeddings,[0,2,1])))
        p = tf.reshape(p, shape=[-1,1])
        tail_embeddings = tf.nn.embedding_lookup(entity_embeddings, tails)
        # o = tf.reduce_mean(tf.matmul(p, tail_embeddings),axis=1, keep_dims=True)
        o = tf.matmul(tf.transpose(p), tail_embeddings)
        a = tf.random_uniform([],0,1, name='alpha')  #TODO a 应当改成变量
        
        # fc_mean, fc_var = tf.nn.moments(o, axes=[0])
        # scale = tf.Variable(tf.ones(o.get_shape().as_list()))
        # shift = tf.Variable(tf.zeros(o.get_shape().as_list()))
        # ema = tf.train.ExponentialMovingAverage(decay=0.5)
        # def mean_var_with_update():
        #     ema_apply_op = ema.apply([fc_mean, fc_var])
        #     with tf.control_dependencies([ema_apply_op]):
        #         return tf.identity(fc_mean), tf.identity(fc_var)
        # mean, var = mean_var_with_update()        
        # o = tf.nn.batch_normalization(o, 0, 1, shift, scale, 0.001)
        # o = tf.layers.batch_normalization(o, True)  #TODO 是否可以加速训练？
        if u is None:
            u = a * o
        else:
            u += a * o

with tf.name_scope('output_layer'):
    y = tf.sigmoid(tf.matmul(v, tf.transpose(u)))

with tf.name_scope('loss_layer'):
    with tf.name_scope('cal_entropy_loss'):
        label = tf.cast(tf.reshape(label, shape=[-1,1]), tf.float32)
        L1 = - tf.reduce_sum(label*tf.log(y) + (1-label)*tf.log(y))
    with tf.name_scope('cal_sim_loss'):
        L2_norm = 0
        
        #TODO 如何固定 relation_embeddings, 减少空间占用?  embedding_look_up
        
        for r in interest_relations:
            ERE = tf.matmul(
                    tf.matmul(entity_embeddings,
                                tf.nn.embedding_lookup(relation_embeddings, r)),\
                                        tf.transpose(entity_embeddings), name='cal_ERE')
            
            I_r = (weight_matrix == r)
            I_r = tf.cast(I_r, tf.float32)
            L2_norm += tf.reduce_sum(tf.square(tf.norm(ERE - I_r, ord='euclidean')))
        L2_norm *= config.lambda2 / 2 

    with tf.name_scope('cal_regularization_loss'):
        L3 = config.lambda1/2 * regularizer(entity_embeddings) + regularizer(
            item_embeddings) + regularizer(relation_embeddings)

    Loss = L1 + L2_norm + L3
    tf.summary.scalar('loss', Loss)

with tf.name_scope('train_op'):
    global_step = tf.Variable(0, trainable=False)
    decay_lr = tf.train.exponential_decay(config.learning_rate, global_step, 100, 0.9)
    # train_op = tf.train.AdamOptimizer(decay_lr).minimize(Loss)
    train_op = tf.train.MomentumOptimizer(decay_lr, config.momentum).minimize(Loss)
    saver = tf.train.Saver(tf.global_variables())


with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/lr-%d-%.3f" %(num_nodes,config.learning_rate), sess.graph)
    run_metadata = tf.RunMetadata()
    users,items,labels = utils.load_data(config.train_data_path) 
    tf.global_variables_initializer().run()
    total_step = 0
    for epoch in range(config.epoches):
        for j in range(len(users)):
            total_step += 1
            cur_cur = users[j]
            neighbors = network.neighbors(cur_cur)
            
            adjacent_matrix = np.zeros((num_nodes, num_nodes))
            for neighbor in neighbors:
                adjacent_matrix[neighbor,cur_cur] = network[cur_cur][neighbor]['label']
                adjacent_matrix[cur_cur,neighbor] = network[cur_cur][neighbor]['label']
            
            la = np.reshape(np.array([labels[j]]),(-1,1))
            loss,res = sess.run([Loss, merged], 
                                    feed_dict={
                                                user:adjacent_matrix, 
                                                item:[items[j]],
                                                label:la},
                                                options=tf.RunOptions(
                                                            trace_level=tf.RunOptions.FULL_TRACE),
                                                            run_metadata=run_metadata)
            writer.add_summary(res, total_step)
            if total_step % 20 == 0:
                loss,_ = sess.run([Loss, merged], feed_dict={
                                                        user:adjacent_matrix, 
                                                        item:[items[j]],
                                                        label:la})
                writer.add_run_metadata(run_metadata, 'step %03d' % total_step)
                
                print('epoch:{}, train_loss:{}'.format(total_step, loss))
    saver.save(sess,'checkpoint/ripple.ckpt')
    opts = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY.copy()
    opts['show_name_regexes'] = ['.*my_code.py.*']
    opts['output'] = 'timeline:outfile=timeline'
    # tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #     tf.get_default_graph(),
    #     run_meta=run_metadata,
    #     tfprof_cmd='code',
    #     tfprof_options=opts)
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        run_meta=run_metadata,
        tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
    
    
 