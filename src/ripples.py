
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import utils
import config
import networkx as nx
import numpy as np
from sklearn import metrics

from tensorflow.python.client import timeline


network,label_mapping, node_mapping = utils.read_network(config.network_path, config.interest_relations)

num_nodes = len(network.node) 
num_nodes = config.num_nodes
num_relations = len(config.interest_relations) 
embedding_size = config.embedding_size
network_weights = nx.to_numpy_matrix(network, nodelist=np.arange(1,num_nodes+1), weight='label')
regularizer = tf.contrib.layers.l2_regularizer(config.lambda1)
interest_relations = range(len(config.interest_relations))

# define device here
cpu = '/cpu:0'
gpu = '/gpu:0'

def get_next_hop_matrix(adjacent_matrix, weight_matrix):
    next_hop_adjacent_matrix = tf.matmul(adjacent_matrix, adjacent_matrix)
    return tf.multiply(next_hop_adjacent_matrix, weight_matrix)

with tf.name_scope('input_layer'):
    with tf.device(cpu):
        user = tf.placeholder(dtype=tf.int32, shape=(num_nodes, num_nodes), name='user')
        item = tf.placeholder(dtype=tf.int32, shape=(1),name='item')
        label = tf.placeholder(dtype=tf.int32, shape=[1, 1], name='label')

    with tf.device(gpu):
        weight_matrix = tf.constant(dtype=tf.int32, value=network_weights)

with tf.name_scope('parameters'):
    with tf.device(cpu):
        entity_embeddings = tf.get_variable("entity_embeddings", shape=[num_nodes, embedding_size], initializer=tf.contrib.layers.xavier_initializer())
        relation_embeddings = tf.get_variable("relation_embeddings", shape=[num_relations, embedding_size, embedding_size],initializer=tf.contrib.layers.xavier_initializer())
        item_embeddings = tf.get_variable("item_embeddings", shape=[num_nodes, embedding_size],initializer=tf.contrib.layers.xavier_initializer())
        a = tf.get_variable("alpha", shape=[config.hops]) 
        a = a / tf.reduce_sum(a)
        

print(tf.trainable_variables())
# contruct ripple layers


adj = user
u = tf.zeros([1,config.embedding_size])

for layer_num in range(1, config.hops + 1):
    print('construction of ripple layer %d' % layer_num)
    with tf.device(cpu):
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
            relations = tf.map_fn(lambda x: weight_matrix[x[0],x[1]], tf.stack([heads, tails], axis=1),dtype=tf.int32, name='obtain_relation_set')
            r = tf.nn.embedding_lookup(relation_embeddings, relations)
            head_embeddings = tf.nn.embedding_lookup(entity_embeddings, heads)
            head_embeddings = tf.reshape(head_embeddings, shape=[-1, 1,embedding_size])
            v = tf.nn.embedding_lookup(item_embeddings, item)
            v_tile = tf.tile(tf.expand_dims(v,0), [tf.size(heads),1,1], name='v_stack')   # 复制 v，方便计算
            softmax_item = tf.matmul(tf.matmul(v_tile, r), tf.transpose(head_embeddings,[0,2,1]), name='cal_softmax_matrix')
            p = tf.nn.softmax(softmax_item, name='softmax')
            p = tf.reshape(p, shape=[-1,1])
            tail_embeddings = tf.nn.embedding_lookup(entity_embeddings, tails)
            # o = tf.reduce_mean(tf.matmul(p, tail_embeddings),axis=1, keep_dims=True)
            o = tf.matmul(tf.transpose(p), tail_embeddings, name='output_of_layer_%d' % layer_num)
            
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
            u = tf.concat([u, o], axis=0)        

u = tf.reduce_mean(tf.multiply(u, a),axis=0)
u = tf.reshape(u,[1,-1])
with tf.name_scope('output_layer'):
    y = tf.sigmoid(tf.matmul(v, tf.transpose(u)))

with tf.device(gpu):
    with tf.name_scope('loss_layer'):
        with tf.name_scope('cal_entropy_loss'):
            label = tf.cast(tf.reshape(label, shape=[-1,1]), tf.float32)
            entropy_loss = label*tf.log(y) + (1-label)*tf.log(y)
            L1 = - tf.reduce_sum(entropy_loss)
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
                L2 = ERE - I_r
                L2_norm += tf.reduce_sum(tf.square(tf.norm(L2, ord='euclidean')))
            L2_norm *= config.lambda2 / 2 

        with tf.name_scope('cal_regularization_loss'):
            L3 = config.lambda1/2 * regularizer(entity_embeddings) + regularizer(
                item_embeddings) + regularizer(relation_embeddings)

        Loss = L1 + L2_norm + L3
        tf.summary.scalar('loss', Loss)
with tf.device(gpu):
    with tf.name_scope('train_op'):
        global_step = tf.Variable(0, trainable=False)
        decay_lr = tf.train.exponential_decay(config.learning_rate, global_step, 100, 0.9)
        # train_op = tf.train.AdamOptimizer(decay_lr).minimize(Loss)
        train_op = tf.train.MomentumOptimizer(decay_lr, config.momentum).minimize(Loss)
    

# define config for session heer
sess_config = tf.ConfigProto()
# sess_config.log_device_placement = True
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True

with tf.Session(config=sess_config) as sess:
    # sess.graph.finalize() # 判断计算图中是否不断增加节点
    saver = tf.train.Saver(tf.global_variables())
    if tf.gfile.Exists(config.model_dir + 'checkpoint'):
        print('restoring models')
        saver.restore(sess,config.model_path)
    else:
        print('init models')
        tf.global_variables_initializer().run()

    # configure parameter for sess.run
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config.log_path, sess.graph)
    
    with tf.device(cpu):
        #load data
        users,items,labels = utils.load_data(config.train_data_path, config.interest_relations) 
        users = users.apply(lambda x: node_mapping[x] if x in node_mapping else x)    
        items = items.apply(lambda x: node_mapping[x] if x in node_mapping else x)
        labels = list(labels.apply(lambda x: label_mapping[x] if x in label_mapping else x))

    total_step = 0
    batch_size = config.batch_size
    if batch_size > len(users):
        batch_size = len(users)
    evaluations = []
    for epoch in range(config.epoches):
        predictions = []
        for j in range(len(users)):
            total_step += 1
            cur_cur = users[j]
            neighbors = network.neighbors(cur_cur)
            
            adjacent_matrix = np.zeros((num_nodes, num_nodes))
            for neighbor in neighbors:
                adjacent_matrix[neighbor,cur_cur] = network[cur_cur][neighbor]['label']
                adjacent_matrix[cur_cur,neighbor] = network[cur_cur][neighbor]['label']
            
            la = np.reshape(np.array([labels[j]]),(-1,1))
            loss,res, pred = sess.run([Loss, merged, y], 
                                    feed_dict={
                                                user:adjacent_matrix, 
                                                item:[items[j]],
                                                label:la},
                                                options=run_options,
                                                run_metadata=run_metadata)
            predictions.extend(np.ravel(pred).tolist())
            # print('epoch:{}, train_loss:{}'.format(total_step, loss))
            if total_step % batch_size == 0:
                loss,_ = sess.run([Loss, merged], feed_dict={
                                                        user:adjacent_matrix, 
                                                        item:[items[j]],
                                                        label:la})
                # add summary to tf
                writer.add_summary(res, total_step)
                writer.add_run_metadata(run_metadata, 'step %03d' % total_step)  
                print('epoch:{}, train_loss:{}'.format(total_step, loss))
        
        # perform evaluation
        fpr, tpr, threshold = metrics.roc_curve(np.ravel(np.array(labels)), predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        evaluations.append(auc)
        print(predictions)
        print(labels)
        print('AUC %.3f' % auc)
        # record allocation for resources with timeline
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(config.timeline_path, "w") as f:
            f.write(chrome_trace)

    print('train complete')

    pd.DataFrame(evaluations).to_csv(config.score_path)
    #save model
    # saver.save(sess,config.model_path)
    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        run_meta=run_metadata,
        tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
    
    
 
