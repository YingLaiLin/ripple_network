

# network and relations relative information
num_nodes = 4553
interest_relations = [0,1,2]

# hyperparameter here
batch_size = 4
hops = 1
epoches = 5 # 1~10
embedding_size = 20
learning_rate = 0.9
momentum = 0.9
lambda1 = 0.3
lambda2 = 0.3

# parameter for batch normalization
mean = 0
var = 1
epsilon = 0.001

# specify data directory
base_dir = 'data/'
train_data_path = base_dir + "train.txt"
test_data_path = base_dir + "test.txt"
network_path = base_dir + "network.txt"

# specify model directory here
model_dir = 'params-%d/' % num_nodes
model_path = model_dir + 'ripple.ckpt'

# specify log directory here
log_dir = 'logs/'
timeline_path = log_dir + 'ripple_tl-%d.json' % num_nodes
log_path = log_dir + 'ripple-%d-%.3f.log' % (num_nodes, learning_rate)

# specify performance file here
score_dir = 'score/'
score_path = score_dir + 'auc.txt'