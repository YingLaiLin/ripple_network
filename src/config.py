
# specify data directory
base_dir = 'data/'
train_data_path = base_dir + "train.txt"
test_data_path = base_dir + "test.txt"
network_path = base_dir + "network.txt"


# network and relations relative information
num_nodes = 8
interest_relations = [0,1,2]

# hyperparameter here
hops = 3
epoches = 10 # 1~10
embedding_size = 2
learning_rate = 0.1
momentum = 0.9
lambda1 = 0.01
lambda2 = 0.01
# parameter for batch normalization
mean = 0
var = 1
epsilon = 0.001