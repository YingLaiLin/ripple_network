
# specify data directory
base_dir = 'data/'
train_data_path = base_dir + "train.txt"
test_data_path = base_dir + "test.txt"
network_path = base_dir + "network.txt"


# filter no-relavant items by label
interest_relations = [0,1,2]

# hyperparameter here
hops = 1
epoches = 10
batch_size = 100
embedding_size = 2
learning_rate = 0.1
lambda1 = 0.5
lambda2 = 1
