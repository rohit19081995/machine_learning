import tensorflow as tf
from snake import Snake
import random
import numpy as np
import time
from collections import Counter

DISPLAY = False

resolution = [15,15]

nnodes_hl1 = 500
nnodes_hl2 = 500
nnodes_hl3 = 500

n_classes = 5
batch_size = 100

x = tf.placeholder(tf.int8, [None, resolution[0],resolution[1]])
y = tf.placeholder(tf.int8)

def neural_network_model(data):
	hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([resolution[0],resolution[1], nnodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([nnodes_hl1]))}
	
	hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([nnodes_hl1, nnodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([nnodes_hl2]))}
	
	hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([nnodes_hl2, nnodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([nnodes_hl3]))}
	
	output_layer = {'weights':tf.Variable(tf.random_normal([nnodes_hl3, n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}


	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
	
	return output

def generate_training_data(games = 10):
	scores = []
	games_to_train = []
	i=0
	while len(scores) < 10:
		game_data = []
		s = Snake(resolution = resolution, display = DISPLAY)
		prev_data = None
		start_time = time.time()
		while not s.done or (time.time() - start_time) > 10:
			data, output = get_game_data(s)
			if prev_data is not None:
				game_data.append([prev_data, output])
			prev_data = data
			keypress = [0,0,0,0,0]
			dir_index = random.randint(1,4)
			keypress[dir_index] = 1
			s.next(keypress)
			# print s.score, s.head, s.food
		# print s.score
		print i, len(scores)
		i+=1
		if s.score > 1:
			games_to_train.extend(game_data)
			scores.append(s.score)
	return game_data, scores

def get_game_data(s):
	result = np.zeros([resolution[0],resolution[1]])
	for tile in s.snake:
		result[tile[0]][tile[1]] = 1 # snake
	result[s.head[0]][s.head[1]] = 2 # head
	result[s.food[0]][s.food[1]] = 3 # food
	direction = [0,0,0,0]
	if s.direction == 'left':
		direction[0] = 1
	if s.direction == 'right':
		direction[1] = 1
	if s.direction == 'up':
		direction[2] = 1
	if s.direction == 'down':
		direction[3] = 1

	return result, direction

# def train_neural_network(x):
# 	prediction = neural_network_model(x)
# 	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 	optimizer = tf.train.AdamOptimizer().minimize(cost)

# 	hm_epochs = 10

# 	with tf.Session() as sess:
# 		sess.run(tf.initialize_all_variables())

# 		for hm in range(hm_epochs):
# 			epoch_loss = 0
# 			for _ in range(int(mnist.train.num_examples/batch_size)):
# 				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
# 				_,c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
# 				epoch_loss += c
# 			print hm, epoch_loss

# 		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# 		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
# 		print accuracy.eval({x:mnist.test.images, y:mnist.test.labels})

game_data, scores = generate_training_data(10)
print Counter(scores)
# train_neural_network(x)