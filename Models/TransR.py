import tensorflow
import pickle
import numpy

class TransR():

	def __init__(self, config):

		self.config = config
		self.train_loss_history = []

		self.entity_embedding_vectors = tensorflow.get_variable(name = "entity_embeddings", shape = [config['entity_count'], config['embedding_size']])
		self.relation_embedding_vectors = tensorflow.get_variable(name = "relation_embeddings", shape = [config['relation_count'], config['embedding_size']])
		self.transfer_matrix = tensorflow.get_variable(name = "transfer_matrix", shape = [config['relation_count'], config['embedding_size'] * config['embedding_size']])
		
		self.positive_head_id = tensorflow.placeholder(tensorflow.int32)
		self.positive_tail_id = tensorflow.placeholder(tensorflow.int32)
		self.positive_relation_id = tensorflow.placeholder(tensorflow.int32)

		self.negative_head_id = tensorflow.placeholder(tensorflow.int32)
		self.negative_tail_id = tensorflow.placeholder(tensorflow.int32)
		self.negative_relation_id = tensorflow.placeholder(tensorflow.int32)

		self.positive_distances = tensorflow.placeholder(tensorflow.float32)

		positive_head_embedding_vector = tensorflow.reshape(tensorflow.nn.embedding_lookup(self.entity_embedding_vectors, self.positive_head_id), [-1, config['embedding_size'], 1])
		positive_tail_embedding_vector = tensorflow.reshape(tensorflow.nn.embedding_lookup(self.entity_embedding_vectors, self.positive_tail_id), [-1, config['embedding_size'], 1])
		positive_relation_embedding_vector = tensorflow.reshape(tensorflow.nn.embedding_lookup(self.relation_embedding_vectors, self.positive_relation_id), [-1, config['embedding_size']])

		negative_head_embedding_vector = tensorflow.reshape(tensorflow.nn.embedding_lookup(self.entity_embedding_vectors, self.negative_head_id), [-1, config['embedding_size'], 1])
		negative_tail_embedding_vector = tensorflow.reshape(tensorflow.nn.embedding_lookup(self.entity_embedding_vectors, self.negative_tail_id), [-1, config['embedding_size'], 1])
		negative_relation_embedding_vector = tensorflow.reshape(tensorflow.nn.embedding_lookup(self.relation_embedding_vectors, self.negative_relation_id), [-1, config['embedding_size']])

		positive_matrix = tensorflow.reshape(tensorflow.nn.embedding_lookup(self.transfer_matrix, self.positive_relation_id), [-1, config['embedding_size'], config['embedding_size']])
		negative_matrix = tensorflow.reshape(tensorflow.nn.embedding_lookup(self.transfer_matrix, self.negative_relation_id), [-1, config['embedding_size'], config['embedding_size']])

		positive_head_embedding_vector = tensorflow.nn.l2_normalize(tensorflow.reshape(tensorflow.matmul(positive_matrix, positive_head_embedding_vector), [-1, config['embedding_size']]), 1)
		positive_tail_embedding_vector = tensorflow.nn.l2_normalize(tensorflow.reshape(tensorflow.matmul(positive_matrix, positive_tail_embedding_vector), [-1, config['embedding_size']]), 1)

		negative_head_embedding_vector = tensorflow.nn.l2_normalize(tensorflow.reshape(tensorflow.matmul(negative_matrix, negative_head_embedding_vector), [-1, config['embedding_size']]), 1)
		negative_tail_embedding_vector = tensorflow.nn.l2_normalize(tensorflow.reshape(tensorflow.matmul(negative_matrix, negative_tail_embedding_vector), [-1, config['embedding_size']]), 1)

		positive = tensorflow.reduce_sum((positive_head_embedding_vector + positive_relation_embedding_vector - positive_tail_embedding_vector) ** 2, 1, keep_dims = True)
		negative = tensorflow.reduce_sum((negative_head_embedding_vector + negative_relation_embedding_vector - negative_tail_embedding_vector) ** 2, 1, keep_dims = True)

		self.predict = positive

		if config['discount_factor'] == None:
			self.discount_factor = tensorflow.get_variable(name = "discount_factor", shape = [1,])
			cluster = tensorflow.reduce_sum(tensorflow.maximum(tensorflow.reduce_mean(tensorflow.nn.l2_normalize(positive_relation_embedding_vector, 1)) - (1 - tensorflow.pow(self.discount_factor, self.positive_distances)), 0))
		else:		
			cluster = tensorflow.reduce_sum(tensorflow.maximum(tensorflow.reduce_mean(tensorflow.nn.l2_normalize(positive_relation_embedding_vector, 1)) - (1 - tensorflow.pow(config['discount_factor'], self.positive_distances)), 0))
			
		if config['original']:
			self.loss = tensorflow.reduce_sum(tensorflow.maximum(positive - negative + config['margin'], 0)) 
		else:
			self.loss = tensorflow.reduce_sum(tensorflow.maximum(positive - negative + cluster + config['margin'], 0)) 

		self.global_step = tensorflow.Variable(0, name = "global_step", trainable = False)
		self.optimizer = tensorflow.train.AdamOptimizer(learning_rate = config['learning_rate'])
		self.gradients = self.optimizer.compute_gradients(self.loss)
		self.learn = self.optimizer.apply_gradients(self.gradients, global_step = self.global_step)

	def save_model(self, file_name):

		dct = 	{
					'config': self.config,
					'global_step': self.global_step.eval(),
					'entity_embeddings': self.entity_embedding_vectors.eval(),
					'relation_embeddings': self.relation_embedding_vectors.eval(),
					'transfer_matrix': self.transfer_matrix.eval(),
					'train_loss_history': numpy.array(self.train_loss_history)
				}

		if self.config['discount_factor'] == None:
			dct['discount_factor'] = self.discount_factor.eval()

		fileObject = open(file_name, "wb")
		pickle.dump(dct, fileObject)
		fileObject.close()

	def load_model(self, file_name):

		fileObject = open(file_name, "rb")
		dct = pickle.load(fileObject)
		fileObject.close()

		self.config = dct['config']
		self.train_loss_history = dct['train_loss_history'].tolist()
		tensorflow.assign(self.global_step, dct['global_step']).eval()
		tensorflow.assign(self.entity_embedding_vectors, dct['entity_embeddings']).eval()
		tensorflow.assign(self.relation_embedding_vectors, dct['relation_embeddings']).eval()
		tensorflow.assign(self.transfer_matrix, dct['transfer_matrix']).eval()

		if self.config['discount_factor'] == None:
			tensorflow.assign(self.discount_factor, dct['discount_factor']).eval()

	def train_model(self, session, positive_heads, positive_tails, positive_relations, positive_distances, negative_heads, negative_tails, negative_relations):

		feed_dict = {
						self.positive_head_id: positive_heads,
						self.positive_tail_id: positive_tails,
						self.positive_relation_id: positive_relations,
						self.positive_distances: positive_distances,
						self.negative_head_id: negative_heads,
						self.negative_tail_id: negative_tails,
						self.negative_relation_id: negative_relations
					}

		_, step, loss = session.run([self.learn, self.global_step, self.loss], feed_dict = feed_dict)

		self.train_loss_history.append(loss)

		return loss

	def test_model(self, session, positive_heads, positive_tails, positive_relations):

		feed_dict = {
						self.positive_head_id: positive_heads,
						self.positive_tail_id: positive_tails,
						self.positive_relation_id: positive_relations
					}

		loss = session.run(self.predict, feed_dict = feed_dict)
		loss = numpy.reshape(loss, (-1,))

		return loss
