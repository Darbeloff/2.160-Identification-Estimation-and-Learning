import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def train_model(model1, model2, x, y):
	if len(np.shape(x))==1:
		x = x[:, None]
	if len(np.shape(y))==1:
		y = y[:, None]
	D_in = np.shape(x)[1]
	D_out = np.shape(y)[1]
	N = 50

	dataset = TensorDataset(x, y)

	N_train = int(3*len(y)/5)
	train_dataset, val_dataset = random_split(dataset, [N_train,len(y)-N_train])

	train_loader = DataLoader(dataset=train_dataset, batch_size=N)
	val_loader = DataLoader(dataset=val_dataset, batch_size=N)

	loss_fn = torch.nn.MSELoss(reduction='sum')

	model = torch.nn.Sequential(model1, model2)

	# Use the optim package to define an Optimizer that will update the weights of
	# the model for us. Here we will use Adam; the optim package contains many other
	# optimization algorithms. The first argument to the Adam constructor tells the
	# optimizer which Tensors it should update.
	learning_rate = .0001
	n_epochs = 10000
	training_losses = []
	validation_losses = []
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	for t in range(n_epochs):
		batch_losses = []

		with torch.no_grad():
			val_losses = []
			for x_val, y_val in val_loader:
				x_val = x_val.to(device)
				y_val = y_val.to(device)
				yhat = model(x_val)
				val_loss = loss_fn(y_val, yhat).item()
				val_losses.append(val_loss)
			validation_loss = np.mean(val_losses)
			validation_losses.append(validation_loss)

		for x_batch, y_batch in train_loader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			# Forward pass: compute predicted y by passing x to the model.
			y_pred = model(x_batch)

			# Compute and print loss.
			loss = loss_fn(y_pred, y_batch)

			optimizer.zero_grad()

			# Backward pass: compute gradient of the loss with respect to model
			# parameters
			loss.backward()

			# Calling the step function on an Optimizer makes an update to its
			# parameters
			optimizer.step()

			batch_losses.append(loss.item())
		training_loss = np.mean(batch_losses)
		training_losses.append(training_loss)

		print(f"[{t+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

		if t>100 and validation_losses[-2]<=validation_losses[-1]:
			break
	
	plt.figure()
	plt.semilogy(range(len(training_losses)), training_losses, label='Training Loss')
	plt.semilogy(range(len(training_losses)), validation_losses, label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

	model.eval()
	return model

def robot_kinematic_model(q1, q2): # Eq. 1
	x1 = torch.cos(q1)+torch.cos(q2)
	x2 = torch.sin(q1)+torch.sin(q2)
	return x1, x2

if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	dtype = torch.FloatTensor
	torch.manual_seed(3)

	H, D_in, D_out = 4, 2, 2
	NN1 = torch.nn.Sequential( # Eq. 2
		torch.nn.Linear(D_in, H),
		torch.nn.ReLU(),
		torch.nn.Linear(H, D_out),
	)
	NN2 = copy.deepcopy(NN1)
	NN2.load_state_dict(torch.load('NN2.pt'))

	q = torch.rand(10000, D_in)*2*3.14
	x1, x2 = robot_kinematic_model(q[:,0], q[:,1])
	x = torch.cat((x1[:,None], x2[:,None]), 1)

	NN_comb = train_model(NN1, NN2, x, x)
	# torch.save(NN_comb.state_dict(), 'NN_comb.pt')