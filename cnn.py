import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from sklearn.metrics import confusion_matrix 

train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[1000:2000]
test_labels = mnist.test_labels()[1000:2000]

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
  
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  
  loss = -np.log(out[label])
  pred_label = np.argmax(out)
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc, pred_label

def train(im, label, lr=.005):
  
  out, loss, acc, _ = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc

print('MNIST CNN initialized!')

for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]


  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
      #print(
       # 'Step %d : Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        #(i + 1, loss / 100, num_correct)
      #)
      print(f"step = {i + 1}, loss = {round((loss/100),2)}, acc = {num_correct}%")
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc

#Test
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
actual_mat, pred_mat = [], []
for im, label in zip(test_images, test_labels):
  _, l, acc, pred_label = forward(im, label)
  actual_mat.append(label)
  pred_mat.append(pred_label)
  loss += l
  num_correct += acc
cf_mat = confusion_matrix(actual_mat, pred_mat)
num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
print(cf_mat)
