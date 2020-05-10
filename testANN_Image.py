# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from ANN import ANN
import PIL
from PIL import Image

# Setup the ANN configuration
inode = 784
hnode = 100
onode = 10

# Set the learning rate
lr = 0.1 # default 0.1

# Instantiate an ANN object named ann
ann = ANN(inode, hnode, onode, lr)

# Create the training list data
dataFile = open('datasets/mnist_train_100.csv') #option  _100 dataset
dataList = dataFile.readlines()
dataFile.close()

# Train the ANN using all the records in the list
for record in dataList:
      recordx = record.split(',')
      inputT = (np.asfarray(recordx[1:])/255.0*0.99) + 0.01
      train = np.zeros(onode) + 0.01
      train[int(recordx[0])] = 0.99

      # Training begins here
      ann.trainNet(inputT, train)

# Create the test list data from an image
img = Image.open('images/zero.jpg')
img = img.resize((28,28), PIL.Image.ANTIALIAS)
#img = img.resize((28,28), PIL.Image.NEAREST)

# Image it as a grayscale
plt.imshow(img, cmap="Greys", interpolation="None")
plt.show()

# Read pixels into list
pixels = list(img.getdata())
#print ('pixels into list')
#print (pixels) #added

# Convert into single values from tuples
pixels = [i[0] for i in pixels]
#print ('Convert into single values from tuples')
#print (pixels) #added

# Save to a temp file named test.csv with comma delimiters
imgTmp = np.array(pixels)
imgTmp.tofile('test.csv', sep=',')
print ('imgTmp Array')
print (imgTmp) #added

# Open the temp file and read into list
testDataFile = open('test.csv')
testDataList = testDataFile.readlines()
testDataFile.close()
#print ('Open the temp file and read into list')
#print (testDataList)

# Iterate through all list elements
for record in testDataList:
      recordx = record.split(',')
      # Adjust record values for ANN
      input = (np.asfarray(recordx[0:])/255.0*0.99)+0.01
      output = ann.testNet(input)

# Display output data vector
#print (input) #added
print (output)
