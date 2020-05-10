from keras.datasets import fashion_mnist

(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

print('Number of training records and size of each training record: ', train_images.shape)
print()
print('Number of training labels: ', len(train_labels))
print()
print('Training label: ', train_labels)
print()
print('Number of test records and size of each test record:', test_images.shape)
print()
print('Number of test labels: ', len(test_labels))
print()
