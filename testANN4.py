# Import required libraries
from ANN import ANN
inputT = [0.8, 0.5, 0.6]
print('Input data vector')
print(inputT)
print()

train = inputT
ann = ANN(3,3,3,0.3)
output = ann.testNet(inputT)
print('After one iteration')
print(output)
print()

matrixList = ann.getMatrices()
print('wtgih matrix')
print(matrixList[0])
print()
print('wtgho matrix')
print(matrixList[1])
print()

for i in range(499):
    ann.trainNet(inputT, train)

output = ann.testNet(inputT)
print('After 500 iterations')
print(output)
print()

matrixList = ann.getMatrices()
print('wtgih matrix')
print(matrixList[0])
print()
print('wtgho matrix')
print(matrixList[1])
print()
