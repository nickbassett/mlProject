# Import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = sns.load_dataset("iris")

# Generate the scatter plot
sns.jointplot(x="sepal_length",y="sepal_width", data=iris,height=6) # changed size to height

# Display the plots
plt.show()
