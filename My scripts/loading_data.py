# exercise 1.5.2
import numpy as np
import xlrd
from sklearn.preprocessing import LabelEncoder

from pathlib import Path

# Get the absolute path to the script's directory
script_dir = Path(__file__).resolve().parent  # Directory of the script
print(script_dir)

# Build the correct absolute path to the data file
filename = script_dir / "../data/south_african_heart_disease.xls"

# Convert to string for xlrd
filename = str(filename.resolve())

# Load the data
doc = xlrd.open_workbook(filename).sheet_by_index(0)


# # Get path to the datafile
# filename = "Assignment1/data/south_african_heart_disease.xls"

print("\nLocation of the south_african_heart_disease.xls file: {}".format(filename))

#Note : We've loaded the csv file in excel and shifted the famhist column to where we want it to make things easier

# Load xls sheet with data
# There's only a single sheet in the .xls, so we take out that sheet
doc = xlrd.open_workbook(filename).sheet_by_index(0)

# Extract attribute names
# Extract all columns (except for the first one)
attributeNames = doc.row_values(rowx=0, start_colx=1, end_colx=10)

# Extract class names to python list, then encode with integers (dict):
classNames = set(['chd no', 'chd yes'])
classDict = dict(zip(classNames, range(len(classNames))))

#Exctract y
y = np.int32(np.array(doc.col_values(10, 1, 463)))  # check out help(doc.col_values)


# Preallocate memory
X = np.empty((462, 9))

#extract data to matrix X (except column 5) - for simplicity we store it at the end of the matrix
for j in range(8):
    X[:, j] = np.array(doc.col_values(j+1, 1, 463)).T
    j += 1

#We convert famhist to 1 and 0
famhist = doc.col_values(9,1,463)
mask = [value == "Present" for value in famhist]

#insert it in our data matrix:
X[:,8] = mask

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)
