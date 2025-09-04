import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, norm, probplot
import xlrd

df = pd.read_csv("south_african_heart_disease.csv", encoding='ISO-8859-1')

# Drop irrelevant variables
df.drop('row.names', axis=1, inplace=True)

# Simple summary of info
df.info()

# Transform famhist to binary values
df['famhist'] = np.where(df['famhist'] == 'Present', 1, 0).astype(int)
df.describe()

# Round to 2 decimal places
df_rounded = df.describe().round(2)

# Apply styling with black text and formatting to 2 decimal places
styled_df = df_rounded.style.format("{:.2f}").set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('color', 'black')]},  # Set header text to black
    {'selector': 'td', 'props': [('background-color', 'white'), ('color', 'black')]},      # Set cell text to black
    {'selector': 'table', 'props': [('border', '1px solid #ddd'), ('border-collapse', 'collapse')]}
])
styled_df

df["typea"].hist(bins=30, edgecolor='black')
plt.axvline(x=55, color='red', linestyle='dashed', linewidth=2, label="Type A Threshold")  # Add vertical line
plt.xlabel("Type A behaviour")
plt.ylabel("Frequency")
plt.title("Distribution of Type A behaviour")
plt.text(40, plt.ylim()[1] * 0.9, "Not Type A", fontsize=12, color="black", ha="center")
plt.text(70, plt.ylim()[1] * 0.9, "Type A", fontsize=12, color="black", ha="center")
plt.legend()
plt.show()

# Feature Scale Comparison
plt.figure(figsize=(6, 4))  # width, height in inches
df.boxplot(column=["sbp", "tobacco", "ldl", "adiposity", "obesity", "alcohol"])
plt.xticks(rotation=45)
plt.title("Feature Scale Comparison")
plt.show()

# Plotting histograms for each feature
df.hist(bins=30, figsize=(10, 10), edgecolor='black')
plt.show()

from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Load the data
filename = "south_african_heart_disease.xls"
doc = xlrd.open_workbook(filename).sheet_by_index(0)

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
    if j == 4:
        #We convert famhist to 1 and 0
        famhist = doc.col_values(j+1,1,463)
        mask = [value == "Present" for value in famhist]
        X[:,j] = mask
    else:
        X[:, j] = np.array(doc.col_values(j+1, 1, 463)).T
    j += 1

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

normalised_X = np.copy(X)
#transform
normalised_X[:,6] = np.log(1 + X[:,6]) #add 1 because some alcohol values are 0
#normalise
normalised_X = zscore(normalised_X, axis = 0, ddof = 1)

attributeNames_norm = np.copy(attributeNames)
attributeNames_norm[6] = 'log-alc'
attributeNames_norm = ['normalized ' + attribute for attribute in attributeNames_norm]

#Or without the last binary data

Y = np.copy(normalised_X[:,:-1])
N_y, M_y = Y.shape

attributeNames_y = np.copy(attributeNames_norm[:-1])

#First generate N points between 0 and 1 equally distanced

ppoints = np.linspace(0.01,.99, num = N)

#Then we get the quantiles

normal_quantiles = norm.ppf(ppoints) #quantiles

normal_counts, normal_bin_edges = np.histogram(normal_quantiles, bins = 30, density = True)

normal_bin_midpoints = (normal_bin_edges[:-1] + normal_bin_edges[1:]) / 2

f, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (10,6))
axes = axes.ravel()
for idx, attribute in enumerate(attributeNames_y):
    ax = axes[idx]
    # axes[idx].hist(Y[:,idx], bins = 30)
    #or using seaborn
    sns.histplot(Y[:,idx], bins = 30, ax = ax, kde = True, color = 'purple', stat = 'density', alpha = 0.8)
    ax.lines[-1].set_label('KDE function')
    ax.plot(normal_bin_midpoints, normal_counts, label = 'Normal data', marker = '.', color = 'red')
    ax.grid(True)
    if idx != 3 and idx != 4 :
        ax.legend(loc="upper right")
    else:
        ax.legend(loc='upper left')
    
    ax.set_title(f'Histogram of {attribute}')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

f.delaxes(axes[8])

plt.tight_layout()  # Prevent overlapping
plt.show()

## Q-Q plot : 

f, axes = plt.subplots(nrows = 3, ncols = 3,figsize = (15,10))
axes = axes.ravel()
for i, attribute in enumerate(attributeNames_y):
    ax = axes[i]
    probplot(Y[:,i], plot = ax)
    ax.get_lines()[0].set_color('purple')
    ax.set_title('Q-Q plot ' + attribute)
    ax.tick_params(axis='both', labelsize=8)
f.delaxes(axes[8])
plt.tight_layout()
plt.show()

from dtuimldmtools import similarity

np.random.seed(123) # we seed the random number generator to get the same random sample every time
sample_size = 75
subsample_mask = np.random.choice(N, sample_size, replace=False)
Y_sample = Y[subsample_mask, :]
y_sample = y[subsample_mask]

sorted_indices = np.argsort(y_sample) # sort rows in X acording to whether they are red of white
Y_sorted = Y_sample[sorted_indices]
y_sorted = y_sample[sorted_indices]

plt.figure(figsize = (10,10))
sns.heatmap(Y_sorted, cmap = 'Purples', linewidths=0.5, cbar=True)
plt.xticks(ticks=np.arange(len(attributeNames_y)), labels=attributeNames_y, rotation="vertical")
plt.yticks(ticks=np.arange(sample_size), labels=y_sorted, rotation="horizontal")
plt.show()

#in our sample we see more 'no' reponse to chd rather than 'yes', by curiosity we look at the count of 'yes' in the total data.

count = np.sum(y)
print(count)

#So we see that it accounts for approximatily 1/3 of the data, and therefore realize that our sample is pretty representative of our data.

sample_norm_X = normalised_X[subsample_mask, :]
# the y is already taken care of above so we only take care of the X

sorted_norm_X = sample_norm_X[sorted_indices]

plt.figure(figsize = (10,10))
sns.heatmap(sorted_norm_X, cmap = 'Purples', linewidths=0.5, cbar=True)
plt.xticks(ticks=np.arange(len(attributeNames_norm)), labels=attributeNames_norm, rotation="vertical")
plt.yticks(ticks=np.arange(sample_size), labels=y_sorted, rotation="horizontal")
plt.show()

measure = 'correlation'

AttributeCorr = np.zeros((9,9))
ClassCorr = []

for idx1 in range(len(attributeNames_norm)) :
    ClassCorr.append(similarity(normalised_X[:,idx1], y, method = measure)[0,0])
    for idx2 in range(len(attributeNames_norm)):
        AttributeCorr[idx1, idx2] = similarity(normalised_X[:,idx1], normalised_X[:,idx2], method = measure)[0,0]

print(f'Correlation for classification : {dict(zip(attributeNames_norm, np.round(ClassCorr,3)))}' )
# Correlation categories
def categorize_correlation(value):
    abs_val = abs(value)  # Work with absolute values
    if abs_val <= 0.1:
        return "---"  # No correlation
    elif abs_val <= 0.3:
        return "--"  # Weak correlation
    elif abs_val <= 0.6:
        return "-"  # Moderate correlation
    elif abs_val <= 0.85:
        return "+"  # Strong correlation
    elif abs_val <= 0.95:
        return "++"  # Very strong correlation
    else:
        return "+++"  # Near-perfect correlatioN

# Convert correlation matrix to labeled matrix
labeled_matrix = np.vectorize(categorize_correlation)(np.round(AttributeCorr,3))

# Create a DataFrame with proper labels
columns = ["Sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "log-alc", "age", "famhist"]
df = pd.DataFrame(AttributeCorr, index = columns, columns = columns)
df_labeled = pd.DataFrame(labeled_matrix, index=columns, columns=columns)

# Create pairplot (scatterplots for all variable combinations)

df = pd.DataFrame(Y)
sns.pairplot(df, diag_kind="hist")  

# Show plot
plt.show()


from scipy.linalg import svd

# PCA by computing SVD of Y
U, S, Vh = svd(Y, full_matrices=False)

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

# Compute variance explained by principal components 
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title("South african heart disease data: PCA")
# Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
plt.legend(classNames)
plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))

# Output result to screen
plt.show()


threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure()
# Plot attribute coefficients in principal component space
for att in range(V.shape[1]):
    plt.arrow(0, 0, V[att, i], V[att, j])
    plt.text(V[att, i], V[att, j], attributeNames[att])
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xlabel("PC" + str(i + 1))
plt.ylabel("PC" + str(j + 1))
plt.grid()
# Add a unit circle
plt.plot(
    np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01)), color = 'purple'
)
plt.title("Attribute coefficients")
plt.axis("equal")

plt.show()

#We print the principal compenents
print( f'PC1 : {dict(zip(attributeNames_y, np.round(V[:,0],3) ))}' )
print( f'PC2 : {dict(zip(attributeNames_y, np.round(V[:,1],3) ))}' )

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0, 1]
legendStrs = ["PC" + str(e + 1) for e in pcs]
bw = 0.2
r = np.arange(1, M_y + 1)

for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)

plt.xticks(r + bw, attributeNames[:-1])
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("NanoNose: PCA Component Coefficients")
plt.tight_layout()
plt.show()