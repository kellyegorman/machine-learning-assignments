# plot the ROC curve for a random classifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

np.random.seed(0)
# 500 data points to get a smoother curve
y_true = np.random.randint(0, 2, 500)  
# 500 random scores 0 -> 1
y_scores = np.random.rand(500)  

# compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
#diagonal line
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Classifier')
plt.legend(loc="lower right")
plt.grid()
plt.show()