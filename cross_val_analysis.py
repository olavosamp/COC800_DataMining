import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

def cross_val_analysis(n_split=10,classifier=None,trainDf=None, y_train=None):
	'''#"Classification and ROC analysis
	#Run classifier with cross-validation and plot ROC curves"'''

	kf = KFold(n_splits=n_split)
	kf.get_n_splits(trainDf)
		
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	i = 0
	#start_time = time.time() 
	for train, val in kf.split(trainDf,y_train):
		print('Train Process for %i Fold'%(i+1))
		#print("TRAIN:", train_index, "TEST:", test_index)
		#trainX, valX = trainDf[train_index], trainDf[val_index]
		#trainY, valY = y_train[train_index], y_train[val_index]
		probas_ = classifier.fit(trainDf.iloc[train], y_train[train]).predict_proba(trainDf.iloc[val])
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(y_train[val], probas_[:, 1])
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = auc(fpr, tpr)
		aucs.append(roc_auc)
		plt.plot(fpr, tpr, lw=1, alpha=0.3,
				 label='ROC fold %d (AUC = %0.2f)' % (i, 100*roc_auc))

		i += 1
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
			 label='Luck', alpha=.8)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b',
			 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (100*mean_auc, 100*std_auc),
			 lw=2, alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
					 label=r'$\pm$ 1 std. dev.')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
