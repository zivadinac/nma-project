import pickle
import numpy as np
import matplotlib.pyplot as plt
import movement 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA
import training_model

if False:
    dat = np.load('data/stringer_spontaneous.npy', allow_pickle=True).item()
    neural_data = dat['sresp']
    run_data = dat['run']
    run_onset, run_speed = movement.detect_movement_onset(run_data)
    det_window = 3
    #penalty="l2"
    penalty = "l1"

    pca_coms = [10, 50, 100, 500, 1000, 2500, 5000, 7018]
    pca_train_acc = []
    pca_test_acc = []
    for pca_com in pca_coms:
        neural_data, run_onset, run_speed = training_model.load_data()
        training_model.model(neural_data, run_onset, det_window, penalty, pca_com=pca_com)
        pca_train_acc.append(training_model.train_acc[(pca_com, det_window, penalty)])
        pca_test_acc.append(training_model.test_acc[(pca_com, det_window, penalty)])
        training_model.clean_dicts()
        print(f"Finished for {pca_com} PCA components.")

    #neuron_nums = [10, 50, 100, 500, 1000, 2500, 5000, 7500, 10000, len(neural_data)]
    neuron_nums = [10, 50, 100, 500, 1000, 2500, 5000, 7500, 10000]
    models_per_neuron_num = 100
    neurons_train_acc = []
    neurons_test_acc = []
    for neuron_num in neuron_nums:
        neural_data, run_onset, run_speed = training_model.load_data()
        test_acc = []
        train_acc = []
        for i in range(models_per_neuron_num):
            training_model.model(neural_data, run_onset, det_window, penalty, neuron_num=neuron_num)
            train_acc.append(training_model.train_acc[(neuron_num, det_window, penalty)])
            test_acc.append(training_model.test_acc[(neuron_num, det_window, penalty)])
            training_model.clean_dicts()
        neurons_train_acc.append(np.mean(train_acc))
        neurons_test_acc.append(np.mean(test_acc))
        print(f"Finished for {neuron_num} neurons.")

    # Ploting 
    plt.plot(pca_coms, pca_train_acc, label="train")
    plt.plot(pca_coms, pca_test_acc, label="test")
    plt.legend()
    plt.title("Accuracy vs number of PCA components (l1 reg)", fontsize=30)
    plt.show()

    plt.plot(neuron_nums, neurons_train_acc, label="train")
    plt.plot(neuron_nums, neurons_test_acc, label="test")
    plt.legend()
    plt.title("Accuracy vs number of neurons (l1 reg)", fontsize=30)
    plt.show()

    with open(f"acc_vs_feat_len_{penalty}.pck", "wb") as f:
        d = {}
        d["pca_coms"] = pca_coms
        d["pca_train_acc"] = pca_train_acc
        d["pca_test_acc"] = pca_test_acc
        d["neuron_nums"] = neuron_nums
        d["neurons_train_acc"] = neurons_train_acc
        d["neurons_test_acc"] = neurons_test_acc
        pickle.dump(d, f)
else:
    penalty = "l2"
    with open(f"acc_vs_feat_len_{penalty}.pck", "rb") as f:
        d = pickle.load(f)

    plt.plot(d["pca_coms"], d["pca_train_acc"], label="train")
    plt.plot(d["pca_coms"], d["pca_test_acc"], label="test")
    plt.ylim(ymin=0.5, ymax=1.)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.title(f"Accuracy vs number of PCA components ({penalty} reg)", fontsize=30)
    plt.show()

    train_ci = np.std(d["neurons_train_acc"])# / np.mean(d["neurons_train_acc"])
    test_ci = np.std(d["neurons_test_acc"])# / np.mean(d["neurons_test_acc"])
    plt.plot(d["neuron_nums"], d["neurons_train_acc"], label="train")
    #plt.errorbar(d["neuron_nums"], d["neurons_train_acc"], train_ci)
    plt.fill_between(d["neuron_nums"], d["neurons_train_acc"]-train_ci, d["neurons_train_acc"]+train_ci, alpha=.1)
    plt.plot(d["neuron_nums"], d["neurons_test_acc"], label="test")
    plt.fill_between(d["neuron_nums"], d["neurons_test_acc"]-test_ci, d["neurons_test_acc"]+test_ci, alpha=.1)
    #plt.errorbar(d["neuron_nums"], d["neurons_test_acc"], test_ci)
    plt.ylim(ymin=0.5, ymax=1.)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.title(f"Accuracy vs number of neurons ({penalty} reg)", fontsize=30)
    plt.show()

