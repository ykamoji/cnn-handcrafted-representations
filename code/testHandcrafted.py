import numpy as np
import os
import utils
import time
import glob

import digitFeatures
import linearModel

# There are three versions of MNIST dataset
dataTypes = ['digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat']
# dataTypes = ['digits-normal.mat']

# You have to implement three types of features
featureTypes = ['pixel', 'hog', 'lbp']
# featureTypes = ['lbp']

# You have to implement feature transformations
featureTransform = 'none' # options are 'none', 'sqrt', 'l2'

# Accuracy placeholder
accuracy = np.zeros((len(dataTypes), len(featureTypes)))
trainSet = 1
testSet = 3

paramMap = {
    "digits-normal.mat": {
        "pixel": { "lambda":-3, "eta": -1, "featureTransform":"sqrt" },
        "hog": { "lambda": -1, "eta": -2, "featureTransform":"sqrt" },
        "lbp": { "lambda": -5, "eta": -2, "featureTransform":"sqrt" }
    },
    "digits-scaled.mat": {
        "pixel": { "lambda": -4, "eta": -2, "featureTransform":"none" },
        "hog": { "lambda": -3, "eta": -2, "featureTransform":"sqrt" },
        "lbp": { "lambda": -3, "eta": -2, "featureTransform":"sqrt" }
    },
    "digits-jitter.mat": {
        "pixel": { "lambda": -5, "eta": 0, "featureTransform":"sqrt" },
        "hog": { "lambda": 0, "eta": -3, "featureTransform":"none" },
        "lbp": { "lambda": -3, "eta": -2, "featureTransform":"sqrt" }
    },

}

for i in range(len(dataTypes)):

    dataType = dataTypes[i]

    #Load data
    path = os.path.join('..', 'data', dataType)
    data = utils.loadmat(path)
    print('+++ Loading dataset: {} ({} images)'.format(dataType, data['x'].shape[2]))

    # To montage the digits in the val set use
    # utils.montageDigits(data['x'][:, :, data['set']==1])

    for j in range(len(featureTypes)):
        featureType = featureTypes[j]

        params = {
            'lambda': 10 ** paramMap[dataType][featureType]['lambda'],
            'maxiter': 1000,
            'eta': 10 ** paramMap[dataType][featureType]['eta'],
            'addBias': True
        }

        # Extract features
        tic = time.time()
        features = digitFeatures.getFeature(data['x'], featureType, paramMap[dataType][featureType]['featureTransform'])
        print('{:.2f}s to extract {} features ({} dim).'.format(time.time() - tic,
                                                                featureType, features.shape[0]))

        file = glob.glob(f"{dataType}_{featureType}*.npy")
        if file:
            model = np.load(file[0], allow_pickle=True).item()
        else:
            # Train model
            tic = time.time()
            model = linearModel.train(features[:, data['set'] == trainSet], data['y'][data['set'] == trainSet], params)
            print('{:.2f}s to train model.'.format(time.time() - tic))

        # Test the model
        tic = time.time()
        ypred = linearModel.predict(model, features[:, data['set'] == testSet])
        print('{:.2f}s to test model.'.format(time.time() - tic))
        y = data['y'][data['set'] == testSet]

        # Measure accuracy
        (acc, conf) = utils.evaluateLabels(y, ypred, False)
        print('Accuracy [testSet={}] {:.2f} %\n'.format(testSet, acc * 100))
        accuracy[i, j] = acc

        if testSet == 2:
            if file:
                best_acc = file[0].split('_')[-1].split('.npy')[0]
                if float(best_acc) < acc:
                    print(f"Got better model for {dataType}_{featureType} !")
                    np.save(f"{dataType}_{featureType}_{acc}", model)
                else:
                    accuracy[i, j] = float(best_acc)
            else:
                np.save(f"{dataType}_{featureType}_{acc}", model)


# Print the results in a table
print('+++ Accuracy Table [trainSet={}, testSet={}]'.format(trainSet, testSet))
print('--------------------------------------------------')
print('dataset\t\t\t', end="")
for j in range(len(featureTypes)):
    print('{}\t'.format(featureTypes[j]), end="")

print()
print('--------------------------------------------------')
for i in range(len(dataTypes)):
    print('{}\t'.format(dataTypes[i]), end="")
    for j in range(len(featureTypes)):
        print('{:.2f}\t'.format(accuracy[i, j]*100), end="")
    print()

# Once you have optimized the hyperparameters, you can report test accuracy
# by setting testSet=3. Do not optimize your hyperparameters on the
# test set. That would be cheating!
