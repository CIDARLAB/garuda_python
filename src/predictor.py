
# # Python Script to Predict test Vectors

import itertools
import numpy as np
import scipy 
from sklearn.externals import joblib
import os
import argparse
import sys

# # Function convert test vector to desirable formate

def format_test_vector(test_vector, feature_name):
    test_data = (np.in1d(feature_name,test_vector)).astype(int)
    return test_data


# # Function to get predictions

def get_prediction(test_data, model, class_name):
    pred_label = np.empty((0), np.str_)
    if len(test_data.shape) == 1:
        test_data = test_data.reshape(1, -1)
    pred = model.predict(test_data)
    pred_prob = np.max(model.predict_proba(test_data), axis=1)
    for i in range(len(pred)):
        pred_label = np.append(pred_label, class_name[pred[i]])
    return pred_label, pred_prob


if __name__ == "__main__":

    # # Load test data

    parser = argparse.ArgumentParser()
    parser.add_argument("testData", help="enter a test vector for the prediction model")
    test_arg = parser.parse_args()
    test_data = np.asarray([x.strip() for x in  test_arg.testData.split(',')])
    # print(test_data)


    # # Import Feature names and Class names

    model_name = np.load(os.path.join(os.getcwd() ,'garuda_python/model/feature_class_name.npz'))
    feature_name = model_name['arr_0']
    class_name = model_name['arr_1']


    # # Load the trained Linear SVM models


    # classfier without weighting
    clf = joblib.load(os.path.join(os.getcwd() ,'garuda_python/model/classifier_no_weighting.pkl'))
    # classfier class weighting
    wclf = joblib.load(os.path.join(os.getcwd() ,'garuda_python/model/classifier_class_weighting.pkl'))
    # classfier sample weighting
    wsclf = joblib.load(os.path.join(os.getcwd() ,'garuda_python/model/classifier_sample_weighting.pkl'))


    # # Testing 

    test_in = format_test_vector(test_data, feature_name)
    
    print("Linear SVM without weighting")
    pl, pp = get_prediction(test_in, clf, class_name)
    print(pl[0], pp[0]*100)
    
    print("Linear SVM with weighting")
    pl, pp = get_prediction(test_in, wclf, class_name)
    print(pl[0], pp[0]*100)

    print("Linear SVM with sample weighting")
    pl, pp = get_prediction(test_in, wsclf, class_name)
    print(pl[0], pp[0]*100)

    sys.exit(0)



