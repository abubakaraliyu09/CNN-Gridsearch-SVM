import os,cv2,time,argparse,warnings
os.environ['IF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np,matplotlib.pyplot as plt
from imutils import paths
from sklearn.model_selection import GridSearchCV
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split, KFold, cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix,classification_report, precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16
from sklearn.svm import SVC
from keras.layers import BatchNormalization
warnings.filterwarnings('ignore')

data = []
labels = []
IMG_SIZE=224
print("[INFO] loading images...")
def load_data(DIR):
    imagePaths = list(paths.list_images(DIR))
    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)
    print('[INFO]', len(data), "Data Loaded Successfully!")
    return data, labels
load_data('Binary_data/small_data')#provide data directory here

def pre_trained_model():
    print("[INFO] loading Pretrained Model...")
    model = VGG16(weights='imagenet')
    model = Model(inputs=model.input, outputs=model.layers[-3].output)
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = False
    return model

def extract_features():
    model = pre_trained_model()
    print("[INFO] Extracting Features...")
    counter=0
    features=[]
    for point in data:
        img=img_to_array(point)
        img=np.expand_dims(img, axis=0)
        feat=model.predict(img)
        features.append(feat)
        counter+=1
        if counter%1==0:
            print("[INFO]: Extracting features of image {}".format(counter),"out of", len(data), "images")
    return features

def pre_process():
    features = extract_features()
    X=np.array(features)
    Y= np.array(labels)
    #Reshape the features
    X=X.reshape(X.shape[0], 1*1*4096)# ResNet50=2048
    le=LabelEncoder()
    Y=le.fit_transform(Y)
    print("features.shape:",X.shape)
    print("labels.shape:",Y.shape)
    return  X,Y
X, Y = pre_process()
X_train,X_test,y_train,y_test = train_test_split(X,Y)

print('------------Grid-Search with Cross-Validation--------------')
def classifier():
    param_grid = {'C': 10. ** np.arange(-3, 3),
                  'gamma' : 10. ** np.arange(-5, 0)}
    np.set_printoptions(suppress=True)
    print(param_grid)
    grid_search = GridSearchCV(SVC(), param_grid, verbose=3, cv=5)

    grid_search.fit(X_train, y_train)
    grid_search.predict(X_test)
    grid_search.score(X_test, y_test)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)


    # We extract just the scores

    scores = grid_search.cv_results_['mean_test_score']
    scores = np.array(scores).reshape(6, 5)

    plt.matshow(scores)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(5), param_grid['gamma'])
    plt.yticks(np.arange(6), param_grid['C'])
classifier()

