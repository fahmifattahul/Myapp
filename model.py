import cv2
import glob
import numpy as np
import pandas as pd
import xlsxwriter as xls
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, regionprops_table
from scipy.stats import skew
from openpyxl import Workbook
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix


def ekstraksiCitra_HSV_GLCM(dirImage, jumlah):
    book = xls.Workbook('static/file/excel/data_ekstraksi.xlsx')
    sheet = book.add_worksheet()

    column = 0

    # kolom hsv
    hsv_feature = ['mean_hue', 'std_hue', 'skew_hue', 'mean_satur',
                   'std_satur', 'skew_satur', 'mean_value', 'std_value', 'skew_value']
    for i in hsv_feature:
        sheet.write(0, column, i)
        column += 1

    # kolom glcm
    glcm_feature = ['correlation', 'homogeneity',
                    'dissimilarity', 'contrast', 'energy', 'ASM']
    angle = ['0', '45', '90', '135']
    for i in glcm_feature:
        for j in angle:
            sheet.write(0, column, i+" "+j)
            column += 1

    label_feature = ['Label']
    for i in label_feature:
        sheet.write(0, column, i)
        column+1

    # baris rimpang
    jenis = ['Jahe', 'Kencur']
    row = 1
    for i in jenis:
        for j in range(1, jumlah+1):
            column = 0
            file_name = dirImage+i+str(j)+'.png'
            img = cv2.imread(file_name)
            print(file_name)

            # preprocessing
            img = cv2.imread(file_name, 1)
            blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            # hsv
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            meanh = np.mean(h)
            stdh = np.std(h)
            skewh = np.mean(skew(h))
            means = np.mean(s)
            stds = np.std(s)
            skews = np.mean(skew(s))
            meanv = np.mean(v)
            stdv = np.std(v)
            skewv = np.mean(skew(v))
            hsv_props = [meanh, stdh, skewh, means,
                         stds, skews, meanv, stdv, skewv]
            for item in hsv_props:
                sheet.write(row, column, item)
                column += 1

            # glcm
            distances = [5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            levels = 256
            symmetric = True
            normed = True

            glcm = graycomatrix(gray, distances, angles,
                                levels, symmetric, normed)

            glcm_props = [propery for name in glcm_feature for propery in graycoprops(glcm, name)[
                0]]
            for item in glcm_props:
                sheet.write(row, column, item)
                column += 1

            # label
            label_name = i
            sheet.write(row, column, label_name)
            column += 1

            row += 1

    book.close()
    excel_file = pd.read_excel('static/file/excel/data_ekstraksi.xlsx')
    excel_file.to_csv("static/file/csv/data_ekstraksi.csv",
                      index=None, header=True)
    json_str = excel_file.to_json(orient='records')
    # print(json_str)
    return json_str


def ekstraksiCitra_NormalisasiData():
    feature_csv = pd.read_csv('static/file/csv/data_ekstraksi.csv')
    split_data = feature_csv.drop('Label', axis=1)
    normalisasi_data = preprocessing.normalize(split_data, norm='l2')
    label_data = feature_csv.iloc[:, -1].values
    feature_baru = pd.DataFrame(normalisasi_data)
    label = pd.DataFrame(label_data)
    feature_baru[33] = label
    feature_baru.to_csv("static/file/csv/data_normalisasi.csv", index=None, header=['mean_hue', 'std_hue', 'skew_hue', 'mean_satur', 'std_satur', 'skew_satur', 'mean_value', 'std_value', 'skew_value', 'correlation 0', 'correlation 45', 'correlation 90',	'correlation 135',	'homogeneity 0',	'homogeneity 45',
                        'homogeneity 90',	'homogeneity 135', 'dissimilarity 0', 'dissimilarity 45', 'dissimilarity 90', 'dissimilarity 135', 'contrast 0', 'contrast 45', 'contrast 90', 'contrast 135', 'energy 0', 'energy 45', 'energy 90', 'energy 135', 'ASM 0', 'ASM 45', 'ASM 90', 'ASM 135', 'Label'])
    normalisasi_file = pd.read_csv('static/file/csv/data_normalisasi.csv')
    json_str = normalisasi_file.to_json(orient='records')
    # print(json_str)
    return json_str


def ekstraksiCitra_IQR():
    feature_csv = pd.read_csv('static/file/csv/data_normalisasi.csv')
    IQR_Max = []
    IQR_Min = []
    for i in range(33):
        data = feature_csv.iloc[:, i].values
        nilai_min = np.min(data)
        nilai_max = np.max(data)

        # Hitung nilai Q1 dan Q3
        Q1 = np.quantile(data, .25)
        Q3 = np.quantile(data, .75)

        # Hitung nilai IQR
        IQR = Q3 - Q1
        min_IQR = Q1 - 1.5 * IQR
        max_IQR = Q3 + 1.5 * IQR

        IQR_Max += max_IQR,
        IQR_Min += min_IQR,

        iqr_Max = np.array(IQR_Max)
        iqr_Min = np.array(IQR_Min)
        IQR = pd.DataFrame({'IQR_Max': iqr_Max, 'IQR_Min': iqr_Min})
        json_str = IQR.to_json()
    return json_str


def SVM_Index_Label():
    feature_csv = pd.read_csv('static/file/csv/data_normalisasi.csv')
    df = pd.DataFrame(feature_csv)
    labelencoder = LabelEncoder()
    df['Label'] = labelencoder.fit_transform(df['Label'])  # table

    # df (run)
    indexLabel = df.to_json(orient='records')
    return indexLabel


def Split_Data_Learning_Test(input_frac):
    feature_csv = pd.read_csv('static/file/csv/data_normalisasi.csv')
    df = pd.DataFrame(feature_csv)
    labelencoder = LabelEncoder()
    df['Label'] = labelencoder.fit_transform(df['Label'])  # table

    # df (run)
    X = int(input_frac)/100
    learning_data = df.sample(frac=X)  # table
    learning_data.to_csv("static/file/csv/data_learning.csv",
                         index=None, header=True)
    test_data = df.drop(learning_data.index)  # table
    test_data.to_csv("static/file/csv/data_test.csv",
                     index=None, header=True)


def SVM_Split_Learning(input_frac):
    Split_Data_Learning_Test(input_frac)
    learning_data = pd.read_csv('static/file/csv/data_learning.csv')
    dataLearning = learning_data.to_json(orient='records')
    return dataLearning


def SVM_Split_TestData():
    learning_data = pd.read_csv('static/file/csv/data_test.csv')
    testData = learning_data.to_json(orient='records')
    return testData


def getLearningData():
    learning_data = pd.read_csv('static/file/csv/data_learning.csv')
    X_learn = learning_data.drop('Label', axis=1)
    Y_learn = learning_data['Label']
    X_learning = X_learn.to_numpy()
    Y_learning = Y_learn.to_numpy()
    return (X_learning, Y_learning)


def getTestData():
    test_data = pd.read_csv('static/file/csv/data_test.csv')
    X_test = test_data.drop('Label', axis=1)
    Y_test = test_data['Label']
    X_testing = X_test.to_numpy()
    Y_testing = Y_test.to_numpy()
    return (X_testing, Y_testing)


def getDataEkstraksi():
    data_ekstraksi = pd.read_csv('static/file/csv/data_ekstraksi.csv')
    X_learn = data_ekstraksi.drop('Label', axis=1)
    Y_learn = data_ekstraksi['Label']
    X_learning = X_learn.to_numpy()
    Y_learning = Y_learn.to_numpy()
    return (X_learning, Y_learning)


def getDataNormalisasi():
    data_normalisasi = pd.read_csv('static/file/csv/data_normalisasi.csv')
    X_learn = data_normalisasi.drop('Label', axis=1)
    Y_learn = data_normalisasi['Label']
    X_learning = X_learn.to_numpy()
    Y_learning = Y_learn.to_numpy()
    return (X_learning, Y_learning)


def getWeightData():
    data_weight = pd.read_csv('static/file/csv/data_weight.csv')
    dataWeight = data_weight.to_json(orient='records')
    return dataWeight


def getSupportVectorData():
    data_vector = pd.read_csv('static/file/csv/data_support_vector.csv')
    dataVector = data_vector.to_json(orient='records')
    return dataVector


def SVMTrainingModel():
    X_learning, Y_learning = getLearningData()
    X_testing, Y_testing = getTestData()

    clf = svm.SVC(kernel='linear')
    clf.fit(X_learning, Y_learning)

    Y_pred = clf.predict(X_testing)  # table

    weight = pd.DataFrame(clf.coef_)
    support_vector = pd.DataFrame(clf.support_vectors_)

    weight.to_csv("static/file/csv/data_weight.csv", index=None, header=['mean_hue', 'std_hue', 'skew_hue', 'mean_satur', 'std_satur', 'skew_satur', 'mean_value', 'std_value', 'skew_value', 'correlation 0', 'correlation 45', 'correlation 90',	'correlation 135',	'homogeneity 0',	'homogeneity 45',
                                                                         'homogeneity 90',	'homogeneity 135', 'dissimilarity 0', 'dissimilarity 45', 'dissimilarity 90', 'dissimilarity 135', 'contrast 0', 'contrast 45', 'contrast 90', 'contrast 135', 'energy 0', 'energy 45', 'energy 90', 'energy 135', 'ASM 0', 'ASM 45', 'ASM 90', 'ASM 135'])
    support_vector.to_csv("static/file/csv/data_support_vector.csv", index=None, header=['mean_hue', 'std_hue', 'skew_hue', 'mean_satur', 'std_satur', 'skew_satur', 'mean_value', 'std_value', 'skew_value', 'correlation 0', 'correlation 45', 'correlation 90',	'correlation 135',	'homogeneity 0',	'homogeneity 45',
                                                                                         'homogeneity 90',	'homogeneity 135', 'dissimilarity 0', 'dissimilarity 45', 'dissimilarity 90', 'dissimilarity 135', 'contrast 0', 'contrast 45', 'contrast 90', 'contrast 135', 'energy 0', 'energy 45', 'energy 90', 'energy 135', 'ASM 0', 'ASM 45', 'ASM 90', 'ASM 135'])

    cm = confusion_matrix(Y_testing, Y_pred)
    sn.heatmap(cm, annot=True)
    plt.xlabel('Kelas Klasifikasi')
    plt.ylabel('Kelas Asli')
    plt.savefig('static/headmap/images/headmap_image.png')
    plt.close('all')

    with open('static/model/clf.pickle', 'wb') as f:
        pickle.dump(clf, f)
    return Y_pred


# def SVM_Model():
#     # index label
#     feature_csv = pd.read_csv('static/file/csv/data_normalisasi.csv')
#     df = pd.DataFrame(feature_csv)
#     labelencoder = LabelEncoder()
#     df['Label'] = labelencoder.fit_transform(df['Label'])  # table

#     # df (run)
#     input_frac = 80  # form buat input

#     X = int(input_frac)/100
#     learning_data = df.sample(frac=X)  # table
#     test_data = df.drop(learning_data.index)  # table

#     X_learn = learning_data.drop('Label', axis=1)
#     Y_learn = learning_data['Label']
#     X_test = test_data.drop('Label', axis=1)
#     Y_test = test_data['Label']

#     # split data vector
#     X_learning = X_learn.to_numpy()
#     Y_learning = Y_learn.to_numpy()
#     X_testing = X_test.to_numpy()
#     Y_testing = Y_test.to_numpy()
#     # ex. Y_testing... (run)

#     # training model svm (linear)
#     clf = svm.SVC(kernel='linear')
#     clf.fit(X_learning, Y_learning)

#     Weight = pd.DataFrame(clf.coef_)
#     # print(Weight)
#     Suport_Vevtors = pd.DataFrame(clf.support_vectors_)
#     # print(Suport_Vevtors)
#     # clasifikasi
#     Y_pred = clf.predict(X_testing)  # table
#     # with open('static/clf.pickle', 'wb') as f:
#     #     pickle.dump(clf, f)
#     print(Y_pred)
#     # Y_pred (run)

#     print("Accuracy:", metrics.accuracy_score(Y_testing, Y_pred))
#     print("Precision:", metrics.precision_score(Y_testing, Y_pred))
#     print("Recall:", metrics.recall_score(Y_testing, Y_pred))
#     print('Iteration       = ', int(clf.n_iter_))
#     print('Kernel          = ', clf.kernel)
#     print('b               = ', clf.intercept_)
#     print('C               = ', clf.C)

#     # print(clf.coef_)
#     # print(clf.support_vectors_)

#     cm = confusion_matrix(Y_testing, Y_pred)
#     sn.heatmap(cm, annot=True)
#     plt.xlabel('Kelas Klasifikasi')
#     plt.ylabel('Kelas Asli')
#     plt.savefig('test_image.png')


# ekstraksiCitra_HSV_GLCM('static/images/')
# ekstraksiCitra_NormalisasiData()
# ekstraksiCitra_IQR()
# SVM_Model()
