from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
from datetime import date
import os
import json
import random
import string
from model import *


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_svm'
ALLOWED_EXTENSION = set(['png', 'jpeg', 'jpg', 'pdf'])
FOLDER_IMAGES = 'images'
FOLDER_EXCEL = 'excel'
FOLDER_FILE = 'file'
app.config['UPLOAD_FOLDER'] = os.path.join('static', FOLDER_IMAGES)

mysql = MySQL(app)


def mysqlconnect():
    try:
        db_connection = MySQLdb.connect(
            app.config['MYSQL_HOST'], app.config['MYSQL_USER'], app.config['MYSQL_PASSWORD'], app.config['MYSQL_DB'])
        db_connection.close()
        return 0
    except:
        return '''<h1> Can't Connect To Database !!! </h1>'''


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase + string.digits
    result_str = ''.join((random.choice(letters) for i in range(length)))
    return result_str


@app.route('/')
def index():
    check = mysqlconnect()
    if check == 0:
        return render_template('index.html')
    return check


@app.route('/home')
def home():
    check = mysqlconnect()
    if check == 0:
        cursor = mysql.connection.cursor(
            MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM STORE_FILES WHERE AKTIF = 1 ")
        allImage = cursor.fetchall()
        cursor.close()
        return render_template('home.html', data=allImage)
    return check


@app.route('/uploadFiles', methods=['POST', 'GET'])
def upload_file():
    check = mysqlconnect()
    if check == 0:
        if request.method == 'POST':
            try:
                files = request.files.getlist("file")
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = file.filename
                        file.save(os.path.join(
                            app.config['UPLOAD_FOLDER'],  filename))
                        file_name, file_ext = os.path.splitext(filename)
                        dir = FOLDER_IMAGES + '/' + filename
                        try:
                            cursor = mysql.connection.cursor(
                                MySQLdb.cursors.DictCursor)
                            cursor.execute(
                                "SELECT COUNT(*) AS TOT FROM STORE_FILES WHERE IMAGE_NAME = '"+file_name+"' AND AKTIF = 1 ")
                            count = cursor.fetchone()

                            if (count['TOT'] == 0):
                                cursor.nextset()
                                cursor.execute("""INSERT INTO STORE_FILES (IMAGE_NAME, EXT, DIR, AKTIF, DATE, NOTE) VALUES (%s, %s, %s, %s, %s, %s)""", (
                                    file_name, file_ext, dir, 1, date.today(), ''))
                                mysql.connection.commit()
                            cursor.close()
                            data = {'respon': 00, 'data': '00'}
                        except:
                            data = {'respon': 99, 'data': '00'}
                return jsonify(data)
            except:
                return 99
        else:
            return redirect("/")
    return check


@app.route('/deleteFiles', methods=['POST', 'GET'])
def delete_file():
    check = mysqlconnect()
    if check == 0:
        if request.method == 'POST':
            try:
                idImage = request.form.get('id')
                print(idImage)
                cursor = mysql.connection.cursor(
                    MySQLdb.cursors.DictCursor)
                cursor.execute(
                    "UPDATE STORE_FILES SET AKTIF = 0 WHERE ID = %s", (idImage,))
                mysql.connection.commit()
                cursor.close()
                data = {'respon': 00, 'data': '00'}
            except:
                data = {'respon': 99, 'data': '00'}
            return jsonify(data)
        else:
            return redirect("/")
    return check


@app.route('/ekstraksi', methods=['POST', 'GET'])
def Ekstraksi():
    return render_template('ekstraksi.html')


@app.route('/ekstraksiTblGambar', methods=['POST', 'GET'])
def EkstraksiImage():
    if request.method == 'POST':
        check = mysqlconnect()
        if check == 0:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("SELECT * FROM STORE_FILES WHERE AKTIF = 1 ")
            allImage = cursor.fetchall()
            cursor.close()
            return jsonify(data=allImage)
        return check
    else:
        return redirect("/")


@app.route('/ekstraksiCitra', methods=['POST', 'GET'])
def EkstraksiCitra():
    if request.method == 'POST':
        cursor = mysql.connection.cursor(
            MySQLdb.cursors.DictCursor)
        cursor.execute(
            "SELECT COUNT(*) AS JAHE FROM STORE_FILES WHERE AKTIF = 1 AND IMAGE_NAME LIKE 'Jahe%' ")
        count = cursor.fetchone()
        cursor.close()
        result = ekstraksiCitra_HSV_GLCM(
            app.config['UPLOAD_FOLDER'] + '\\', count['JAHE'])
        return result
    return redirect("/")


@app.route('/ekstraksiCitraNormal', methods=['POST', 'GET'])
def EkstraksiCitraNormalisasi():
    if request.method == 'POST':
        result = ekstraksiCitra_NormalisasiData()
        return result
    return redirect("/")


@app.route('/model', methods=['POST', 'GET'])
def Model():
    return render_template('model.html')


@app.route('/modelIQR', methods=['POST', 'GET'])
def ModelIQR():
    if request.method == 'POST':
        result = ekstraksiCitra_IQR()
        return result
    return redirect("/")


@app.route('/modelIndexLabel', methods=['POST', 'GET'])
def ModelIndexLabel():
    if request.method == 'POST':
        result = SVM_Index_Label()
        return result
    return redirect("/")


@app.route('/modelLearningData', methods=['POST', 'GET'])
def ModelLearningData():
    if request.method == 'POST':
        ratio = request.form.get('countSplit')
        result = SVM_Split_Learning(ratio)
        return result
    return redirect("/")


@app.route('/modelTestingData', methods=['POST', 'GET'])
def ModelTestingData():
    if request.method == 'POST':
        result = SVM_Split_TestData()
        return result
    return redirect("/")


@app.route('/modelCreateModel', methods=['POST', 'GET'])
def ModelCreateModel():
    if request.method == 'POST':
        result = SVMTrainingModel()
        return jsonify(1)
    return redirect("/")


@app.route('/modelWeighData', methods=['POST', 'GET'])
def ModelWeightData():
    if request.method == 'POST':
        result = getWeightData()
        return result
    return redirect("/")


@app.route('/modelSupportVector', methods=['POST', 'GET'])
def ModelSupportVector():
    if request.method == 'POST':
        result = getSupportVectorData()
        return result
    return redirect("/")


@app.route('/clasifikasi', methods=['POST', 'GET'])
def Clasifikasi():
    return render_template('clasifikasi.html')


@app.route('/clasifikasiTesting', methods=['POST', 'GET'])
def ClasifikasiSVM():

    X_testing, Y_testing = getTestData()
    with open('static/model/clf.pickle', 'rb') as f:
        clf = pickle.load(f)
    Y_pred = clf.predict(X_testing)

    data = {'acc': str(metrics.accuracy_score(Y_testing, Y_pred)),
            'prec': str(metrics.precision_score(Y_testing, Y_pred)),
            'rcal': str(metrics.recall_score(Y_testing, Y_pred)),
            'iteration': str(int(clf.n_iter_)),
            'kernel': str(clf.kernel),
            'b': str(clf.intercept_),
            'c': str(clf.C)}
    return data


@app.route('/saveAll', methods=['POST', 'GET'])
def SaveAll():
    check = mysqlconnect()
    if check == 0:
        if request.method == 'POST':
            acc = request.form.get('acc')
            prec = request.form.get('prec')
            rcal = request.form.get('rcal')
            id = get_random_string(8)
            data_training = getLearningData()
            data_testing = getTestData()
            data_ekstraksi = getDataEkstraksi()
            data_normalisasi = getDataNormalisasi()
            jml_train = len(data_training[0])
            jml_test = len(data_testing[0])
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                "SELECT COUNT(*) AS JUM FROM MASTER WHERE AKTIF = 1 ")
            count = cursor.fetchone()
            cursor.nextset()
            cursor.execute("""INSERT INTO MASTER (ID_MASTER, NAME, AKTIF, DATE_CREATED) VALUES (%s, %s, %s, %s)""",
                           (id, 'Model_' + str(count['JUM']), 1, date.today(), ))
            cursor.nextset()
            for idx, data in enumerate(data_training[0]):
                cursor.execute("""INSERT INTO EKTRAKSI_CITRA (ID_MASTER, LABEL, TIPE, AKTIF, DATE, MEAN_HUE, STD_HUE, SKEW_HUE, MEAN_SATUR, STD_SATUR, SKEW_SATUR, MEAN_VALUE, STD_VALUE, SKEW_VALUE, CORRELATION_0, CORRELATION_45, CORRELATION_90, CORRELATION_135, HOMOGENEITY_0, HOMOGENEITY_45, HOMOGENEITY_90, HOMOGENEITY_135, DISSIMILARITY_0, DISSIMILARITY_45, DISSIMILARITY_90, DISSIMILARITY_135, CONTRAST_0, CONTRAST_45, CONTRAST_90, CONTRAST_135, ENERGY_0, ENERGY_45, ENERGY_90, ENERGY_135, ASM_0, ASM_45, ASM_90, ASM_135) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                               (id, data_training[1][idx], 'training', '1', date.today(), data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31], data[32],))
            cursor.nextset()
            for idx, data in enumerate(data_testing[0]):
                cursor.execute("""INSERT INTO EKTRAKSI_CITRA (ID_MASTER, LABEL, TIPE, AKTIF, DATE, MEAN_HUE, STD_HUE, SKEW_HUE, MEAN_SATUR, STD_SATUR, SKEW_SATUR, MEAN_VALUE, STD_VALUE, SKEW_VALUE, CORRELATION_0, CORRELATION_45, CORRELATION_90, CORRELATION_135, HOMOGENEITY_0, HOMOGENEITY_45, HOMOGENEITY_90, HOMOGENEITY_135, DISSIMILARITY_0, DISSIMILARITY_45, DISSIMILARITY_90, DISSIMILARITY_135, CONTRAST_0, CONTRAST_45, CONTRAST_90, CONTRAST_135, ENERGY_0, ENERGY_45, ENERGY_90, ENERGY_135, ASM_0, ASM_45, ASM_90, ASM_135) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                               (id, data_testing[1][idx], 'testing', '1', date.today(), data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31], data[32],))
            cursor.nextset()
            for idx, data in enumerate(data_ekstraksi[0]):
                cursor.execute("""INSERT INTO EKTRAKSI_CITRA (ID_MASTER, LABEL, TIPE, AKTIF, DATE, MEAN_HUE, STD_HUE, SKEW_HUE, MEAN_SATUR, STD_SATUR, SKEW_SATUR, MEAN_VALUE, STD_VALUE, SKEW_VALUE, CORRELATION_0, CORRELATION_45, CORRELATION_90, CORRELATION_135, HOMOGENEITY_0, HOMOGENEITY_45, HOMOGENEITY_90, HOMOGENEITY_135, DISSIMILARITY_0, DISSIMILARITY_45, DISSIMILARITY_90, DISSIMILARITY_135, CONTRAST_0, CONTRAST_45, CONTRAST_90, CONTRAST_135, ENERGY_0, ENERGY_45, ENERGY_90, ENERGY_135, ASM_0, ASM_45, ASM_90, ASM_135) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                               (id, data_ekstraksi[1][idx], 'ekstraksi', '1', date.today(), data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31], data[32],))
            cursor.nextset()
            for idx, data in enumerate(data_normalisasi[0]):
                cursor.execute("""INSERT INTO EKTRAKSI_CITRA (ID_MASTER, LABEL, TIPE, AKTIF, DATE, MEAN_HUE, STD_HUE, SKEW_HUE, MEAN_SATUR, STD_SATUR, SKEW_SATUR, MEAN_VALUE, STD_VALUE, SKEW_VALUE, CORRELATION_0, CORRELATION_45, CORRELATION_90, CORRELATION_135, HOMOGENEITY_0, HOMOGENEITY_45, HOMOGENEITY_90, HOMOGENEITY_135, DISSIMILARITY_0, DISSIMILARITY_45, DISSIMILARITY_90, DISSIMILARITY_135, CONTRAST_0, CONTRAST_45, CONTRAST_90, CONTRAST_135, ENERGY_0, ENERGY_45, ENERGY_90, ENERGY_135, ASM_0, ASM_45, ASM_90, ASM_135) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                               (id, data_normalisasi[1][idx], 'normalisasi', '1', date.today(), data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31], data[32],))
            cursor.nextset()
            cursor.execute("""INSERT INTO PREDICT (ID_MASTER, JML_DATA, JML_DATA_TRAIN, JML_DATA_TEST, ACC, PREC, RCALL, AKTIF, DATE) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                           (id, jml_train+jml_test, jml_train, jml_test, acc, prec, rcal, 1, date.today(), ))
            mysql.connection.commit()
            cursor.close()

        return redirect("/")
    return check


@app.route('/dataModel', methods=['POST', 'GET'])
def DataModelAll():
    return render_template('dataAll.html')


@app.route('/dataModelPredict', methods=['POST', 'GET'])
def DataModelPredict():
    check = mysqlconnect()
    if check == 0:
        if request.method == 'POST':
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute(
                "SELECT A.ID_MASTER, A.NAME, B.JML_DATA, B.JML_DATA_TRAIN, JML_DATA_TEST, ACC, PREC, RCALL FROM MASTER A LEFT JOIN PREDICT B ON A.ID_MASTER=B.ID_MASTER WHERE A.AKTIF = 1 ORDER BY A.NAME ")
            allImage = cursor.fetchall()
            cursor.close()
            return jsonify(data=allImage)
        return redirect("/")
    return check


if __name__ == '__main__':
    app.run(debug=True, port='5001')
