from os import listdir                  #use for get dir file in folder
from os.path import isfile, join, isdir #use for get dir file in folder
import sys #use for arvg
import pickle   #use for load file pkl
import tensorflow as tf     #use tensorflow
import facenet              #use for 
import align.detect_face    #use for detect face
import numpy as np
from sklearn.metrics import classification_report,f1_score, accuracy_score
import pandas as pd # use for save predict Y
import os   #use create file
from scipy import misc #use readimg
import matplotlib.pyplot as plt




def Load(pathModelsFolder, FACENET_MODEL_PATH, nameML):
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160

    #use tensorflow
    with tf.Graph().as_default():
        # Set up GPU
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load model MTCNN
            # print('Loading feature extraction model') as .pb
            facenet.load_model(FACENET_MODEL_PATH)

            # Get tensor input and output
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            #get list name file pkl
            listModelinFolder = [join(pathModelsFolder, f) for f in listdir(pathModelsFolder) if isfile(join(pathModelsFolder, f))]

            #get dataset Test
            # _, _, testX, testY = t.loadDuLieu(pathDataSetFolder) ## chu y, tham so cua ham loadDuLieu
            # listNameofDataSet = t.timFolderName(pathDataSetFolder)
            # predicY = []
            tempDictY = {}
            res  = [[],[],[],[],[],[],[],[],[]]
            if not os.path.exists("./tempLuu"):
                os.makedirs("./tempLuu")
            save = open('./tempLuu/classification_report_'+ nameML+'.txt', 'w')
            for ia in range(len(listModelinFolder)):
                tempPredicY = []
                #read file pkl
                with open(listModelinFolder[ia], 'rb') as file:
                    model, listNameofDataSet,testX, testY  = pickle.load(file)
                #duyet tung anh
                for ib in testX:
                    IMG_READ = misc.imread(ib)
                    scaled = facenet.prewhiten(IMG_READ)                   
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    # classifier
                    tempPredicY.append(model.predict(emb_array))
                    # predictions = model.predict_proba(emb_array)
                    # best_class_indices = np.argmax(predictions, axis=1)
                    # tempPredicY.append(best_class_indices[0])

                tempDictY[ia] = tempPredicY
                # print(testY[ia])
                # print(predicY)
                # predicY.append(tempPredicY)

                save.write(classification_report(testY, tempPredicY, digits = 4, target_names= listNameofDataSet) + "\n")
                print( str(ia+1) + "/10", "accuracy:" , accuracy_score(testY, tempPredicY))
                tempScore = f1_score(testY, tempPredicY, average=None)
                for ib in range(len(res)):
                    res[ib].append(tempScore[ib])
            dfpredicY = pd.DataFrame(tempDictY)
            dfpredicY.to_csv('./tempLuu/predict_'+ nameML+'.txt', index=None)
            save.close()
        #luu f1_score and name of class
        with open("./tempLuu/f_measure_" + nameML + ".txt", "wb") as f:
            pickle.dump((res, listNameofDataSet), f)     
# def showGrap(y_f1_1, y_f1_2, y_f1_3):

def setGiaTriTungClass(inSubplot,y_f1 ,y_f1_2 ,y_f1_3, title):
    nameX = ['test\n' + str(x) for x in range(1,11)]
    x = [ia for ia in range(1,11)]
    inSubplot.set_xticks(x)
    inSubplot.set_xticklabels(nameX)
    inSubplot.plot(x, y_f1, label='KNN(k=100)', marker='o', linestyle='-', linewidth=1.5)
    inSubplot.plot(x, y_f1_2, label='LinearSvc', marker='s', linestyle='-.', linewidth=1.5)
    inSubplot.plot(x, y_f1_3, label='NaiveBayes', marker='v', linestyle='--', linewidth=1.5)
    inSubplot.legend(loc='best', fancybox=True, shadow=True) #hiển thị ô chú thích các điểm
    inSubplot.set_title(title, color='#000000', weight="bold", size="large")
    inSubplot.set_ylabel('F1_Score')
    inSubplot.set_xlabel('Data Sample')
    inSubplot.grid()
    return inSubplot

def showGrap(y_f1_1, y_f1_2 ,y_f1_3, listNameofDataSet):
    # y_f1_1, y_f1_2, y_f1_3 = self.graphFscore()
    
    
    fig, ((ax1, ax2, ax3), (bx1, bx2, bx3),(cx1, cx2, cx3) )  = plt.subplots(3,3) 
    ax1 = setGiaTriTungClass(ax1, y_f1_1[0], y_f1_2 ,y_f1_3, listNameofDataSet[0] )
    ax2 = setGiaTriTungClass(ax2, y_f1_1[1], y_f1_2 ,y_f1_3, listNameofDataSet[1] )
    ax3 = setGiaTriTungClass(ax3, y_f1_1[2], y_f1_2 ,y_f1_3, listNameofDataSet[2] )
    
    bx1 = setGiaTriTungClass(bx1, y_f1_1[3], y_f1_2 ,y_f1_3, listNameofDataSet[3] )
    bx2 = setGiaTriTungClass(bx2, y_f1_1[4], y_f1_2 ,y_f1_3, listNameofDataSet[4] )
    bx3 = setGiaTriTungClass(bx3, y_f1_1[5], y_f1_2 ,y_f1_3, listNameofDataSet[5] )

    cx1 = setGiaTriTungClass(cx1, y_f1_1[6], y_f1_2 ,y_f1_3, listNameofDataSet[6] )
    cx2 = setGiaTriTungClass(cx2, y_f1_1[7], y_f1_2 ,y_f1_3, listNameofDataSet[7] )
    cx3 = setGiaTriTungClass(cx3, y_f1_1[8], y_f1_2 ,y_f1_3, listNameofDataSet[8] )
    # bx.set_xticks(x)
    # bx.set_xticklabels(nameX)
    # bx.plot(x, y_f1_1[1], label='NaviveBayes', marker='o', linestyle='-', linewidth=1.5)
    # # bx.plot(x, y_f1_2[1], label='DecisionTree', marker='v', linestyle='--', linewidth=1.5)
    # # bx.plot(x, y_f1_3[1], label='LinearSvc', marker='s', linestyle='-.', linewidth=1.5)
    # bx.legend(loc='best', fancybox=True, shadow=True) #hiển thị ô chú thích các điểm
    # bx.set_title('Spam', color='#000000', weight="bold", size="large")
    # bx.set_ylabel('F1_Score')
    # bx.set_xlabel('Data Sample')
    # bx.grid()
    plt.show()



# a = [[0.9655172413793104, 0.8750000000000001, 0.923076923076923, 0.9565217391304348, 1.0, 1.0, 1.0, 0.8750000000000001, 1.0, 0.9090909090909091], [1.0, 1.0, 1.0, 0.9565217391304348, 0.9411764705882353, 1.0, 0.8571428571428571, 1.0, 0.9411764705882353, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9655172413793104, 1.0], [0.8000000000000002, 0.9166666666666666, 0.8695652173913044, 0.8750000000000001, 0.9473684210526316, 0.9090909090909091, 1.0, 0.9, 0.9333333333333333, 0.7499999999999999], [1.0, 1.0, 1.0, 0.9565217391304348, 0.9600000000000001, 1.0, 1.0, 1.0, 0.9411764705882353, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9473684210526316, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.9714285714285714, 1.0, 0.9090909090909091, 0.9523809523809523, 0.9565217391304348, 0.8333333333333333, 0.9523809523809523, 0.9333333333333333, 0.9090909090909091, 0.8750000000000001]]
# b = ['0_DoanNgocTai_16521045', '1_LeTuTuan_18521598', '2_NguyenHoangAn_18520430', '3_NguyenHongQuang_18521297', '4_NguyenNgocDien_19521359', '5_NguyenQuangHieu_18520748', '6_NguyenQuocDat__19521338', '7_PhanHoangNguyen_18521163', '8_TranDucPhiLam_19521741']
# print(type(a))
# print(type(b))

# with open("./tempLuu/f_measureKNN_100.data", "wb") as f:
# with open("./tempLuu/f_measureKNN_100.data", "wb") as f:
#     pickle.dump((a, b), f)
def loadFscore(name):
    with open("./tempLuu/" + name, "rb") as f:
        a, b = pickle.load(f)
    return a, b

Load(sys.argv[1], sys.argv[2],sys.argv[3]) 