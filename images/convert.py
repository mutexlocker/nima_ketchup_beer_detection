import os, fnmatch
import xml.etree.ElementTree
import cv2
import numpy as np
import random
import imutils


files = fnmatch.filter(os.listdir('train'), '*.jpg')
total_size = (len(files)*11)
print(total_size)

random_file_name = random.sample(range(1,100000000), 900000)
print(random_file_name)


foldername= 'train/'
foldername_export= 'train_crop/'

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

gamma_values =np.array([0.4,0.7,1.5,2.3])
print(gamma_values)

def CLAHE(image,clipLimit=10.0,gridsize=5):
   lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
   lab_planes = cv2.split(lab)
   clahe = cv2.createCLAHE(clipLimit,tileGridSize=(gridsize,gridsize))
   lab_planes[0] = clahe.apply(lab_planes[0])
   lab = cv2.merge(lab_planes)
   bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
   return bgr



# #CLAHE (Contrast Limited Adaptive Histogram Equalization) Example
# img = cv2.imread(foldername+files[0],1)
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# lab_planes = cv2.split(lab)
# clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(10,10))
# lab_planes[0] = clahe.apply(lab_planes[0])
# lab = cv2.merge(lab_planes)
# bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#
# cv2.imshow('image',bgr)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# rotate_valuse=np.array([30,60,90,120,150,180,210,240,270,300,330])
# ##########################Rotate images
# print(files[0])
# img = cv2.imread(foldername+files[0],1)
# xmlfile = files[0].replace('.jpg', '') + '.xml.xml'
# et = xml.etree.ElementTree.parse(foldername + xmlfile)
# root = et.getroot()
# print(len(root.findall('object')))
#
#
# for objects in root.findall('object'):
#    print (img.shape)
#    rows, cols,c = img.shape
#    # for pos in objects:
#    #   rank = pos.find('name').text
#    #   name = pos.get('xmin')
#    xmin = int(objects[4][0].text)
#    ymin = int(objects[4][1].text)
#    xmax = int(objects[4][2].text)
#    ymax = int(objects[4][3].text)
#    theta = 90
#    A1 = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#    B1 = [[xmin],[ymin]]
#    AB1 = abs(np.dot(A1,B1))
#
#    A2 = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#    B2 = [[xmax],[ymax]]
#    AB2 = abs(np.dot(A2,B2))
#
#    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
#    print("Rotation MAtrix\n",M)
#    print('first point=',xmin,ymin)
#    print('translated to= ',AB1)
#
#    print('second point=',xmax, ymax)
#    print('translated to= ',AB2)
#
#    rotated = cv2.warpAffine(img, M, (cols, rows))
#    # cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
#    #rotated = imutils.rotate_bound(img, 30)
#    ap = cv2.rectangle(rotated, (AB1[0], AB1[1]), (AB2[0], AB2[1]), (0, 0, 255), 3)
#    ap = cv2.rectangle(ap, (xmin, ymin), (xmax, ymax), (255, 0, 255), 3)
#
# # cv2.imshow('image',ap)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# #
# ########################## END Rotate images
#




#
# ##########################Affine transform
# for f in files:
#     jpgfile = f
#     xmlfile = f.replace('.jpg', '') + '.xml.xml'
#     x = foldername + jpgfile  # location of the image
#     original = cv2.imread(x, 1)
#     et = xml.etree.ElementTree.parse(foldername + xmlfile)
#     root = et.getroot()
#     count = 1
#     for x in np.nditer(gamma_values):
#         gamma = x                                 # change the value here to get different result
#         adjusted = adjust_gamma(original, gamma=gamma)
#         #cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
#         #cv2.imshow("gammam image 1", adjusted)
#         xml_savepath = xmlfile.replace('.xml.xml', '') + "_" + str(count) + '.xml.xml'
#         root[1].text = jpgfile.replace('.jpg','') + "_" + str(count) + '.jpg'
#         root[2].text = root[2].text.replace('.jpg','') + "_" + str(count) + '.jpg'
#         print(xml_savepath)
#         et.write(foldername + xml_savepath)
#         image_savepath = jpgfile.replace('.jpg','') + "_" + str(count) + '.jpg'
#         print(image_savepath)
#         cv2.imwrite(foldername + image_savepath, adjusted)
#         count = count + 1
#
#
#
# et = xml.etree.ElementTree.parse('train/'+xmlfile)
# root = et.getroot()
# print(root[1].text)
# print(root[2].text)
# root[1].text = jpgfile.replace('.jpg','')  + '_1' + '.jpg'
# root[2].text = root[2].text.replace('.jpg','') + '_1'+ '.jpg'
# et.write('output.xml')
#
# ########################## END Rotate images






counter = 0


# # ##Read images and XML and generate croped images in new directory
random_crop = 1

# for filename in files:
#     #print (filename)
#     img = cv2.imread(foldername+filename,1)
#     xmlfile = filename.replace('.jpg', '') + '.xml.xml'
#     et = xml.etree.ElementTree.parse(foldername + xmlfile)
#     et_2 = xml.etree.ElementTree.parse(foldername + xmlfile)
#     root = et.getroot()
#     root_copy = et_2.getroot()
#     object_size = len(root_copy.findall('object'))
#     for i in range(0,object_size-1):
#         root_copy.remove(root_copy.findall('object')[i])
#     for objects in root.findall('object'):
#        rows, cols,c = img.shape
#        xmin = int(objects[4][0].text)
#        ymin = int(objects[4][1].text)
#        xmax = int(objects[4][2].text)
#        ymax = int(objects[4][3].text)
#        crop_img = img[ymin:ymax, xmin:xmax]
#        rows_crop, cols_crop, c = crop_img.shape
#        if random_crop == 0:
#             save_path = foldername_export+filename.replace('.jpg', '') + '_' +str(counter)+'.jpg'
#        else:
#            save_path =  foldername_export+str(random_file_name[counter]) + '.jpg'
#            print(counter)
#        print(save_path)
#        cv2.imwrite(save_path, crop_img)
#        if random_crop == 0:
#            xml_savepath = foldername_export + xmlfile.replace('.xml.xml', '') + "_" + str(counter) + '.xml.xml'
#            root_copy[1].text = filename.replace('.jpg','') + "_" + str(counter) + '.jpg'
#            root_copy[2].text = root[2].text.replace('.jpg','') + "_" + str(counter) + '.jpg'
#        else:
#            xml_savepath = foldername_export+ str(random_file_name[counter]) + '.xml.xml'
#            root_copy[1].text = str(random_file_name[counter]) + '.jpg'
#            root_copy[2].text = '/home/sina/Desktop/nima-bottle-ketchap/images/' + str(random_file_name[counter]) + '.jpg'
#
#
#        label = objects[0].text
#
#        size_copy = (root_copy.findall('size')[0])
#        size_copy[0].text = str(cols_crop)
#        size_copy[1].text = str(rows_crop)
#
#
#        object_copy = (root_copy.findall('object')[0])
#        object_copy[0].text = label
#        object_copy[4][0].text = str(1)
#        object_copy[4][1].text = str(1)
#        object_copy[4][2].text = str(cols_crop-1)
#        object_copy[4][3].text = str(rows_crop-1)
#        print(xml_savepath)
#        #root.remove(objects)
#        et_2.write(xml_savepath)
#        counter = counter + 1
# # # # # ########################## END  crop image


# # ### Image resizer
# random_crop = 1
# size_array = np.array([0.5,1.5,2])
# for filename in files:
#     #print (filename)
#     img = cv2.imread(foldername+filename,1)
#     xmlfile = filename.replace('.jpg', '') + '.xml.xml'
#     et = xml.etree.ElementTree.parse(foldername + xmlfile)
#     root = et.getroot()
#     for sizes in size_array:
#        objects =  root.findall('object')[0]
#        rows, cols,c = img.shape
#        resized_img = cv2.resize(img, (0,0), fx=sizes, fy=sizes)
#        rows_resized, cols_resized, c = resized_img.shape
#        if random_crop == 0:
#             save_path = foldername_export+filename.replace('.jpg', '') + '_' +str(counter)+'.jpg'
#        else:
#            save_path =  foldername_export+str(random_file_name[counter]) + '.jpg'
#
#        print(counter)
#        print(save_path)
#        cv2.imwrite(save_path, resized_img)
#        if random_crop == 0:
#            xml_savepath = foldername_export + xmlfile.replace('.xml.xml', '') + "_" + str(counter) + '.xml.xml'
#            root[1].text = filename.replace('.jpg','') + "_" + str(counter) + '.jpg'
#            root[2].text = root[2].text.replace('.jpg','') + "_" + str(counter) + '.jpg'
#        else:
#            xml_savepath = foldername_export+ str(random_file_name[counter]) + '.xml.xml'
#            root[1].text = str(random_file_name[counter]) + '.jpg'
#            root[2].text = '/home/sina/Desktop/nima-bottle-ketchap/images/' + str(random_file_name[counter]) + '.jpg'
#
#        size_copy = (root.findall('size')[0])
#        size_copy[0].text = str(cols_resized)
#        size_copy[1].text = str(rows_resized)
#
#        objects[4][0].text = str(1)
#        objects[4][1].text = str(1)
#        objects[4][2].text = str(cols_resized-1)
#        objects[4][3].text = str(rows_resized-1)
#        print(xml_savepath)
#        #root.remove(objects)
#        et.write(xml_savepath)
#        counter = counter + 1
#
#
#
# # ################






# # Image Rotator
# random_rotate = 1
# degress_array = np.array([45,90,135,180,240,270,340])
# for filename in files:
#     #print (filename)
#     img = cv2.imread(foldername+filename,1)
#     xmlfile = filename.replace('.jpg', '') + '.xml.xml'
#     et = xml.etree.ElementTree.parse(foldername + xmlfile)
#     root = et.getroot()
#     for degrees in degress_array:
#        objects =  root.findall('object')[0]
#        rows, cols,c = img.shape
#        rotated_img =  imutils.rotate_bound(img, degrees)
#        rows_rotated, cols_rotated, c = rotated_img.shape
#        if random_rotate == 0:
#             save_path = foldername_export+filename.replace('.jpg', '') + '_' +str(counter)+'.jpg'
#        else:
#            save_path =  foldername_export+str(random_file_name[counter]) + '.jpg'
#
#        print(counter)
#        print(save_path)
#        cv2.imwrite(save_path, rotated_img)
#        if random_rotate == 0:
#            xml_savepath = foldername_export + xmlfile.replace('.xml.xml', '') + "_" + str(counter) + '.xml.xml'
#            root[1].text = filename.replace('.jpg','') + "_" + str(counter) + '.jpg'
#            root[2].text = root[2].text.replace('.jpg','') + "_" + str(counter) + '.jpg'
#        else:
#            xml_savepath = foldername_export+ str(random_file_name[counter]) + '.xml.xml'
#            root[1].text = str(random_file_name[counter]) + '.jpg'
#            root[2].text = '/home/sina/Desktop/nima-bottle-ketchap/images/' + str(random_file_name[counter]) + '.jpg'
#
#        size_copy = (root.findall('size')[0])
#        size_copy[0].text = str(cols_rotated)
#        size_copy[1].text = str(rows_rotated)
#
#        objects[4][0].text = str(1)
#        objects[4][1].text = str(1)
#        objects[4][2].text = str(cols_rotated-1)
#        objects[4][3].text = str(rows_rotated-1)
#        print(xml_savepath)
#        #root.remove(objects)
#        et.write(xml_savepath)
#        counter = counter + 1
#


################














#uncoomemnt this if you want random file##3
count = 0
for f in files:
    jpgfile = f
    xmlfile = f.replace('.jpg', '') + '.xml.xml'
    x = foldername + jpgfile  # location of the image
    original = cv2.imread(x, 1)
    et = xml.etree.ElementTree.parse(foldername + xmlfile)
    root = et.getroot()

    for x in np.nditer(gamma_values):
        gamma = x                                 # change the value here to get different result
        adjusted = adjust_gamma(original, gamma=gamma)
        #cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        #cv2.imshow("gammam image 1", adjusted)
        xml_savepath = str(random_file_name[count])+ '.xml.xml'
        root[1].text = str(random_file_name[count]) + '.jpg'
        root[2].text = '/home/sina/Desktop/nima-bottle-ketchap/images/'+ str(random_file_name[count]) + '.jpg'
        print(xml_savepath)
        et.write(foldername_export + xml_savepath)
        image_savepath = str(random_file_name[count])+'.jpg'
        print(image_savepath)
        cv2.imwrite(foldername_export + image_savepath, adjusted)
        count = count + 1




##uncoomemnt this if you want clean file names with order!##3


# for f in files:
#     jpgfile = f
#     xmlfile = f.replace('.jpg', '') + '.xml.xml'
#     x = foldername + jpgfile  # location of the image
#     original = cv2.imread(x, 1)
#     et = xml.etree.ElementTree.parse(foldername + xmlfile)
#     root = et.getroot()
#     count = 1
#     for x in np.nditer(gamma_values):
#         print(x)
#         adjusted = adjust_gamma(original, gamma=x)
#         #cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
#         #cv2.imshow("gammam image 1", adjusted)
#         xml_savepath = xmlfile.replace('.xml.xml', '') + "_" + str(count) + '.xml.xml'
#         root[1].text = jpgfile.replace('.jpg','') + "_" + str(count) + '.jpg'
#         root[2].text = root[2].text.replace('.jpg','') + "_" + str(count) + '.jpg'
#         print(xml_savepath)
#         et.write(foldername_export + xml_savepath)
#         image_savepath = jpgfile.replace('.jpg','') + "_" + str(count) + '.jpg'
#         print(image_savepath)
#         cv2.imwrite(foldername_export + image_savepath, adjusted)
#         count = count + 1
#


# et = xml.etree.ElementTree.parse('train/'+xmlfile)
# root = et.getroot()
# print(root[1].text)
# print(root[2].text)
# root[1].text = jpgfile.replace('.jpg','')  + '_1' + '.jpg'
# root[2].text = root[2].text.replace('.jpg','') + '_1'+ '.jpg'
# et.write('output.xml')
#
#




## apply CLAHE with non random image inedexing
# cliplimit_values = np.linspace(1, 11, num=10)
# cliplimit_values = np.around(cliplimit_values)
# print(cliplimit_values)
# for f in files:
#     jpgfile = f
#     xmlfile = f.replace('.jpg', '') + '.xml.xml'
#     x = foldername + jpgfile  # location of the image
#     original = cv2.imread(x, 1)
#     et = xml.etree.ElementTree.parse(foldername + xmlfile)
#     root = et.getroot()
#     count = 1
#     for x in np.nditer(cliplimit_values):
#         cliplimit = x                                 # change the value here to get different result
#         adjusted = CLAHE(original,gridsize=cliplimit)
#         #cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
#         #cv2.imshow("gammam image 1", adjusted)
#         xml_savepath = xmlfile.replace('.xml.xml', '') + "_" + str(count) + '.xml.xml'
#         root[1].text = jpgfile.replace('.jpg','') + "_" + str(count) + '.jpg'
#         root[2].text = root[2].text.replace('.jpg','') + "_" + str(count) + '.jpg'
#         print(xml_savepath)
#         et.write(foldername + xml_savepath)
#         image_savepath = jpgfile.replace('.jpg','') + "_" + str(count) + '.jpg'
#         print(image_savepath)
#         cv2.imwrite(foldername + image_savepath, adjusted)
#         count = count + 1
#
#
#
# et = xml.etree.ElementTree.parse('train/'+xmlfile)
# root = et.getroot()
# print(root[1].text)
# print(root[2].text)
# root[1].text = jpgfile.replace('.jpg','')  + '_1' + '.jpg'
# root[2].text = root[2].text.replace('.jpg','') + '_1'+ '.jpg'
# et.write('output.xml')



# ## apply CLAHE with non random image inedexing
# cliplimit_values = np.linspace(1, 11, num=10)
# cliplimit_values = np.around(cliplimit_values)
# print(cliplimit_values)
# for f in files:
#     jpgfile = f
#     xmlfile = f.replace('.jpg', '') + '.xml.xml'
#     x = foldername + jpgfile  # location of the image
#     original = cv2.imread(x, 1)
#     et = xml.etree.ElementTree.parse(foldername + xmlfile)
#     root = et.getroot()
#     count = 1
#     for x in np.nditer(cliplimit_values):
#         cliplimit = x                                 # change the value here to get different result
#         adjusted = CLAHE(original,gridsize=cliplimit)
#         #cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
#         #cv2.imshow("gammam image 1", adjusted)
#         xml_savepath = xmlfile.replace('.xml.xml', '') + "_" + str(count) + '.xml.xml'
#         root[1].text = jpgfile.replace('.jpg','') + "_" + str(count) + '.jpg'
#         root[2].text = root[2].text.replace('.jpg','') + "_" + str(count) + '.jpg'
#         print(xml_savepath)
#         et.write(foldername + xml_savepath)
#         image_savepath = jpgfile.replace('.jpg','') + "_" + str(count) + '.jpg'
#         print(image_savepath)
#         cv2.imwrite(foldername + image_savepath, adjusted)
#         count = count + 1
#
#
#
# et = xml.etree.ElementTree.parse('train/'+xmlfile)
# root = et.getroot()
# print(root[1].text)
# print(root[2].text)
# root[1].text = jpgfile.replace('.jpg','')  + '_1' + '.jpg'
# root[2].text = root[2].text.replace('.jpg','') + '_1'+ '.jpg'
# et.write('output.xml')