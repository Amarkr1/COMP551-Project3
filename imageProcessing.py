import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import pickle
from random import randrange
from scipy import ndimage


def image_seg(file, showplot=False, saveimage=False, toCSV=False, set_dimensions=(64,64), threshold=230):
    """Cleans then segments the images"""
    with open(file) as csvfile:
        count = 0
        csv = []
        for line in csvfile:
            array = line.split(",")
            matrix = np.array(array).astype("int")
            picture = []
            for pixle in matrix:
                if pixle > threshold:
                    pixle = 2
                elif pixle > 200:
                    pixle = 1
                else:
                    pixle = 0
                picture.append(pixle)
            # csv.append(picture)
            picture = np.array(picture)
            picture = picture.reshape(set_dimensions)
            labels, numobjects = ndimage.label(picture, structure=[[1,1,1],
                                                                   [1,1,1],
                                                                   [1,1,1]])
            foundObjects = ndimage.find_objects(labels)
            objectDict = {}
            i = 0
            for item in foundObjects:
                size = abs(item[0].start - item[0].stop) * abs(item[1].start - item[1].stop)
                if size > 20:
                    objectDict[i] = size
                i += 1

            pickList = sorted(objectDict, key=objectDict.__getitem__)[:-4:-1]
            threeObjects = []
            # print("Number Frame Co-ordinates")
            for index in pickList:
                # print(foundObjects[index])
                threeObjects.append(foundObjects[index])

            while len(threeObjects) < 3:
                amalgam = threeObjects.pop(0)
                height = abs(amalgam[0].start - amalgam[0].stop)
                width = abs(amalgam[1].start - amalgam[1].stop)
                if height > width:
                    cutPoint = int((amalgam[0].start + amalgam[0].stop)/2)
                    first = (slice(amalgam[0].start, cutPoint,None), slice(amalgam[1].start, amalgam[1].stop, None))
                    second = (slice(cutPoint, amalgam[0].stop,None), slice(amalgam[1].start, amalgam[1].stop, None))
                else:
                    cutPoint = int((amalgam[1].start + amalgam[1].stop) / 2)
                    first = (slice(amalgam[0].start, amalgam[0].stop, None), slice(amalgam[1].start, cutPoint,None))
                    second = (slice(amalgam[0].start, amalgam[0].stop, None), slice(cutPoint, amalgam[1].stop,None))

                threeObjects.append(first)
                threeObjects.append(second)

            finalFrame = []
            for objectSlice in threeObjects:
                centroid = (int((objectSlice[0].start + objectSlice[0].stop)/2), int((objectSlice[1].start + objectSlice[1].stop)/2))
                # print(centroid)
                y_axis = [centroid[0] - 14, centroid[0] + 14]
                if y_axis[0] < 0:
                    y_axis[1] += abs(y_axis[0])
                    y_axis[0] += abs(y_axis[0])
                elif y_axis[1] > 64:
                    y_axis[0] -= (y_axis[1] - 64)
                    y_axis[1] -= (y_axis[1] - 64)
                x_axis = [centroid[1] - 14, centroid[1] + 14]
                if x_axis[0] < 0:
                    x_axis[1] += abs(x_axis[0])
                    x_axis[0] += abs(x_axis[0])
                elif x_axis[1] > 64:
                    x_axis[0] -= (x_axis[1] - 64)
                    x_axis[1] -= (x_axis[1] - 64)
                finalFrame.append((slice(y_axis[0],y_axis[1],None), slice(x_axis[0],x_axis[1],None)))
            # print(finalFrame)

            if saveimage is True:
                for imageFrame in finalFrame:

                    # if len(finalFrame) < 3:
                    #     fig, ax = plt.subplots()
                    #     ax.imshow(labels)
                    #     plt.show()
                    #     If you want the original images, uncomment the next 2 lines and recomment the 2 after that
                    #     dirtyPic = matrix.reshape(64,64)
                    #     dirtyPic[imageFrame].reshape(-1,)
                    ID = [count]
                    features = labels[imageFrame].reshape(-1,)
                    for element in features:
                        if element == 0:
                            ID.append(0)
                        else:
                            ID.append(255)
                    csv.append(ID)

            if showplot is True:
                # fig, ax = plt.subplots()
                # ax.imshow(labels)
                # ax.set_title('Labeled objects')
                fig, axes = plt.subplots(ncols=3)
                for ax, finalSlice in zip(axes.flat, finalFrame):
                    ax.imshow(labels[finalSlice], vmin=0, vmax=numobjects)
                plt.show()
            count += 1
        if toCSV is True:
            print("Finished cleaning. Saving file to CSV. \nMay take several minutes if file is large")
            my_df = pd.DataFrame(csv)
            my_df.to_csv('dirtySegmentedImages.csv', encoding='utf8', index=False, header=False)


def gen_images(num_images, letters, numbers):

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 20,
               21, 24, 25, 27, 28, 30, 32, 35, 36,
               40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

    letter_images = np.loadtxt(letters, delimiter=',', skiprows=1)
    number_images = np.loadtxt(numbers, delimiter=',', skiprows=1)
    pickle.dump(letter_images, open("letter_images.pkl", 'wb'))
    pickle.dump(number_images, open("number_images.pkl", 'wb'))
    letter_images = pickle.load(open("letter_images.pkl", 'rb'))
    number_images = pickle.load(open("number_images.pkl", 'rb'))
    newPictures = []
    for i in range(num_images):
        canvas = np.zeros((64, 64), int)
        alphanum = []
        letter = (letter_images[np.random.randint(0, len(letter_images)-1)])
        label = [letter[0].astype(int)]
        alphanum.append(np.rot90(np.array(letter[1:], int).reshape(28, 28), 3))
        for j in range(2):
            number = number_images[np.random.randint(0, len(number_images)-1)]
            label.append(number[0])
            alphanum.append(number[1:])
        if 11 in label:
            label.remove(11)
            classification = label[0] * label[1]
            if classification not in classes:
                i -= 1
                break

        elif 10 in label:
            label.remove(10)
            classification = label[0] + label[1]
            if classification not in classes:
                i -= 1
                break

        np.random.shuffle(alphanum)
        num_of_images = 0
        for image in alphanum:
            count = 0
            while count < 30:
                startingPoint = (np.random.randint(0, 40), np.random.randint(0, 40))
                window = (slice(startingPoint[0], startingPoint[0] + 24, None),
                          slice(startingPoint[1], startingPoint[1] + 24, None))
                if np.count_nonzero(canvas[window]) < 10:
                    image = np.reshape(image, (28, 28))
                    image = ndimage.rotate(image, angle=randrange(-45, 45, 5))
                    image = image[slice(int(image.shape[0]/2)-12, int(image.shape[0]/2)+12, None),
                                  slice(int(image.shape[1]/2)-12, int(image.shape[1]/2)+12, None)]
                    canvas[window] = image
                    num_of_images += 1
                    break

                else:
                    count += 1

        if num_of_images == 3:
            # plt.imshow(canvas)
            # plt.show()
            output = list(canvas.ravel())
            newPictures.append([classification, output])
            # print(label)
            # print(classification)
        else:
            i -= 1

        if i % 50 == 0:
            print(i*100/num_images)

        if len(newPictures) > 50000:
            writer = open('GeneratedData.csv', 'a')
            for labels, row in newPictures:
                writer.write(str(labels) + ',' + str(row).replace('[', "").replace(']', "") + "\n")
            writer.close()
            newPictures = []


    writer = open('GeneratedData.csv', 'a')
    for labels, row in newPictures:
        writer.write(str(labels) + ',' + str(row).replace('[', "").replace(']', "") + "\n")
    writer.close()


def plot_y_distro(labelFile):
    with open(labelFile, encoding='utf8') as f:
        fileList = []
        for i in f:
            fileList.append(int(i.strip("\n")))

    labelStats = {}
    for label in fileList:
        if label in labelStats.keys():
            labelStats[label] += 1
        else:
            labelStats[label] = 0

    print(sorted(labelStats.items()))

    X = list(labelStats.keys())
    y = list(labelStats.values())
    plt.title("Training Data")
    plt.ylabel("Number of Occurences")
    plt.xlabel("Label")
    plt.grid(True)
    plt.bar(X, y, color="blue")
    plt.show()


start = time.time()
plot_y_distro('train_y.csv')
# image_seg('test_x.csv', saveimage=True, showplot=True)
gen_images(10000, 'train_digits.csv', 'train_letters.csv')
print("Finished after: {}".format((time.time() - start) / 60))
