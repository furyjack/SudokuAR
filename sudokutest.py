import cv2
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier


class Classifier:

    '''
    This class deals with the machine learning part of the problem namely optical character recognition

    '''


    def __init__(self):
        samples = np.loadtxt('data/imagedata.txt', np.float32)
        responses = np.loadtxt('data/labels.txt', np.float32)
        # Training data used for the KNN classifier
        self.model = KNeighborsClassifier(n_neighbors=2)
        self.model.fit(samples, responses)
        #Tells us the no of iteration of a particular morhology to be used
        self.morphs = [-1, 0, 1, 2]
        self.lvl = 0  # index of morphs

    def ocr(self, image):
        # preprocessing for OCR
        # convert image to grayscale
        gray = cv2.cvtColor(image.output, cv2.COLOR_BGR2GRAY)
        # noise removal with gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        image.outputGray = gray
        # attempt to read the image with 4 different morphology values and find the best result
        self.success = [0, 0, 0, 0]
        self.errors = [0, 0, 0, 0]
        for self.lvl in self.morphs:
            image.output = np.copy(image.outputBackup)
            self.ocr_read(image)

        best = 103 #some random value
        for i in range(4):
            if self.success[i] > best and self.errors[i] >= 0:
                best = self.success[i]
                ibest = i

        if best == 103:
            print "ERROR - OCR FAILURE"
            return None, None

        else:
            print "final morph erode iterations:", self.morphs[ibest]
            image.output = np.copy(image.outputBackup)
            self.lvl = self.morphs[ibest]
            arr = self.ocr_read(image)
            cv2.imshow('output', image.output)
            key = cv2.waitKey(5)
            return arr, key

    def ocr_read(self, image):
        # perform actual OCR using kNearest model
        thresh = cv2.adaptiveThreshold(image.outputGray, 255, 1, 1, 7, 2)
        morph = None
        if self.lvl >= 0:
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None, iterations=self.lvl)

        elif self.lvl == -1:
            morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, None, iterations=1)

        thresh_copy = morph.copy()
        # thresh2 changes after findContours
        _, contours, hierarchy = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        thresh = thresh_copy

        #final puzzle states
        current = np.zeros((9, 9), np.uint8)
        numeric_array = np.zeros((9, 9), np.uint8)

        # testing section
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h > 20 and h < 40 and w > 8 and w < 40:
                    if w < 20:
                        diff = 20 - w
                        x -= diff / 2
                        w += diff
                    sudox = x / 50
                    sudoy = y / 50
                    cv2.rectangle(image.output, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 2)
                    # prepare region of interest for OCR kNearest model
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, (25, 35))
                    roismall = roismall.reshape((1, 875))
                    roismall = np.float32(roismall)
                    # find result
                    label = self.model.predict(roismall)
                    # check for read errors
                    if label[0] != 0:
                        string = str(int(label[0]))
                        if current[sudoy, sudox] == 0:
                            current[sudoy, sudox] = int(string)
                            numeric_array[sudoy, sudox] = int(string)
                        else:
                            self.errors[self.lvl + 1] = -2  # double read error
                        self.success[self.lvl + 1] += 1
                        cv2.putText(image.output, string, (x, y + h), 0, 1.4, (255, 0, 0), 3)
                    else:
                        self.errors[self.lvl + 1] = 0  # read zero error
        return numeric_array


class imageClass:
    # this class defines all of the important image matrices, and information about the images.
    # also the methods associated with capturing input, displaying the output,
    # and warping and transforming any of the images to assist with OCR
    def __init__(self):
        # .captured is the initially captured image
        self.captured = []
        # .gray is the grayscale captured image
        self.gray = []
        # .thres is after adaptive thresholding is applied
        self.thresh = []
        # .contours contains information about the contours found in the image
        self.contours = []
        # .biggest contains a set of four coordinate points describing the
        # contours of the biggest rectangular contour found
        self.biggest = None
        # .maxArea is the area of this biggest rectangular found
        self.maxArea = 0
        # .output is an image resulting from the warp() method
        self.output = []
        self.outputBackup = []
        self.outputGray = []
        # .mat is a matrix of 100 points found using a simple gridding algorithm
        # based on the four corner points from .biggest
        self.mat = np.zeros((100, 2), np.float32)
        # .reshape is a reshaping of .mat
        self.reshape = np.zeros((100, 2), np.float32)

    def captureImage(self, IMG):
        # captures the image and finds the biggest rectangle
        print "Loading sudoku.jpg..."
        # for testing purposes
        img = IMG

        self.captured = cv2.resize(img, (600, 600))

        # convert to grayscale
        self.gray = cv2.cvtColor(self.captured, cv2.COLOR_BGR2GRAY)

        # noise removal with gaussian blur
        self.gray = cv2.GaussianBlur(self.gray, (5, 5), 0)
        # then do adaptive thresholding

        self.thresh = cv2.adaptiveThreshold(self.gray, 255, 1, 1, 11, 2)

        # find countours in threshold image
        _, self.contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # evaluate all blobs to find blob with biggest area
        # biggest rectangle in the image must be sudoku square
        self.biggest = None
        self.maxArea = 0
        for i in self.contours:
            area = cv2.contourArea(i)
            if area > 50000:  # 50000 is an estimated value for the kind of blob we want to evaluate
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > self.maxArea and len(approx) == 4:
                    self.biggest = approx
                    self.maxArea = area
                    best_cont = i
        if self.maxArea > 0:

            cv2.polylines(self.captured, [self.biggest], True, (0, 0, 255), 3)
            self.reorder()
            h = np.array([[0, 0], [599, 0], [599, 599], [0, 599]], np.float32)
            #
            retval = cv2.getPerspectiveTransform(self.biggest, h)
            self.captured = cv2.warpPerspective(self.captured, retval, (600, 600))

            self.gray = cv2.cvtColor(self.captured, cv2.COLOR_BGR2GRAY)

            # noise removal with gaussian blur
            self.gray = cv2.GaussianBlur(self.gray, (5, 5), 0)
            # then do adaptive thresholding
            self.thresh = cv2.adaptiveThreshold(self.gray, 255, 1, 1, 11, 2)

            self.biggest = h

            # reorder self.biggest
            return True

        else:
            print "error in reading puzzle"
            return False

    def reorder(self):
        # reorders the points obtained from finding the biggest rectangle
        # [top-left, top-right, bottom-right, bottom-left]
        a = self.biggest.reshape((4, 2))
        b = np.zeros((4, 2), dtype=np.float32)

        add = a.sum(1)
        b[0] = a[np.argmin(add)]  # smallest sum
        b[2] = a[np.argmax(add)]  # largest sum

        diff = np.diff(a, axis=1)  # y-x
        b[1] = a[np.argmin(diff)]  # min diff
        b[3] = a[np.argmax(diff)]  # max diff
        self.biggest = b

    def perspective(self):
        # create 100 points using "biggest" and simple gridding algorithm,
        # these 100 points define the grid of the sudoku puzzle
        # topLeft-topRight-bottomRight-bottomLeft = "biggest"
        b = np.zeros((100, 2), dtype=np.float32)
        c_sqrt = 10
        # if self.biggest == None:
        #     self.biggest = [[0,0],[640,0],[640,480],[0,480]]
        tl, tr, br, bl = self.biggest[0], self.biggest[1], self.biggest[2], self.biggest[3]

        for k in range(0, 100):
            i = k % c_sqrt
            j = k / c_sqrt
            ml = [tl[0] + (bl[0] - tl[0]) / 9 * j, tl[1] + (bl[1] - tl[1]) / 9 * j]
            mr = [tr[0] + (br[0] - tr[0]) / 9 * j, tr[1] + (br[1] - tr[1]) / 9 * j]
            self.mat.itemset((k, 0), ml[0] + (mr[0] - ml[0]) / 9 * i)
            self.mat.itemset((k, 1), ml[1] + (mr[1] - ml[1]) / 9 * i)
        self.reshape = self.mat.reshape((c_sqrt, c_sqrt, 2))

        for i in range(10):
            for j in range(10):
                cv2.circle(self.captured, (self.reshape[i][j][0], self.reshape[i][j][1]), 3, (0, 255, 0), -1)

        cv2.imshow('ff', self.captured)

    def warp(self):
        # take distorted image and warp to flat square for clear OCR reading
        mask = np.zeros((self.gray.shape), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        close = cv2.morphologyEx(self.gray, cv2.MORPH_CLOSE, kernel)

        division = np.float32(self.gray) / (close)

        result = np.uint8(cv2.normalize(division, division, 0, 255, cv2.NORM_MINMAX))
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        output = np.zeros((450, 450, 3), np.uint8)
        c_sqrt = 10
        for i, j in enumerate(self.mat):
            ri = i / c_sqrt
            ci = i % c_sqrt
            if ci != c_sqrt - 1 and ri != c_sqrt - 1:
                source = self.reshape[ri:ri + 2, ci:ci + 2, :].reshape((4, 2))
                dest = np.array([[ci * 450 / (c_sqrt - 1), ri * 450 / (c_sqrt - 1)], [(ci + 1) * 450 / (c_sqrt - 1),
                                                                                      ri * 450 / (c_sqrt - 1)],
                                 [ci * 450 / (c_sqrt - 1), (ri + 1) * 450 / (c_sqrt - 1)],
                                 [(ci + 1) * 450 / (c_sqrt - 1), (ri + 1) * 450 / (c_sqrt - 1)]], np.float32)
                trans = cv2.getPerspectiveTransform(source, dest)
                warp = cv2.warpPerspective(result, trans, (450, 450))


                #correct from source
                temp = warp[i * 450 / (c_sqrt - 1):(ri + 1) * 450 / (c_sqrt - 1),
                       ci * 450 / (c_sqrt - 1):(ci + 1) * 450 / (c_sqrt - 1)].copy()
                output[ri * 450 / (c_sqrt - 1):(ri + 1) * 450 / (c_sqrt - 1),
                ci * 450 / (c_sqrt - 1):(ci + 1) * 450 / (c_sqrt - 1)] = temp

        output_backup = np.copy(output)

        self.output = output
        self.outputBackup = output_backup

    def virtualImage(self, puzzle, marker):
        # output known sudoku values to the real image
        j = 0
        tsize = (math.sqrt(self.maxArea)) / 400
        w = int(20 * tsize)
        h = int(25 * tsize)
        for i in range(100):
            x = int(self.mat.item(i, 0) + 8 * tsize)
            y = int(self.mat.item(i, 1) + 8 * tsize)
            if i % 10 != 9 and i / 10 != 9:
                yc = j % 9
                xc = j / 9
                j += 1
                if marker[xc, yc] == 0:
                    string = str(puzzle[xc, yc])
                    cv2.putText(self.captured, string, (x + w / 4, y + h), 0, tsize, (0, 255, 0), 2)
                    cv2.imshow('sudoku', self.captured)
                    key = cv2.waitKey(20)
        cv2.polylines(self.captured, np.int32([self.biggest]), True, (0, 255, 0), 3)
        cv2.imshow('sudoku', self.captured)
        cv2.waitKey(0)


def capture():
    cap = cv2.VideoCapture(0)
    image = imageClass()
    keep = True
    while keep:
        ret, frame = cap.read()
        keep = image.captureImage(frame)
        keep = not keep
        cv2.imshow('vf', image.captured)
        cv2.waitKey(10)
    del cap
    return image


if __name__ == '__main__':

    image = imageClass()
    image.captureImage(cv2.imread('images/j2.jpg'))

    image.perspective()
    image.warp()

    reader = Classifier()
    arr, key = reader.ocr(image)

    print 'Are there any mistakes?'
    print 'hit y for yes and n for no!'
    x = raw_input('')
    if x == 'y':
        nerr = raw_input('Enter how many mistakes')
        nerr = int(nerr)
        for k in range(nerr):
            print 'enter row, column and number of error cell'
            r = raw_input('')
            c = raw_input('')
            num = raw_input('')
            arr[int(r)][int(c)] = int(num)

    print arr

    from puzzlesolver import Puzzle

    marker = np.array(arr)
    puz = Puzzle(arr)
    puz.solve()

    print puz.X
    image.virtualImage(puz.X, marker)
