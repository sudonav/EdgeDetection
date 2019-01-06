import cv2
import numpy as np

sobelFilterGx = []
sobelFilterGy = []

def readImage(path, gray):
    readAsGray = 0
    if(gray == False):
        readAsGray = -1
    image = cv2.imread(path, readAsGray)
    return image

def writeImage(imageName, image):
    maximum = max([max([abs(image[i][j]) for j in range(len(image[0]))]) for i in range(len(image))])
    writableImage = [[round((image[i][j]/maximum)*255) for j in range(len(image[0]))]for i in range(len(image))]
    cv2.imwrite(imageName+'.png',np.asarray(writableImage))

def generateSobelFilterGx():
    sobelFilterGx = [[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]]
    return sobelFilterGx

def generateSobelFilterGy():
    sobelFilterGy = [[-1,-2,-1],
                     [0,0,0],
                     [1,2,1]]
    return sobelFilterGy

def eliminateZeros(image):
    minimum = min([min([image[i][j] for j in range(len(image[0]))])for i in range(len(image))])
    maximum = max([max([image[i][j] for j in range(len(image[0]))])for i in range(len(image))])
    return [[round(((image[i][j] - minimum)/(maximum - minimum)),2) for j in range(len(image[0]))]for i in range(len(image))]

def normalizeZeros(image):
    maximum = max([max([abs(image[i][j]) for j in range(len(image[0]))]) for i in range(len(image))])
    return [[(abs(image[i][j])/maximum) for j in range(len(image[0]))]for i in range(len(image))]

def computeMagnitude(edgeX, edgeY):
    return (((edgeX**2) + (edgeY**2))**0.5)

def normalizeMagnitude(image):
    maxMagnitude = max([max([image[i][j] for j in range(len(image[0]))])for i in range(len(image))])
    return [[(image[i][j]/maxMagnitude) for j in range(len(image[0]))]for i in range(len(image))]

def computeIntensity(image):
    return [[sum([image[i][j][k] for k in range(len(image[i][j]))]) for j in range(len(image[0]))] for i in range(len(image))]

def generatePatch(image, x, y):
    return [[(image[i][j] if ((i >= 0 and i < len(image)) and (j >= 0 and j < len(image[0]))) else 0) for j in range(y-1, y+2, 1)] for i in range(x-1, x+2, 1)]

def convolvePatch(patch):
    convolvedPatch = [patch[j][i] for j in range(len(patch[0])-1,-1,-1) for i in range(len(patch)-1,-1,-1)]
    return [convolvedPatch[i:i+len(patch)] for i in range(0, len(convolvedPatch), len(patch))]

def generateGradientX(x, y, imageIntensity, convolve):
    patch = generatePatch(imageIntensity, x, y)
    if(convolve == True):
        convolvedPatch = convolvePatch(patch)
    else:
        convolvedPatch = patch
    sobelFilterGx = generateSobelFilterGx()
    gradientX = sum([sum([convolvedPatch[i][j] * sobelFilterGx[i][j] for j in range(len(sobelFilterGx))]) for i in range(len(sobelFilterGx))])
    return int(gradientX)

def generateGradientY(x, y, imageIntensity, convolve):
    patch = generatePatch(imageIntensity, x, y)
    if(convolve == True):
        convolvedPatch = convolvePatch(patch)
    else:
        convolvedPatch = patch
    sobelFilterGy = generateSobelFilterGy()
    gradientY = sum([sum([convolvedPatch[i][j] * sobelFilterGy[i][j] for j in range(len(sobelFilterGy))]) for i in range(len(sobelFilterGy))])
    return gradientY

def applyFilter(image, gray, imgFilter=2):
    convolve = True
    if(gray == False):
        imageIntensity = computeIntensity(image)
    else:
        imageIntensity = image
    
    outputImage = [[0 for j in range(len(image[0]))]for i in range(len(image))]

    if(imgFilter == 2): 
        for i in range(len(image)):
            for j in range(len(image[0])):
                localEdgeX = generateGradientX(i, j, imageIntensity, convolve)
                localEdgeY = generateGradientY(i, j, imageIntensity, convolve)
                outputImage[i][j] = computeMagnitude(localEdgeX,localEdgeY)
    elif(imgFilter == 0):
        for i in range(len(image)):
            for j in range(len(image[0])):
                outputImage[i][j] = generateGradientX(i, j, imageIntensity, convolve)               
    elif(imgFilter == 1):
        for i in range(len(image)):
            for j in range(len(image[0])):
                outputImage[i][j] = generateGradientY(i, j, imageIntensity, convolve)
    
    if(imgFilter == 2):
        outputImage = normalizeMagnitude(outputImage)
    elif(imgFilter == 0 or imgFilter == 1):
        outputImage = normalizeZeros(outputImage)
        
    return np.asarray(outputImage)

gray = True
image = readImage('task1.png', gray)
cv2.imshow('Original',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
outputImage = applyFilter(image, gray)
writeImage("XY_Edges",outputImage)
outputImage = applyFilter(image, gray, 0)
writeImage("X_Edge",outputImage)
outputImage = applyFilter(image, gray, 1)
writeImage("Y_Edge",outputImage)

