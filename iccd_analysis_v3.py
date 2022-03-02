import spe2pyCustom as spe
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from os.path import join
import imageio
from pandas import DataFrame, read_excel, read_csv
from PIL import Image, ImageDraw, ImageFont
import tifffile as tf


def set_style():
    plt.close("all")
    sns.set_palette("dark")
    sns.set(font='Times New Roman')
    sns.set_style("dark", {
            "font.family": "Times New Roman"
            })
    sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2})
    # sns.set(color_codes=True)


def excelToDict(filePath, fileName, **kwargs):
    fullPath = join(filePath, fileName)
    xlsx = read_excel(fullPath, skiprows=1, usecols='A:AC')
    return xlsx


def shotDict(df, shotNums):
    # shotStr = 'Shot ' + str(shotNum).zfill(4)
    dataDict = {}
    keys = df.keys().to_list()
    for shot in shotNums:
        # print(df.keys())  # TROUBLESHOOT LINE
        index = df['shotNum'][xlsx['shotNum'] == shot].index[0]
        shotStr = 'Shot ' + str(shot).zfill(4)
        dataDict[shotStr] = {}
        for key in keys:
            dataDict[shotStr][key] = df[key][index]

    return dataDict


def getUnique(inputList):
    # inputArr = np.array(inputList)
    uList, idx = np.unique(inputList, return_index=True)
    uniqueList = list(uList[np.argsort(idx)])
    return uniqueList


def arrayFromDict(dataDict, key):
    keyValList = []
    for shot in list(dataDict.keys()):
        keyValList.append(dataDict[shot][key])
    return np.array(keyValList)


def find_last(lst, sought_elt):
    for r_idx, elt in enumerate(reversed(lst)):
        if elt == sought_elt:
            return len(lst) - 1 - r_idx


def getUniqueLast(inputList):
    uniqueList = getUnique(inputList)
    lastIndices = []
    uniqueListLast = []
    for uniqueItem in uniqueList:
        lastIndex = find_last(inputList, uniqueItem)
        lastIndices.append(lastIndex)
        uniqueListLast.append(inputList[lastIndex])
    return lastIndices, uniqueListLast


set_style()

saveBool = 0
shotRange = [2400, 2422]
xBin = 512
yBin = 512
iccdFolder = '2399-2422'  # data folder

savepath = './images/'
vidPath = './videos/'

filePath = 'C:/Users/mccul/PycharmProjects/iccd_processing/'  # './iccd/hiper-pit'
fileName = 'shot 2399.spe'  # '2021 October 04 17_24_04.spe'
fullPath = join(filePath, iccdFolder, fileName)
os.chdir(filePath + iccdFolder)  # set current directory

shotRangeStr = 'shot_' + str(shotRange[0]) + '_to_' + str(shotRange[1])
vidFile1 = shotRangeStr + '_unscaled_v1.gif'
fullVidPath1 = join(vidPath, vidFile1)
vidFile2 = shotRangeStr + '_series_scaled_v1.gif'
fullVidPath2 = join(vidPath, vidFile2)
vidFile3 = shotRangeStr + '_global_scaled_v1.gif'
fullVidPath3 = join(vidPath, vidFile3)

figPath = 'C:/Users/mccul/PycharmProjects/iccd_processing/'
figFile1 = shotRangeStr + '_images_v1.pdf'
fullFigPath1 = join(figPath, figFile1)
figFile2 = shotRangeStr + '_pixel_intensity_v1.pdf'
fullFigPath2 = join(figPath, figFile2)

speFile = spe.load([fullPath])
speFile.image()

logPath = 'C:/Users/mccul/OneDrive/Documents/MSAA/Research'
logFile = 'hiper-pit_log_10-6-2021.xlsx'

shotRangeList = list(range(shotRange[0], shotRange[1] + 1))
xlsx = excelToDict(logPath, logFile)
# dataDict = xlsx.to_dict()
logDict = shotDict(xlsx, shotRangeList)

shotList = arrayFromDict(logDict, 'shotNum')
iccdDelay = arrayFromDict(logDict, 'iccdDelay')

lastIndices, uniqueDelay = getUniqueLast(iccdDelay)
shotNums = shotList[lastIndices]
iccdDelay = iccdDelay[lastIndices]
filenames = [join(filePath, iccdFolder, 'shot ' + str(int(shotNum)) + '.tif') for shotNum in shotNums]
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

minIntensities = [image.min() for image in images]
correctedImages = [(image - imMin) for image, imMin in zip(images, minIntensities)]
# plt.imshow(correctedImages[5])
maxIntensities = [image.max() for image in correctedImages]
seriesMaxIntensity = max(maxIntensities)  # 65000 # max(maxIntensities)/255
seriesNormalizedImages = [np.round(image/seriesMaxIntensity*255).astype(np.uint8) for image in correctedImages]
iccdMaxIntensity = 65000
globalNormalizedImages = [np.round(image/iccdMaxIntensity*255).astype(np.uint8) for image in correctedImages]

imageio.imwrite('./test.tif', images[-1])
data = imageio.imread('./test.tif')
plt.figure()
plt.imshow(data, cmap='plasma')

intensityAlongX = []
intensityAlongY = []
xPixelList = []
yPixelList = []
for image in correctedImages:
    xPixels, yPixels = np.shape(image)
    intensityAlongX.append(image[yBin])
    intensityAlongY.append([image[row][xBin] for row in list(range(yPixels))])
    xPixelList.append(list(range(1, xPixels + 1)))
    yPixelList.append(list(range(1, yPixels + 1)))

fig1, ax1 = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(16, 10), gridspec_kw=dict(wspace=0.0, hspace=-0.1))
fig2, ax2 = plt.subplots(4, 5, sharex=True, sharey=True, figsize=(16, 10), gridspec_kw=dict(wspace=0.0, hspace=0.0))

i = 0
j = 0
for count, image in enumerate(images):
    if j > 4:
        i += 1
        j = 0
    ax1[i, j].imshow(image, cmap='plasma')
    ax1[i, j].set_xticklabels([])
    ax1[i, j].set_yticklabels([])
    ax1[i, j].text(10, 80, str(np.round(iccdDelay[count] - 17.05, 2)) + '$\mu$s', fontsize=12, color='white')

    ax2[i, j].plot(xPixelList[count], intensityAlongX[count])
    ax2[i, j].plot(yPixelList[count], intensityAlongY[count])
    ax2[i, j].text(5, 22000, str(np.round(iccdDelay[count] - 17.05, 2)) + '$\mu$s', fontsize=12, color='black')
    if j == 0:
        ax2[i, j].set_ylabel('Intensity [a.u.]')
    if i == 2:
        ax2[i, j].set_xlabel('Pixels')

    j += 1

ax2[0, 0].legend(['Along Z', 'Along R'], ncol=2, bbox_to_anchor=(0, 1.01, 1, 0.1), loc="lower left",
                 borderaxespad=0, facecolor='white', framealpha=1)
fig1.suptitle(shotRangeStr)
fig2.suptitle(shotRangeStr)
# for count, image in enumerate(correctedImages):
#     ax2[0].plot(xPixelList[count], intensityAlongX[count])
#     ax2[1].plot(yPixelList[count], intensityAlongY[count])

# fig1.tight_layout()
# for image in normImages:
#     draw = ImageDraw.Draw(image)
#     font = ImageFont.truetype("sans-serif.ttf", 16)
#     draw.text((20, 20), 'Test', (255, 2555, 255), font=font)

if saveBool:
    imageio.mimwrite(fullVidPath1, correctedImages, duration=0.25)
    imageio.mimwrite(fullVidPath2, seriesNormalizedImages, duration=0.25)
    imageio.mimwrite(fullVidPath3, globalNormalizedImages, duration=0.25)
    fig1.savefig(fullFigPath1, bbox_inches='tight')
    fig1.savefig(fullFigPath1[:-4] + '.png', bbox_inches='tight', dpi=400)
    fig2.savefig(fullFigPath2, bbox_inches='tight')
    fig2.savefig(fullFigPath2[:-4] + '.png', bbox_inches='tight', dpi=400)

plt.show()