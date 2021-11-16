import os
import matplotlib.pyplot as plt
import matplotlib.image as img

# Adjust input parameters
shotRange = [2333, 2351] # shot data
iccd_folder = 'Oct2-P_1,1-2kV' # data folder

current_directory = 'C:/Users/McCullohGordonIC1CUS/Desktop/MSAA (2)/Research/ICCD/' # home desktop
os.chdir(current_directory + iccd_folder)
print('current directory: ',os.getcwd())

#test_plot = img.imread('./2333.png')
#cropped = test_plot[57:427, 144:512] # y, x
#plt.imshow(cropped)
#plt.axis('off')
#plt.show()

fig, crop = plt.subplots(3, 6, figsize=(6,3), dpi=400)
ii = 0

for shot in range(shotRange[0],shotRange[1]):
    shotStr = str(shot)
    shotPlot = img.imread('./'+shotStr+'.png')
    shotCrop = shotPlot[57:427, 144:512]
    if ii<6:
        crop[0,ii].imshow(shotCrop)
        crop[0,ii].axis('off')
    else:
        if ii<12:
            crop[1,ii-6].imshow(shotCrop)
            crop[1,ii-6].axis('off')
        else:
            crop[2,ii-12].imshow(shotCrop)
            crop[2,ii-12].axis('off')
    
    ii += 1

plt.axis('off')
plt.show()