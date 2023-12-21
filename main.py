from easyocr import Reader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import arabic_reshaper
from bidi.algorithm import get_display
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from numba import jit

## testing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



##################################################################################################

class Character:
    def __init__(self, char, template='', width=60, height=60, img=None):
        self.char = char
        if img is None:
            self.template = cv2.imread(template, 0)
        else:
            self.template = img
        self.col_sum = np.zeros(shape=(height, width))
        self.corr = 0
        self.resize_and_calculate(width, height)

    def resize_and_calculate(self, width, height):
        # Perform resizing of the template
        dim = (width, height)
        self.template = cv2.resize(self.template, dim, interpolation=cv2.INTER_AREA)

        # Perform calculations using char_calculations function
        self.corr, self.col_sum = self.char_calculations(self.template, height, width)

    @staticmethod
    def char_calculations(A, width, height):
        A_mean = A.mean()
        col_A = 0
        corr_A = 0
        sum_list = np.zeros(shape=(height, width))
        img_row = 0
        while img_row < height:
            img_col = 0
            while img_col < width:
                col_A += (A[img_row, img_col] - A_mean) ** 2
                sum_list[img_row][img_col] = abs(A[img_row, img_col] - A_mean)
                img_col += 1
            corr_A += col_A
            col_A = 0
            img_row += 1
        return corr_A, sum_list

additionsDataBase = []
parasitismsDataBase = []
CharDataBase = []

additionsWidth = 60
additionsHeight = 60

def buildAdditionsDB():
	global additionsDataBase 
	additionsDataBase = []
	hamza = Character('hamza','dataSet/Additions/hamza.jpg',width=additionsWidth,height=additionsHeight)
	no2taB_1 = Character('no2taB','dataSet/Additions/no2taBeh_1.jpg',width=additionsWidth,height=additionsHeight)
	no2taB_2 = Character('no2taB','dataSet/Additions/no2taBeh_2.jpg',width=additionsWidth,height=additionsHeight)
	no2taB_3 = Character('no2taB','dataSet/Additions/no2taBeh_3.jpg',width=additionsWidth,height=additionsHeight)
	no2taN_1 = Character('no2taN','dataSet/Additions/no2taNoon_1.jpg',width=additionsWidth,height=additionsHeight)
	no2taN_2 = Character('no2taN','dataSet/Additions/no2taNoon_2.jpg',width=additionsWidth,height=additionsHeight)
	no2taG = Character('no2taG','dataSet/Additions/no2taGem_1.jpg',width=additionsWidth,height=additionsHeight)
	additionsDataBase.append(hamza)
	additionsDataBase.append(no2taB_1)
	additionsDataBase.append(no2taB_2)
	additionsDataBase.append(no2taB_3)
	additionsDataBase.append(no2taN_1)
	additionsDataBase.append(no2taN_2)
	additionsDataBase.append(no2taG)

def buildParasitismsDB():
    global parasitismsDataBase 
    parasitismsDataBase = []
    bar1 = Character('bar', 'dataSet/Additions/bar1.jpg')
    bar2 = Character('bar', 'dataSet/Additions/bar2.jpg')
    bar3 = Character('bar', 'dataSet/Additions/bar3.jpg')
    bar4 = Character('bar', 'dataSet/Additions/bar4.jpg')
    nesr1 = Character('nesr', 'dataSet/Additions/nesr1.jpg')
    nesr2 = Character('nesr', 'dataSet/Additions/nesr2.jpg')
    nesr3 = Character('nesr', 'dataSet/Additions/nesr3.jpg')
    parasitismsDataBase.append(bar1)
    parasitismsDataBase.append(bar2)
    parasitismsDataBase.append(bar3)
    parasitismsDataBase.append(bar4)
    parasitismsDataBase.append(nesr1)
    parasitismsDataBase.append(nesr2)
    parasitismsDataBase.append(nesr3)

def buildCharDB():
    # Letters
    global CharDataBase
    CharDataBase = []

    Alf1 = Character("alf", 'dataSet/Char/alf_1.jpg',width=120,hieght=120)
    Alf2 = Character("alf", 'dataSet/Char/alf_2.jpg',width=120,hieght=120)
    Alf3 = Character("alf", 'dataSet/Char/alf_3.jpg',width=120,hieght=120)
    Alf4 = Character("alf", 'dataSet/Char/alf_4.jpg',width=120,hieght=120)
    Alf5 = Character("alf", 'dataSet/Char/alf_5.jpg',width=120,hieght=120)
    Alf6 = Character("alf", 'dataSet/Char/alf_6.jpg',width=120,hieght=120)
    Alf7 = Character("alf", 'dataSet/Char/alf_7.png',width=120,hieght=120)
    Alf8 = Character("alf", 'dataSet/Char/alf_8.jpg',width=120,hieght=120)
    Alf9 = Character("alf", 'dataSet/Char/alf_9.jpg',width=120,hieght=120)
    Beh1 = Character("beh", 'dataSet/Char/beh_1.jpg',width=120,hieght=120)
    Beh2 = Character("beh", 'dataSet/Char/beh_2.jpg',width=120,hieght=120)
    Beh3 = Character("beh", 'dataSet/Char/beh_3.jpg',width=120,hieght=120)
    Beh4 = Character("beh", 'dataSet/Char/beh_4.jpg',width=120,hieght=120)
    Beh5 = Character("beh", 'dataSet/Char/beh_5.jpg',width=120,hieght=120)
    Dal1 = Character("dal", 'dataSet/Char/dal_1.jpg',width=120,hieght=120)
    Dal2 = Character("dal", 'dataSet/Char/dal_2.jpg',width=120,hieght=120)
    Dal3 = Character("dal", 'dataSet/Char/dal_3.jpg',width=120,hieght=120)
    Dal4 = Character("dal", 'dataSet/Char/dal_4.jpg',width=120,hieght=120)
    Dal5 = Character("dal", 'dataSet/Char/dal_5.jpg',width=120,hieght=120)
    Dal6 = Character("dal", 'dataSet/Char/dal_6.jpg',width=120,hieght=120)
    Ein1 = Character("ein", 'dataSet/Char/ein_1.png',width=120,hieght=120)
    Ein2 = Character("ein", 'dataSet/Char/ein_2.png',width=120,hieght=120)
    Ein3 = Character("ein", 'dataSet/Char/ein_3.png',width=120,hieght=120)
    Fih1 = Character("fih", 'dataSet/Char/fih_1.jpg',width=120,hieght=120)
    Fih2 = Character("fih", 'dataSet/Char/fih_2.png',width=120,hieght=120)
    Gem1 = Character("gem", 'dataSet/Char/gem_1.jpg',width=120,hieght=120)
    Gem2 = Character("gem", 'dataSet/Char/gem_2.jpg',width=120,hieght=120)
    Gem3 = Character("gem", 'dataSet/Char/gem_3.jpg',width=120,hieght=120)
    Gem4 = Character("gem", 'dataSet/Char/gem_4.jpg',width=120,hieght=120)
    Gem5 = Character("gem", 'dataSet/Char/gem_5.jpg',width=120,hieght=120)
    Heh1 = Character("heh", 'dataSet/Char/heh_1.jpg',width=120,hieght=120)
    Heh2 = Character("heh", 'dataSet/Char/heh_2.png',width=120,hieght=120)
    Heh3 = Character("heh", 'dataSet/Char/heh_3.png',width=120,hieght=120)
    Kaf1 = Character("kaf", 'dataSet/Char/kaf_1.jpg',width=120,hieght=120)
    Kaf2 = Character("kaf", 'dataSet/Char/kaf_2.jpg',width=120,hieght=120)
    Kaf3 = Character("kaf", 'dataSet/Char/kaf_3.jpg',width=120,hieght=120)
    Kaf4 = Character("kaf", 'dataSet/Char/kaf_4.jpg',width=120,hieght=120)
    Kaf5 = Character("kaf", 'dataSet/Char/kaf_5.jpg',width=120,hieght=120)
    Kaf6 = Character("kaf", 'dataSet/Char/kaf_6.jpg',width=120,hieght=120)
    Kaf7 = Character("kaf", 'dataSet/Char/kaf_7.png',width=120,hieght=120)
    Lam1 = Character("lam", 'dataSet/Char/lam_1.png',width=120,hieght=120)
    Lam2 = Character("lam", 'dataSet/Char/lam_2.png',width=120,hieght=120)
    Lam3 = Character("lam", 'dataSet/Char/lam_3.jpg',width=120,hieght=120)
    Mem1 = Character("mem", 'dataSet/Char/mem_1.jpg',width=120,hieght=120)
    Mem2 = Character("mem", 'dataSet/Char/mem_2.jpg',width=120,hieght=120)
    Mem3 = Character("mem", 'dataSet/Char/mem_3.jpg',width=120,hieght=120)
    Mem4 = Character("mem", 'dataSet/Char/mem_4.jpg',width=120,hieght=120)
    Mem5 = Character("mem", 'dataSet/Char/mem_5.jpg',width=120,hieght=120)
    Non1 = Character("non", 'dataSet/Char/non_1.png',width=120,hieght=120)
    Non2 = Character("non", 'dataSet/Char/non_2.png',width=120,hieght=120)
    Reh1 = Character("reh", 'dataSet/Char/reh_1.png',width=120,hieght=120)
    Reh2 = Character("reh", 'dataSet/Char/reh_2.jpg',width=120,hieght=120)
    Reh3 = Character("reh", 'dataSet/Char/reh_3.jpg',width=120,hieght=120)
    Reh4 = Character("reh", 'dataSet/Char/reh_4.jpg',width=120,hieght=120)
    Reh5 = Character("reh", 'dataSet/Char/reh_5.jpg',width=120,hieght=120)
    Sad1 = Character("sad", 'dataSet/Char/sad_1.jpg',width=120,hieght=120)
    Sad2 = Character("sad", 'dataSet/Char/sad_2.jpg',width=120,hieght=120)
    Sad3 = Character("sad", 'dataSet/Char/sad_3.jpg',width=120,hieght=120)
    Sad4 = Character("sad", 'dataSet/Char/sad_4.jpg',width=120,hieght=120)
    Sad5 = Character("sad", 'dataSet/Char/sad_5.jpg',width=120,hieght=120)
    Sad6 = Character("sad", 'dataSet/Char/sad_6.jpg',width=120,hieght=120)
    Sen1 = Character("sen", 'dataSet/Char/sen_1.jpg',width=120,hieght=120)
    Sen2 = Character("sen", 'dataSet/Char/sen_2.png',width=120,hieght=120)
    Tah1 = Character("tah", 'dataSet/Char/tah_1.jpg',width=120,hieght=120)
    Tah2 = Character("tah", 'dataSet/Char/tah_2.jpg',width=120,hieght=120)
    Tah3 = Character("tah", 'dataSet/Char/tah_3.jpg',width=120,hieght=120)
    Waw1 = Character("waw", 'dataSet/Char/waw_1.jpg',width=120,hieght=120)
    Waw2 = Character("waw", 'dataSet/Char/waw_2.jpg',width=120,hieght=120)
    Waw3 = Character("waw", 'dataSet/Char/waw_3.jpg',width=120,hieght=120)
    Waw4 = Character("waw", 'dataSet/Char/waw_4.jpg',width=120,hieght=120)
    Waw5 = Character("waw", 'dataSet/Char/waw_5.jpg',width=120,hieght=120)
    Waw6 = Character("waw", 'dataSet/Char/waw_6.jpg',width=120,hieght=120)
    Waw7 = Character("waw", 'dataSet/Char/waw_7.jpg',width=120,hieght=120)
    Waw8 = Character("waw", 'dataSet/Char/waw_8.jpg',width=120,hieght=120)
    Waw9 = Character("waw", 'dataSet/Char/waw_9.jpg',width=120,hieght=120)
    Yeh1 = Character("yeh", 'dataSet/Char/yeh_1.jpg',width=120,hieght=120)
    Yeh2 = Character("yeh", 'dataSet/Char/yeh_2.jpg',width=120,hieght=120)


    # Numbers
    One1 = Character("1", 'dataSet/Char/one_1.jpg',width=120,hieght=120)
    One2 = Character("1", 'dataSet/Char/one_2.jpg',width=120,hieght=120)
    One3 = Character("1", 'dataSet/Char/one_3.jpg',width=120,hieght=120)
    One4 = Character("1", 'dataSet/Char/one_4.jpg',width=120,hieght=120)
    One5 = Character("1", 'dataSet/Char/one_5.jpg',width=120,hieght=120)
    Two1 = Character("2", 'dataSet/Char/two_1.jpg',width=120,hieght=120)
    Two2 = Character("2", 'dataSet/Char/two_2.jpg',width=120,hieght=120)
    Two3 = Character("2", 'dataSet/Char/two_3.jpg',width=120,hieght=120)
    Two4 = Character("2", 'dataSet/Char/two_4.jpg',width=120,hieght=120)
    Two5 = Character("2", 'dataSet/Char/two_5.jpg',width=120,hieght=120)
    Three1 = Character("3", 'dataSet/Char/three_1.jpg',width=120,hieght=120)
    Three2 = Character("3", 'dataSet/Char/three_2.jpg',width=120,hieght=120)
    Three3 = Character("3", 'dataSet/Char/three_3.jpg',width=120,hieght=120)
    Three4 = Character("3", 'dataSet/Char/three_4.jpg',width=120,hieght=120)
    Three5 = Character("3", 'dataSet/Char/three_5.jpg',width=120,hieght=120)
    Four1 = Character("4", 'dataSet/Char/four_1.jpg',width=120,hieght=120)
    Four2 = Character("4", 'dataSet/Char/four_2.jpg',width=120,hieght=120)
    Four3 = Character("4", 'dataSet/Char/four_3.jpg',width=120,hieght=120)
    Four4 = Character("4", 'dataSet/Char/four_4.jpg',width=120,hieght=120)
    Four5 = Character("4", 'dataSet/Char/four_5.jpg',width=120,hieght=120)
    Five1 = Character("5", 'dataSet/Char/five_1.jpg',width=120,hieght=120)
    Five2 = Character("5", 'dataSet/Char/five_2.jpg',width=120,hieght=120)
    Five3 = Character("5", 'dataSet/Char/five_3.jpg',width=120,hieght=120)
    Five4 = Character("5", 'dataSet/Char/five_4.jpg',width=120,hieght=120)
    Five5 = Character("5", 'dataSet/Char/five_5.jpg',width=120,hieght=120)
    Six1 = Character("6", 'dataSet/Char/six_1.jpg',width=120,hieght=120)
    Six2 = Character("6", 'dataSet/Char/six_2.jpg',width=120,hieght=120)
    Six3 = Character("6", 'dataSet/Char/six_3.jpg',width=120,hieght=120)
    Six4 = Character("6", 'dataSet/Char/six_4.jpg',width=120,hieght=120)
    Seven1 = Character("7", 'dataSet/Char/seven_1.jpg',width=120,hieght=120)
    Seven2 = Character("7", 'dataSet/Char/seven_2.jpg',width=120,hieght=120)
    Seven3 = Character("7", 'dataSet/Char/seven_3.jpg',width=120,hieght=120)
    Seven4 = Character("7", 'dataSet/Char/seven_4.jpg',width=120,hieght=120)
    Seven5 = Character("7", 'dataSet/Char/seven_5.jpg',width=120,hieght=120)
    Eight1 = Character("8", 'dataSet/Char/eight_1.jpg',width=120,hieght=120)
    Eight2 = Character("8", 'dataSet/Char/eight_2.jpg',width=120,hieght=120)
    Eight3 = Character("8", 'dataSet/Char/eight_3.jpg',width=120,hieght=120)
    Eight4 = Character("8", 'dataSet/Char/eight_4.jpg',width=120,hieght=120)
    Nine1 = Character("9", 'dataSet/Char/nine_1.jpg',width=120,hieght=120)
    Nine2 = Character("9", 'dataSet/Char/nine_2.jpg',width=120,hieght=120)
    Nine3 = Character("9", 'dataSet/Char/nine_3.jpg',width=120,hieght=120)
    Nine4 = Character("9", 'dataSet/Char/nine_4.jpg',width=120,hieght=120)
    Nine5 = Character("9", 'dataSet/Char/nine_5.jpg',width=120,hieght=120)


	# Add to database
    # Append Alf instances
    CharDataBase.append(Alf1)
    CharDataBase.append(Alf2)
    CharDataBase.append(Alf3)
    CharDataBase.append(Alf4)
    CharDataBase.append(Alf5)
    CharDataBase.append(Alf6)
    CharDataBase.append(Alf7)
    CharDataBase.append(Alf8)
    CharDataBase.append(Alf9)
    # Append Beh instances
    CharDataBase.append(Beh1)
    CharDataBase.append(Beh2)
    CharDataBase.append(Beh3)
    CharDataBase.append(Beh4)
    CharDataBase.append(Beh5)

    # Append Dal instances
    CharDataBase.append(Dal1)
    CharDataBase.append(Dal2)
    CharDataBase.append(Dal3)
    CharDataBase.append(Dal4)
    CharDataBase.append(Dal5)
    CharDataBase.append(Dal6)

    # Append Ein instances
    CharDataBase.append(Ein1)
    CharDataBase.append(Ein2)
    CharDataBase.append(Ein3)

    # Append Fih instances
    CharDataBase.append(Fih1)
    CharDataBase.append(Fih2)

    # Append Gem instances
    CharDataBase.append(Gem1)
    CharDataBase.append(Gem2)
    CharDataBase.append(Gem3)
    CharDataBase.append(Gem4)
    CharDataBase.append(Gem5)

    # Append Heh instances
    CharDataBase.append(Heh1)
    CharDataBase.append(Heh2)
    CharDataBase.append(Heh3)

    # Append Kaf instances
    CharDataBase.append(Kaf1)
    CharDataBase.append(Kaf2)
    CharDataBase.append(Kaf3)
    CharDataBase.append(Kaf4)
    CharDataBase.append(Kaf5)
    CharDataBase.append(Kaf6)
    CharDataBase.append(Kaf7)

    # Append Lam instances
    CharDataBase.append(Lam1)
    CharDataBase.append(Lam2)
    CharDataBase.append(Lam3)

    # Append Mem instances
    CharDataBase.append(Mem1)
    CharDataBase.append(Mem2)
    CharDataBase.append(Mem3)
    CharDataBase.append(Mem4)
    CharDataBase.append(Mem5)

    # Append Non instances
    CharDataBase.append(Non1)
    CharDataBase.append(Non2)

    # Append Reh instances
    CharDataBase.append(Reh1)
    CharDataBase.append(Reh2)
    CharDataBase.append(Reh3)
    CharDataBase.append(Reh4)
    CharDataBase.append(Reh5)

    # Append Sad instances
    CharDataBase.append(Sad1)
    CharDataBase.append(Sad2)
    CharDataBase.append(Sad3)
    CharDataBase.append(Sad4)
    CharDataBase.append(Sad5)
    CharDataBase.append(Sad6)

    # Append Sen instances
    CharDataBase.append(Sen1)
    CharDataBase.append(Sen2)

    # Append Tah instances
    CharDataBase.append(Tah1)
    CharDataBase.append(Tah2)
    CharDataBase.append(Tah3)

    # Append Waw instances
    CharDataBase.append(Waw1)
    CharDataBase.append(Waw2)
    CharDataBase.append(Waw3)
    CharDataBase.append(Waw4)
    CharDataBase.append(Waw5)
    CharDataBase.append(Waw6)
    CharDataBase.append(Waw7)
    CharDataBase.append(Waw8)
    CharDataBase.append(Waw9)

    # Append Yeh instances
    CharDataBase.append(Yeh1)
    CharDataBase.append(Yeh2)
    
    # Append One instances
    CharDataBase.append(One1)
    CharDataBase.append(One2)
    CharDataBase.append(One3)
    CharDataBase.append(One4)
    CharDataBase.append(One5)
    
    # Append Two instances
    CharDataBase.append(Two1)
    CharDataBase.append(Two2)
    CharDataBase.append(Two3)
    CharDataBase.append(Two4)
    CharDataBase.append(Two5)
    
    # Append Three instances
    CharDataBase.append(Three1)
    CharDataBase.append(Three2)
    CharDataBase.append(Three3)
    CharDataBase.append(Three4)
    CharDataBase.append(Three5)
    
    # Append Four instances
    CharDataBase.append(Four1)
    CharDataBase.append(Four2)
    CharDataBase.append(Four3)
    CharDataBase.append(Four4)
    CharDataBase.append(Four5)
    
    # Append Five instances
    CharDataBase.append(Five1)
    CharDataBase.append(Five2)
    CharDataBase.append(Five3)
    CharDataBase.append(Five4)
    CharDataBase.append(Five5)
    
    # Append Six instances
    CharDataBase.append(Six1)
    CharDataBase.append(Six2)
    CharDataBase.append(Six3)
    CharDataBase.append(Six4)
    
    # Append Seven instances
    CharDataBase.append(Seven1)
    CharDataBase.append(Seven2)
    CharDataBase.append(Seven3)
    CharDataBase.append(Seven4)
    CharDataBase.append(Seven5)
    
    # Append Eight instances
    CharDataBase.append(Eight1)
    CharDataBase.append(Eight2)
    CharDataBase.append(Eight3)
    CharDataBase.append(Eight4)
    
    # Append Nine instances
    CharDataBase.append(Nine1)
    CharDataBase.append(Nine2)
    CharDataBase.append(Nine3)
    CharDataBase.append(Nine4)
    CharDataBase.append(Nine5)
    
      
      
def getSimilarity(img1, img2):
    dim = (120,120)
    img1 = cv2.GaussianBlur(img1,(19,19),0)
    img2 = cv2.GaussianBlur(img2,(19,19),0)
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    _,img1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    result = img1 - img2
    result = result * result
    result = np.sum(result)
    result = np.sqrt(result)
    return result



#################################################################################################


# Function to render Arabic text on the image
def put_arabic_text(image, text, position, color=(0, 255, 0), thickness=1):
    font_file = "Sahel.ttf"  # File containing the Arabic font
    font = ImageFont.truetype(font_file, 18)  # Load the font with size 18
    reshaped_text = arabic_reshaper.reshape(text)  # Reshape Arabic text for correct display
    bidi_text = get_display(reshaped_text)  # Get bidi text for correct RTL display
    
    # Convert the OpenCV image (numpy array) to a Pillow Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Use Pillow to draw the Arabic text on the image
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, bidi_text, color, font=font, stroke_width=thickness)  # Draw the text
    
    # Convert the modified Pillow image back to a numpy array
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image

# Preprocessing steps for the image before plate detection
def pre_process_image(image_path):
    # Read the image from the specified path
    car = cv2.imread(image_path)
    
    # Resize the image to a specific size (600x400 pixels)
    resized_image = cv2.resize(car, (600, 400))
    
    # Convert the resized image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter for noise removal while keeping the edges sharp
    noise_removal = cv2.bilateralFilter(gray, 11, 17, 17)
    
    return noise_removal


# Detect license plate using Haar Cascade Classifier
def plate_detection_haarcascade():
    # Preprocess the image before plate detection
    car = pre_process_image('cars/car23.jpg')
    
    # Load the pre-trained Haar Cascade classifier for license plates
    plate_cascade = cv2.CascadeClassifier("haarcascade_license_plate.xml")
    
    # Detect potential plates in the preprocessed image
    plates = plate_cascade.detectMultiScale(car, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected license plates
    for (x, y, w, h) in plates:
        cv2.rectangle(car, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Extract the region of the first detected plate
    for (x, y, w, h) in plates:
        plate_region = car[y:y + h, x:x + w]
        break  # Extract the first detected plate only
    
    return plate_region


@jit(nopython=True)
def get_white_blue(image, aux):
    
    h = image.shape[0]; w = image.shape[1]
    white = 200; local_thre = 30; global_thre = 60

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            b,g,r = image[y,x]
            
            s,m,l = np.sort(image[y,x])
            
            local_dis = (m-l)*(m-l)+(m-s)*(m-s)
            aux[y, x, 0] = 1 if (local_dis<local_thre*local_thre and abs(white-(s+m+l)/3)<global_thre) else 0
            aux[y, x, 1] = 1 if (s==r and l==b and r<100 and b>120 and b-g>20 and b-g<110) else 0

def check_blue(pixel):
        b,g,r = pixel
        s,m,l = np.sort(pixel)
        if (s==r and l==b and r<100 and b>120 and b-g>20 and b-g<110):
            return True
        else:
            return False

def sum_range(aux, Xmin, Ymin, Xmax, Ymax): 
    res = aux[Ymax][Xmax] 
    
    if (Ymin > 0): res = res - aux[Ymin - 1][Xmax] 
        
    if (Xmin > 0): res = res - aux[Ymax][Xmin - 1] 
    
    if (Ymin > 0 and Xmin > 0): res = res + aux[Ymin - 1][Xmin - 1] 
    
    return res 


def remove_noise(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return blur

#working well with bimodal images (Fast Algo)
def binarization_otsu(gray_img):
    ret,bin_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret,bin_img

def detect_edges(gray_img): #Sobel Edge detection
    scale = 1; delta = 0; ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(gray_img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray_img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x); abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

#depend on aspect ratio, center point of plate, plate color, far or not
def plate_criteria(cum_white, cum_blue, x, y, w, h, aspect_min, aspect_max, far): 
    area = w*h
    [Xmin,Ymin,Xmax,Ymax] = [x,y,x+w-1,y+h-1]
    if(h>0 and aspect_min < float(w)/h and float(w)/h < aspect_max): #Check Aspect ration
        if(area >= cum_white.shape[0] * cum_white.shape[1] * far): #check far or not
            white_ratio = sum_range(cum_white, Xmin, Ymin, Xmax, Ymax)/area*100
            blue_ratio = sum_range(cum_blue, Xmin, Ymin, Xmax, Ymax)/area*100
            if(white_ratio > 35 and white_ratio < 90 and blue_ratio > 7 and blue_ratio < 40):
                return True
    return False

def plate_contour(img, bin_img, aspect_min, aspect_max, far): #Image should be BGR Image not RGB
    #Because Some version return 2 parameters and other return 3 parameters
    major = cv2.__version__.split('.')[0]
    if major == '3': img2, bounding_boxes, hierarchy= cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else: bounding_boxes, hierarchy= cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    aux = np.copy(img)
    get_white_blue(img,aux)

    cum_white = np.cumsum(aux[:,:,0], axis = 0)
    cum_white = np.cumsum(cum_white, axis = 1).astype(np.int64) #To Avoid overflow in sum_range
    cum_blue = np.cumsum(aux[:,:,1], axis = 0)
    cum_blue = np.cumsum(cum_blue, axis = 1).astype(np.int64)   #To Avoid overflow in sum_range

    for box in bounding_boxes:    
        [x,y, w, h] = cv2.boundingRect(box)
        if(plate_criteria(cum_white, cum_blue, x, y, w, h,aspect_min, aspect_max, far)):
            if(y-h/4>=0):
                return np.copy(img[y-int(h/4):y+h-1,x:x+w-1]),1
            else:    
                return np.copy(img[y:y+h-1,x:x+w-1]),1
    return img,0
def resize_image(img):
    if(img.shape[0]*img.shape[1]>1000000):
        h =  np.sqrt(1000000/(img.shape[1]*img.shape[0]))
        y = int(h*img.shape[0])
        x = int(h*img.shape[1])
        #print(x*y,img.shape[0]*img.shape[1])
        img = cv2.resize(img, (x,y), interpolation = cv2.INTER_AREA)
    return img

def rotate_blue(img):
    h,w,_ = img.shape
    x1 = int(img.shape[0]/4)
    y1 = 0
    while(y1 < h and not check_blue(img[y1][x1])):
        y1+=1
    x2 = img.shape[0]-int(img.shape[0]/4)
    y2 = 0
    while(y2 < h and not check_blue(img[y2][x2])):
        y2+=1
    center = (int(w/2),int(h/2))
    angle = np.arctan((y2-y1)/(x2-x1))*180*7/22
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
def crop_up(img):
    y = img.shape[0]
    x = int (img.shape[0]/2)
    for i in range(0,y):
        if(check_blue(img[i][x])):
            return img[i:y,0:img.shape[1]]
    return img

def localization(img): #take BGR image and return BGR image
    img = resize_image(img)
    img = remove_noise(img)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = detect_edges(gray_img)

    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    ret,bin_img = binarization_otsu(closing)

    plate_area_img,flag = plate_contour(img, bin_img, 1.4, 2.5, 0.01) 

    plate_area_img_bin = cv2.adaptiveThreshold(cv2.cvtColor(255-plate_area_img,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,8)

    plate_img,flag2 = plate_contour(plate_area_img, plate_area_img_bin, 1, 2.1, 0.1) 
    cropped = np.copy(plate_img)
    if(flag):
        #rotated = rotate_blue(plate_img)
        cropped = crop_up(plate_img)
    return cropped
# Detect license plate using contours
def plate_detection_using_contours():
    car = pre_process_image('cars/car23.jpg')
    edged = cv2.Canny(car, 10, 200)

  

    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    
    cv2.imshow('Image3', edged)
    cv2.imshow('Image4', closing)
    cv2.imshow('Image5', opening)
    cv2.waitKey(0)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort the contours by area in descending order and select the top 5 contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Iterate through the detected contours
    for contour in contours:
        # Calculate the arc length of the contour
        arc_length = cv2.arcLength(contour, True)
        
        # Approximate the contour shape to a polygonal curve
        approx = cv2.approxPolyDP(contour, 0.02 * arc_length, True)

        # Check if the approximated polygon has 4 sides (likely a license plate)
        if len(approx) == 4:
            plate_contour = approx
            break  # Stop iterating if a valid plate contour is found
    
    # Get the bounding rectangle of the plate contour
    (x, y, w, h) = cv2.boundingRect(plate_contour)
    
    # Extract the region of interest (ROI) containing the detected plate
    plate = car[y:y + h, x:x + w]
    return plate


# buildCharDB()
# Perform plate detection using both methods
# plate1 = plate_detection_using_contours()
# cv2.imshow('Image', plate1)
img = cv2.imread('cars/car23.jpg')
plate2 = localization(img)
cv2.imshow('Image2', plate2)
plate3 = plate_detection_haarcascade()
cv2.imshow('Image3', plate3)
cv2.waitKey(0)
# # Perform OCR to read text from the detected plates
# reader = Reader(['ar'], gpu=False, verbose=False)
# detection1 = reader.readtext(plate1)
# detection2 = reader.readtext(plate2)
# print(detection1)
# print(detection2)

# # Display the detected plates
# cv2.imshow('License Plate 1', plate1)
# cv2.imshow('License Plate 2', plate2)
# cv2.waitKey(0)

# if len(detection) == 0:
#     text = "Impossible to read the text from the license plate"
#     cv2.putText(car, text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 3)
#     cv2.drawContours(car, [plate_cnt], -1, (0, 255, 0), 3)
#     cv2.putText(car, '', (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
#     cv2.imshow('Image', car)
#     cv2.waitKey(0)
# else:
#     cv2.drawContours(car, [plate_cnt], -1, (0, 255, 0), 3)
#     text = f"{detection[0][1]} {detection[0][2] * 100:.2f}%"
#     print(text)
#     car = put_arabic_text(car, text, (x, y - 30), (0, 255, 0), 1)
#     cv2.imshow('license plate', plate)
#     cv2.imshow('Image', car)
#     cv2.waitKey(0)

