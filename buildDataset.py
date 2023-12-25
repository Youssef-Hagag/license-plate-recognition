

# Assuming CharDataBase is populated with Character instances

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

CharDataBase = []

additionsWidth = 60
additionsHeight = 60

def buildCharDB():
    # Letters
    global CharDataBase
    CharDataBase = []

    Alf1 = Character("alf", 'dataSet/Char/alf_1.jpg',width=120,height=120)
    Alf2 = Character("alf", 'dataSet/Char/alf_2.jpg',width=120,height=120)
    Alf3 = Character("alf", 'dataSet/Char/alf_3.jpg',width=120,height=120)
    Alf4 = Character("alf", 'dataSet/Char/alf_4.jpg',width=120,height=120)
    Alf5 = Character("alf", 'dataSet/Char/alf_5.jpg',width=120,height=120)
    Alf6 = Character("alf", 'dataSet/Char/alf_6.jpg',width=120,height=120)
    Alf7 = Character("alf", 'dataSet/Char/alf_7.png',width=120,height=120)
    Alf8 = Character("alf", 'dataSet/Char/alf_8.jpg',width=120,height=120)
    Alf9 = Character("alf", 'dataSet/Char/alf_9.jpg',width=120,height=120)
    Beh1 = Character("beh", 'dataSet/Char/beh_1.jpg',width=120,height=120)
    Beh2 = Character("beh", 'dataSet/Char/beh_2.jpg',width=120,height=120)
    Beh3 = Character("beh", 'dataSet/Char/beh_3.jpg',width=120,height=120)
    Beh4 = Character("beh", 'dataSet/Char/beh_4.jpg',width=120,height=120)
    Beh5 = Character("beh", 'dataSet/Char/beh_5.jpg',width=120,height=120)
    Dal1 = Character("dal", 'dataSet/Char/dal_1.jpg',width=120,height=120)
    Dal2 = Character("dal", 'dataSet/Char/dal_2.jpg',width=120,height=120)
    Dal3 = Character("dal", 'dataSet/Char/dal_3.jpg',width=120,height=120)
    Dal4 = Character("dal", 'dataSet/Char/dal_4.jpg',width=120,height=120)
    Dal5 = Character("dal", 'dataSet/Char/dal_5.jpg',width=120,height=120)
    Dal6 = Character("dal", 'dataSet/Char/dal_6.jpg',width=120,height=120)
    Ein1 = Character("ein", 'dataSet/Char/ein_1.png',width=120,height=120)
    Ein2 = Character("ein", 'dataSet/Char/ein_2.png',width=120,height=120)
    Ein3 = Character("ein", 'dataSet/Char/ein_3.png',width=120,height=120)
    Fih1 = Character("fih", 'dataSet/Char/fih_1.jpg',width=120,height=120)
    Fih2 = Character("fih", 'dataSet/Char/fih_2.png',width=120,height=120)
    Gem1 = Character("gem", 'dataSet/Char/gem_1.jpg',width=120,height=120)
    Gem2 = Character("gem", 'dataSet/Char/gem_2.jpg',width=120,height=120)
    Gem3 = Character("gem", 'dataSet/Char/gem_3.jpg',width=120,height=120)
    Gem4 = Character("gem", 'dataSet/Char/gem_4.jpg',width=120,height=120)
    Gem5 = Character("gem", 'dataSet/Char/gem_5.jpg',width=120,height=120)
    Heh1 = Character("heh", 'dataSet/Char/heh_1.jpg',width=120,height=120)
    Heh2 = Character("heh", 'dataSet/Char/heh_2.png',width=120,height=120)
    Heh3 = Character("heh", 'dataSet/Char/heh_3.png',width=120,height=120)
    Kaf1 = Character("kaf", 'dataSet/Char/kaf_1.jpg',width=120,height=120)
    Kaf2 = Character("kaf", 'dataSet/Char/kaf_2.jpg',width=120,height=120)
    Kaf3 = Character("kaf", 'dataSet/Char/kaf_3.jpg',width=120,height=120)
    Kaf4 = Character("kaf", 'dataSet/Char/kaf_4.jpg',width=120,height=120)
    Kaf5 = Character("kaf", 'dataSet/Char/kaf_5.jpg',width=120,height=120)
    Kaf6 = Character("kaf", 'dataSet/Char/kaf_6.jpg',width=120,height=120)
    Kaf7 = Character("kaf", 'dataSet/Char/kaf_7.png',width=120,height=120)
    Lam1 = Character("lam", 'dataSet/Char/lam_1.png',width=120,height=120)
    Lam2 = Character("lam", 'dataSet/Char/lam_2.png',width=120,height=120)
    Lam3 = Character("lam", 'dataSet/Char/lam_3.jpg',width=120,height=120)
    Mem1 = Character("mem", 'dataSet/Char/mem_1.jpg',width=120,height=120)
    Mem2 = Character("mem", 'dataSet/Char/mem_2.jpg',width=120,height=120)
    Mem3 = Character("mem", 'dataSet/Char/mem_3.jpg',width=120,height=120)
    Mem4 = Character("mem", 'dataSet/Char/mem_4.jpg',width=120,height=120)
    Mem5 = Character("mem", 'dataSet/Char/mem_5.jpg',width=120,height=120)
    Non1 = Character("non", 'dataSet/Char/non_1.png',width=120,height=120)
    Non2 = Character("non", 'dataSet/Char/non_2.png',width=120,height=120)
    Reh1 = Character("reh", 'dataSet/Char/reh_1.png',width=120,height=120)
    Reh2 = Character("reh", 'dataSet/Char/reh_2.jpg',width=120,height=120)
    Reh3 = Character("reh", 'dataSet/Char/reh_3.jpg',width=120,height=120)
    Reh4 = Character("reh", 'dataSet/Char/reh_4.jpg',width=120,height=120)
    Reh5 = Character("reh", 'dataSet/Char/reh_5.jpg',width=120,height=120)
    Sad1 = Character("sad", 'dataSet/Char/sad_1.jpg',width=120,height=120)
    Sad2 = Character("sad", 'dataSet/Char/sad_2.jpg',width=120,height=120)
    Sad3 = Character("sad", 'dataSet/Char/sad_3.jpg',width=120,height=120)
    Sad4 = Character("sad", 'dataSet/Char/sad_4.jpg',width=120,height=120)
    Sad5 = Character("sad", 'dataSet/Char/sad_5.jpg',width=120,height=120)
    Sad6 = Character("sad", 'dataSet/Char/sad_6.jpg',width=120,height=120)
    Sen1 = Character("sen", 'dataSet/Char/sen_1.jpg',width=120,height=120)
    Sen2 = Character("sen", 'dataSet/Char/sen_2.png',width=120,height=120)
    Tah1 = Character("tah", 'dataSet/Char/tah_1.jpg',width=120,height=120)
    Tah2 = Character("tah", 'dataSet/Char/tah_2.jpg',width=120,height=120)
    Tah3 = Character("tah", 'dataSet/Char/tah_3.jpg',width=120,height=120)
    Waw1 = Character("waw", 'dataSet/Char/waw_1.jpg',width=120,height=120)
    Waw2 = Character("waw", 'dataSet/Char/waw_2.jpg',width=120,height=120)
    Waw3 = Character("waw", 'dataSet/Char/waw_3.jpg',width=120,height=120)
    Waw4 = Character("waw", 'dataSet/Char/waw_4.jpg',width=120,height=120)
    Waw5 = Character("waw", 'dataSet/Char/waw_5.jpg',width=120,height=120)
    Waw6 = Character("waw", 'dataSet/Char/waw_6.jpg',width=120,height=120)
    Waw7 = Character("waw", 'dataSet/Char/waw_7.jpg',width=120,height=120)
    Waw8 = Character("waw", 'dataSet/Char/waw_8.jpg',width=120,height=120)
    Waw9 = Character("waw", 'dataSet/Char/waw_9.jpg',width=120,height=120)
    Yeh1 = Character("yeh", 'dataSet/Char/yeh_1.jpg',width=120,height=120)
    Yeh2 = Character("yeh", 'dataSet/Char/yeh_2.jpg',width=120,height=120)


    # Numbers
    One1 = Character("1", 'dataSet/Char/one_1.jpg',width=120,height=120)
    One2 = Character("1", 'dataSet/Char/one_2.jpg',width=120,height=120)
    One3 = Character("1", 'dataSet/Char/one_3.jpg',width=120,height=120)
    One4 = Character("1", 'dataSet/Char/one_4.jpg',width=120,height=120)
    One5 = Character("1", 'dataSet/Char/one_5.jpg',width=120,height=120)
    Two1 = Character("2", 'dataSet/Char/two_1.jpg',width=120,height=120)
    Two2 = Character("2", 'dataSet/Char/two_2.jpg',width=120,height=120)
    Two3 = Character("2", 'dataSet/Char/two_3.jpg',width=120,height=120)
    Two4 = Character("2", 'dataSet/Char/two_4.jpg',width=120,height=120)
    Two5 = Character("2", 'dataSet/Char/two_5.jpg',width=120,height=120)
    Three1 = Character("3", 'dataSet/Char/three_1.jpg',width=120,height=120)
    Three2 = Character("3", 'dataSet/Char/three_2.jpg',width=120,height=120)
    Three3 = Character("3", 'dataSet/Char/three_3.jpg',width=120,height=120)
    Three4 = Character("3", 'dataSet/Char/three_4.jpg',width=120,height=120)
    Three5 = Character("3", 'dataSet/Char/three_5.jpg',width=120,height=120)
    Four1 = Character("4", 'dataSet/Char/four_1.jpg',width=120,height=120)
    Four2 = Character("4", 'dataSet/Char/four_2.jpg',width=120,height=120)
    Four3 = Character("4", 'dataSet/Char/four_3.jpg',width=120,height=120)
    Four4 = Character("4", 'dataSet/Char/four_4.jpg',width=120,height=120)
    Four5 = Character("4", 'dataSet/Char/four_5.jpg',width=120,height=120)
    Five1 = Character("5", 'dataSet/Char/five_1.jpg',width=120,height=120)
    Five2 = Character("5", 'dataSet/Char/five_2.jpg',width=120,height=120)
    Five3 = Character("5", 'dataSet/Char/five_3.jpg',width=120,height=120)
    Five4 = Character("5", 'dataSet/Char/five_4.jpg',width=120,height=120)
    Five5 = Character("5", 'dataSet/Char/five_5.jpg',width=120,height=120)
    Six1 = Character("6", 'dataSet/Char/six_1.jpg',width=120,height=120)
    Six2 = Character("6", 'dataSet/Char/six_2.jpg',width=120,height=120)
    Six3 = Character("6", 'dataSet/Char/six_3.jpg',width=120,height=120)
    Six4 = Character("6", 'dataSet/Char/six_4.jpg',width=120,height=120)
    Seven1 = Character("7", 'dataSet/Char/seven_1.jpg',width=120,height=120)
    Seven2 = Character("7", 'dataSet/Char/seven_2.jpg',width=120,height=120)
    Seven3 = Character("7", 'dataSet/Char/seven_3.jpg',width=120,height=120)
    Seven4 = Character("7", 'dataSet/Char/seven_4.jpg',width=120,height=120)
    Seven5 = Character("7", 'dataSet/Char/seven_5.jpg',width=120,height=120)
    Eight1 = Character("8", 'dataSet/Char/eight_1.jpg',width=120,height=120)
    Eight2 = Character("8", 'dataSet/Char/eight_2.jpg',width=120,height=120)
    Eight3 = Character("8", 'dataSet/Char/eight_3.jpg',width=120,height=120)
    Eight4 = Character("8", 'dataSet/Char/eight_4.jpg',width=120,height=120)
    Nine1 = Character("9", 'dataSet/Char/nine_1.jpg',width=120,height=120)
    Nine2 = Character("9", 'dataSet/Char/nine_2.jpg',width=120,height=120)
    Nine3 = Character("9", 'dataSet/Char/nine_3.jpg',width=120,height=120)
    Nine4 = Character("9", 'dataSet/Char/nine_4.jpg',width=120,height=120)
    Nine5 = Character("9", 'dataSet/Char/nine_5.jpg',width=120,height=120)


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

