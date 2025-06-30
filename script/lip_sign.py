from skimage.transform import radon, rescale
from skimage import io
import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)
'''
this function allow to compute shift on LIP profile
'''
def max_min(pLip):
    max_ind=-1
    max_value=sys.float_info.min
    min_ind=-1
    min_value=sys.float_info.max
    #find index of max element
    for i in range(len(pLip)):
        if(pLip[i]>max_value):
            max_value=pLip[i]
            max_ind=i
        if(pLip[i]<min_value):
            min_value=pLip[i]
            min_ind=i
    return max_ind,max_value,min_ind,min_value

'''
this function allow to aply shift on array
'''
def applyShift(arr,shift):
    norm_arr=[]
    #Shift the array
    for i in range(len(arr)):
        norm_arr.append(arr[(i+shift)%len(arr)])
    return norm_arr
'''
compute lenght of segment a,b. a and b got 2dim.
'''
def lengthSeg(a,b):
    return math.sqrt(((b[0]-a[0])*(b[0]-a[0]))+((b[1]-a[1])*(b[1]-a[1])))

'''
Compute features for an profil of Radon img
'''
def getFeaturesByProfil(profil):
    """
    Calcule six descripteurs (LIP0 à LIP5) à partir d’un profil 1D (extrait d’une transformée de Radon).
    
    Les points clefs identifiés sont :
      - M : maximum du profil (valeur et indice)
      - B : premier point strictement positif (infRho)
      - E : dernier point strictement positif (supRho)
      - H : projeté de M sur l’axe des abscisses (i.e. (argMaxVal, 0))
    
    La normalisation se fait par le nombre de valeurs strictement positives entre B et E.

    Paramètre :
        profil (ndarray) : vecteur 1D contenant le profil.

    Retour :
        tuple : (LIP0, LIP1, LIP2, LIP3, LIP4, LIP5)
    """

    # Initialisation des bornes
    maxVal = sys.float_info.min
    argMaxVal = -1
    infRho = len(profil) - 1  # plus petit indice > 0
    supRho = 0                 # plus grand indice > 0

    # Parcours du profil pour trouver le max et les bornes non nulles
    for i, val in enumerate(profil):
        if val > maxVal:
            maxVal = val
            argMaxVal = i
        if val > 0:
            infRho = min(infRho, i)
            supRho = max(supRho, i)

    # Définition des points clés
    M = (argMaxVal, maxVal)
    H = (argMaxVal, 0)         # Projection verticale de M
    B = (infRho, 0)            # Bord gauche du profil significatif
    E = (supRho, 0)            # Bord droit

    # Moyennes et écarts-types avant et après le maximum
    m1 = np.mean(profil[infRho:argMaxVal]) if argMaxVal > infRho else 0
    m2 = np.mean(profil[argMaxVal:supRho+1]) if supRho >= argMaxVal else 0

    y1 = np.std(profil[infRho:argMaxVal]) if argMaxVal > infRho else 0
    y2 = np.std(profil[argMaxVal:supRho+1]) if supRho >= argMaxVal else 0

    # Longueur du profil = nombre de valeurs strictement positives entre B et E
    profile_length = np.count_nonzero(profil[infRho:supRho+1] > 0)
    if profile_length == 0:
        profile_length = 1  # éviter division par zéro

    # Construction des six signatures LIP normalisées
    LIP0 = maxVal / profile_length
    LIP1 = (argMaxVal-infRho) / profile_length
    LIP2 = m1 / profile_length
    LIP3 = y1 / profile_length
    LIP4 = m2 / profile_length
    LIP5 = y2 / profile_length

    return LIP0, LIP1, LIP2, LIP3, LIP4, LIP5


def plot_column_vector(column_vector, output_path="column_plot.png",title="Radon Transform Column", xlabel="Index", ylabel="Value"):

    plt.figure(facecolor='white')
    plt.plot(column_vector, marker='', linestyle='-', markersize=5, color='b')  # Courbe avec des points
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.grid(True)
    plt.savefig(output_path)

def extract_column_from_radon(radon_transform, column_index):
    
    if column_index < 0 or column_index >= radon_transform.shape[1]:
        raise ValueError("Column index {column_index} is out of bounds for the given Radon transform matrix.")
    
    return radon_transform[:, column_index]


def main(argv):
    outputPath=""
    imgs=["","",""]
    imgs[0]=argv[0]

    imgs[1]=argv[1]
    imgs[2]=argv[2]
    
    outputPath=argv[3]
    #imgs[0]="/volWork/these/source/TLDDC/test_projection/cerf/cerf.png"
    #imgs[1]="/volWork/these/source/VT/trunkdefectclassification/data/VT/trainingdata/data/beech1/beech1-22-ci_s.pgm"
    #outputPath="/volWork/these/source/TLDDC/LIP/"
    #imgs=["beech1-22-ci_m_symHO.pgm"]
    #ANGLE range to compute radon img
    ANGLE=180.
    #Number of projection used to compute feature
    m=180
    #THETA=range(0,ANGLE,int(ANGLE/m))
    THETA=np.arange(0.0,ANGLE,ANGLE/m)
    
    LIP=[[],[],[]]
    LIP1=[[],[],[]]
    LIP2=[[],[],[]]
    LIP3=[[],[],[]]
    LIP4=[[],[],[]]
    LIP5=[[],[],[]]
    OM=[[],[],[]]
    
    cmap = get_cmap(60)
    
    #Add a signature for all image.
    for t in range(len(imgs)):
        img = cv2.imread(imgs[t], 0)
        print("image shape : ",img.shape)
        
        #Radon img
        randon_img = radon(img,THETA,False)
        #test=extract_column_from_radon(randon_img, 70)
        #plot_column_vector(test)
        
        #print("salut")
        #io.imsave(""+imgs[t]+"radon_CV.png",randon_img)
        #print("radon shape : ",randon_img.shape)
        
        for i in range(m):
            #extract feature from current profil
            f0,f1,f2,f3,f4,f5=getFeaturesByProfil(randon_img[:,i])
            
            LIP[t].append(f0)
            LIP1[t].append(f1)
            LIP2[t].append(f2)
            LIP3[t].append(f3)
            LIP4[t].append(f4)
            LIP5[t].append(f5)
        
        #normalize signature : shift
        do,sdo,minId,_=max_min(LIP[t])
        orientation_merits=1-(math.exp(1-sdo))
        OM[t].append(orientation_merits)
        
        LIP[t]=applyShift(LIP[t],do)
        LIP1[t]=applyShift(LIP1[t],do)
        LIP2[t]=applyShift(LIP2[t],do)
        LIP3[t]=applyShift(LIP3[t],do)
        LIP4[t]=applyShift(LIP4[t],do)
        LIP5[t]=applyShift(LIP5[t],do)
        #inverse if min on the right
        if(minId<ANGLE//2):
            LIP[t]=LIP[t][::-1]
            LIP1[t]=LIP1[t][::-1]
            LIP2[t]=LIP2[t][::-1]
            LIP3[t]=LIP3[t][::-1]
            LIP4[t]=LIP4[t][::-1]
            LIP5[t]=LIP5[t][::-1]
    #save LIP
    basename = os.path.basename(imgs[0])
    my_df_lip_m = pd.DataFrame([LIP[0],LIP1[0],LIP2[0],LIP3[0],LIP4[0],LIP5[0]])
    my_df_lip_m=my_df_lip_m.T
    my_df_lip_m.to_csv(outputPath+basename[:-6]+'_m.csv',header = False, index= False)
    
    basename = os.path.basename(imgs[1])
    my_df_lip_s = pd.DataFrame([LIP[1],LIP1[1],LIP2[1],LIP3[1],LIP4[1],LIP5[1]])
    my_df_lip_s=my_df_lip_s.T
    my_df_lip_s.to_csv(outputPath+basename[:-6]+'_s.csv',header = False, index= False)

    basename = os.path.basename(imgs[2])
    my_df_lip_s = pd.DataFrame([LIP[2],LIP1[2],LIP2[2],LIP3[2],LIP4[2],LIP5[2]])
    my_df_lip_s=my_df_lip_s.T
    my_df_lip_s.to_csv(outputPath+basename[:-6]+'_t.csv',header = False, index= False)
    
    #create graphique for main profile
    _,ax=plt.subplots()
    ax.plot(range(m),LIP[0],c=cmap(0),label ='LIP0', markersize=0.50,marker='.')
    ax.plot(range(m),LIP1[0],c=cmap(10),label ='LIP1', markersize=0.50,marker='o')
    ax.plot(range(m),LIP2[0],c=cmap(20),label ='LIP2', markersize=0.50,marker='s')
    ax.plot(range(m),LIP3[0],c=cmap(30),label ='LIP3', markersize=0.50,marker='p')
    ax.plot(range(m),LIP4[0],c=cmap(40),label ='LIP4', markersize=0.50,marker='*')
    ax.plot(range(m),LIP5[0],c=cmap(50),label ='LIP5', markersize=0.50,marker='+')
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    # save
    basename = os.path.basename(imgs[0])
    plt.savefig(outputPath+basename[:-6]+"_m_visu.png")
    
    #create graphique for second profile 
    # clear
    plt.clf()
    _,ax=plt.subplots()
    ax.plot(range(m),LIP[1],c=cmap(0),label ='LIP0', markersize=0.50,marker='.')
    ax.plot(range(m),LIP1[1],c=cmap(10),label ='LIP1', markersize=0.50,marker='o')
    ax.plot(range(m),LIP2[1],c=cmap(20),label ='LIP2', markersize=0.50,marker='s')
    ax.plot(range(m),LIP3[1],c=cmap(30),label ='LIP3', markersize=0.50,marker='p')
    ax.plot(range(m),LIP4[1],c=cmap(40),label ='LIP4', markersize=0.50,marker='*')
    ax.plot(range(m),LIP5[1],c=cmap(50),label ='LIP5', markersize=0.50,marker='+')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    basename = os.path.basename(imgs[1])
    plt.savefig(outputPath+basename[:-6]+"_s_visu.png")

    #create graphique for third profile 
    # clear
    plt.clf()
    _,ax=plt.subplots()
    ax.plot(range(m),LIP[2],c=cmap(0),label ='LIP0', markersize=0.50,marker='.')
    ax.plot(range(m),LIP1[2],c=cmap(10),label ='LIP1', markersize=0.50,marker='o')
    ax.plot(range(m),LIP2[2],c=cmap(20),label ='LIP2', markersize=0.50,marker='s')
    ax.plot(range(m),LIP3[2],c=cmap(30),label ='LIP3', markersize=0.50,marker='p')
    ax.plot(range(m),LIP4[2],c=cmap(40),label ='LIP4', markersize=0.50,marker='*')
    ax.plot(range(m),LIP5[2],c=cmap(50),label ='LIP5', markersize=0.50,marker='+')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    basename = os.path.basename(imgs[2])
    plt.savefig(outputPath+basename[:-6]+"_t_visu.png")

    

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
