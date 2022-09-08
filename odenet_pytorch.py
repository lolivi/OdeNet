import torch
import torch.nn as nn #neural network
import torch.optim as optim
import torch.nn.functional as F #for activation function
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchmetrics
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import numpy as np
from scipy.stats import norm
import os,math,sys,time,datetime,openpyxl
from siren_pytorch import SirenNet

userdir = "/mnt/c/Users/leona/"
if(not os.path.exists(userdir)): userdir = "/media/leonardo/HDD/"
if(not os.path.exists(userdir)): userdir = "/content/drive/My Drive/"
seed = 42

drawopt = True #disegno dataset
shiftpix = True #shifta tra 0 e pixelcloud
medianopt = False #ricostruisce video / mediana
normalization = True #utilizza sigmoide e dati normalizzati
bkgopt = True #addestra solo quella delle meteore
parxin = True #addestra posizione iniziale
plotbaseline = True #disegna anche angolo e velocita baseline
losstype = 2 #l1 o l2

iani = 0 #serve per gif
h_iss = 420 #km
h_met = 100 #km
pixelkm = (h_iss-h_met)*math.tan(math.radians(22.))/24.
cutrf = 0.78
cutcnn = 0.9
dist_sd = 24.117 #km
newspeed = (h_iss-h_met)/dist_sd
gtusec = 0.04096 #sec

#hyperparameters
dim_in, dim_out = 2,1 #input pixel2d, output cnts/pixel/gtu

#8 layers (come paper ma non funziona)
#hidden_size, n_fourier = 128,6 #neuroni e numero di mappe fourier
#n_layers = 8 #n. layer

#2 layers
#hidden_size, n_fourier = 128,0 #neuroni e numero di mappe fourier 
#n_layers = 2 #n. layer

#3 layers
hidden_size, n_fourier = 15,0 #neuroni e numero di mappe fourier 
n_layers = 3 #n. layer

skipopt = True #implementa skip connection

#coordinati globali di fondo o di meteora
class OdeNetShort(nn.Module): #inerita da nn.Module

    #quella che funziona è 0.001,1, stackCNN, normalization, hiddensize = 15 e 0 nfourier,senza il sigmoide nei singoli e senza sigmoide nella somma e con background a mediana
    #e con parxin = True

    def __init__(self,dim_in,dim_out,dim_hidden,dim_fourier): #costruttore 

        super(OdeNetShort,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.dim_fourier = dim_fourier
        self.skipopt = skipopt #True utilizza la skip connection al layer 4

        #operazione affina y = Wx + b
        self.fc1 = nn.Linear(2+4*self.dim_fourier,self.dim_hidden) 
        self.fc2 = nn.Linear(self.dim_hidden,self.dim_hidden)   
        self.fc3 = nn.Linear(self.dim_hidden,self.dim_out)  

    def forward(self,x): #da input a output

        x,nff = self.fourier(x) #mappo x in coordinate fourier e restituisco dimensione
        x = x.view(1,nff) #reshape in (1 x dimensione fourier)

        #Layer denso lineare -> ReLu
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)

        return x

    def fourier(self,x):
        x2d = x #pixel originale
        nff = 2 #dimensione iniziale 2

        for f in range(self.dim_fourier): 
            nff = nff + 4
            x_proj = pow(2,f)*x2d
            x = torch.cat([x,torch.sin(x_proj), torch.cos(x_proj)], dim=-1) #aggiorno x con tutte le mappe di fourier

        return x,nff 

#coordinati globali di fondo o di meteora
class OdeNetLogReg(nn.Module): #inerita da nn.Module

    #quella che funziona è 0.001,0.99, stackCNN, normalization, hiddensize = 128 e 0 nfourier,senza il sigmoide nei singoli ma solo nella somma

    def __init__(self,dim_in,dim_out,dim_hidden,dim_fourier): #costruttore 

        super(OdeNetShort,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.dim_fourier = dim_fourier
        self.skipopt = skipopt #True utilizza la skip connection al layer 4

        #operazione affina y = Wx + b
        self.fc1 = nn.Linear(2+4*self.dim_fourier,self.dim_hidden) 
        self.fc2 = nn.Linear(self.dim_hidden,self.dim_out)   

    def forward(self,x): #da input a output

        x,nff = self.fourier(x) #mappo x in coordinate fourier e restituisco dimensione
        x = x.view(1,nff) #reshape in (1 x dimensione fourier)

        #Layer denso lineare -> ReLu
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)

        return x

    def fourier(self,x):
        x2d = x #pixel originale
        nff = 2 #dimensione iniziale 2

        for f in range(self.dim_fourier): 
            nff = nff + 4
            x_proj = pow(2,f)*x2d
            x = torch.cat([x,torch.sin(x_proj), torch.cos(x_proj)], dim=-1) #aggiorno x con tutte le mappe di fourier

        return x,nff 

#coordinati globali di fondo o di meteora
class OdeNet(nn.Module): #inerita da nn.Module

    def __init__(self,dim_in,dim_out,dim_hidden,dim_fourier,skipopt = False): #costruttore 

        super(OdeNet,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.dim_fourier = dim_fourier
        self.skipopt = skipopt #True utilizza la skip connection al layer 4

        #operazione affina y = Wx + b
        self.fc1 = nn.Linear(2+4*self.dim_fourier,self.dim_hidden) 
        self.fc2 = nn.Linear(self.dim_hidden,self.dim_hidden) 
        self.fc3 = nn.Linear(self.dim_hidden,self.dim_hidden) 
        if (not self.skipopt): self.fc4 = nn.Linear(self.dim_hidden,self.dim_hidden)
        if (self.skipopt): self.fc4 = nn.Linear(2+4*self.dim_fourier,self.dim_hidden)
        self.fc5 = nn.Linear(self.dim_hidden,self.dim_hidden) 
        self.fc6 = nn.Linear(self.dim_hidden,self.dim_hidden) 
        self.fc7 = nn.Linear(self.dim_hidden,self.dim_hidden) 
        self.fc8 = nn.Linear(self.dim_hidden,self.dim_out)   

    def forward(self,x): #da input a output

        x,nff = self.fourier(x) #mappo x in coordinate fourier e restituisco dimensione
        x = x.view(1,nff) #reshape in (1 x dimensione fourier)

        #salvo dato iniziale per skip
        if (self.skipopt): 
            xinit = x
            xskip = F.relu(self.fc4(xinit))

        #Layer denso lineare -> ReLu
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        if (not self.skipopt): x = F.relu(self.fc4(x))
        if (self.skipopt): x = x + xskip
        x = F.relu(self.fc5(x)) 
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)

        return x

    def fourier(self,x):
        x2d = x #pixel originale
        nff = 2 #dimensione iniziale 2

        for f in range(self.dim_fourier): 
            nff = nff + 4
            x_proj = pow(2,f)*x2d
            x = torch.cat([x,torch.sin(x_proj), torch.cos(x_proj)], dim=-1) #aggiorno x con tutte le mappe di fourier

        return x,nff 

#modello totale con equazioni differenziali
class FullOdeNet(nn.Module): #inerita da nn.Module

    def __init__(self,dim_in,dim_out,dim_hidden,dim_fourier,x_in,baseth,basespeed,skipopt = False,training = True): #costruttore 

        super(FullOdeNet,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.dim_fourier = dim_fourier
        self.skipopt = skipopt

        if (training and parxin): self.x_in = nn.Parameter(x_in)
        else: self.x_in = x_in

        if (baseth): 
            self.basespeed = basespeed
            self.baseth = baseth
            speed = torch.Tensor([self.basespeed*math.cos(math.radians(self.baseth)),math.sin(self.basespeed*math.radians(self.baseth))]) #inizializzo  le velocità 

        if (not baseth): speed = torch.Tensor([0,0]) #inizializzo  le velocità 

        speed = speed.view(1,2)
        if (training): self.speed = nn.Parameter(speed) #è importante che siano parametri -> learnable
        else: self.speed = speed

        #layer globali e locali meteora
        if (n_layers==8):
            self.globalode = OdeNet(self.dim_in,self.dim_out,self.dim_hidden,self.dim_fourier,self.skipopt)
            self.localode = OdeNet(self.dim_in,self.dim_out,self.dim_hidden,self.dim_fourier,self.skipopt)

        if (n_layers==3):
            self.globalode = OdeNetShort(self.dim_in,self.dim_out,self.dim_hidden,self.dim_fourier)
            self.localode = OdeNetShort(self.dim_in,self.dim_out,self.dim_hidden,self.dim_fourier)

        if (n_layers==2):
            self.globalode = OdeNetLogReg(self.dim_in,self.dim_out,self.dim_hidden,self.dim_fourier)
            self.localode = OdeNetLogReg(self.dim_in,self.dim_out,self.dim_hidden,self.dim_fourier)

    def forward(self,x,t,bkg): #da input a output

        #transformazione globale
        pix2d = x #mi serve per salvarla 
        if (not bkg): bkgcounts = self.globalode(pix2d) #prima passo in quella di fondo
        if (bkg): bkgcounts = bkg
        
        #transformazione locale
        localpix = self.timetransform(pix2d,t)
        metcounts = self.localode(localpix) #poi in quella 

        totcounts = metcounts + bkgcounts

        #sommo output 1 con output 2
        if (normalization and n_layers==2): totcounts = torch.sigmoid(totcounts)
        return totcounts 

    def timetransform(self,x,t): #transformazione con equazione differenziale
        x = x - self.x_in - self.speed*t
        return x

    def setspeed(self,vx,vy): #set parameter

        speed = torch.Tensor([vx,vy]) #inizializzo  le velocità
        speed = speed.view(1,2)
        self.speed = nn.Parameter(speed) #è importante che siano parametri -> learnable

    def getbkgcnts(self,x,t,bkg): 
        
        #transformazione globale
        pix2d = x #mi serve per salvarla 
        if (not bkg): bkgcounts = self.globalode(pix2d) #prima passo in quella di fondo
        if (bkg): bkgcounts = bkg
        return bkgcounts

    def getlocalcnts(self,x,t): 
        
        #transformazione locale
        pix2d = x #mi serve per salvarla 
        localpix = self.timetransform(pix2d,t)
        metcounts = self.localode(localpix) #poi in quella 
        return metcounts

class SirenOdeNet(nn.Module): #inerita da nn.Module

    def __init__(self,dim_in,dim_out,dim_hidden,n_layers,x_in,baseth,basespeed,training = True): #costruttore 

        super(SirenOdeNet,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers

        if (training and parxin): self.x_in = nn.Parameter(x_in)
        else: self.x_in = x_in

        if (baseth): 
            self.basespeed = basespeed
            self.baseth = baseth
            speed = torch.Tensor([self.basespeed*math.cos(math.radians(self.baseth)),math.sin(self.basespeed*math.radians(self.baseth))]) #inizializzo  le velocità 

        if (not baseth): speed = torch.Tensor([0,0]) #inizializzo  le velocità 

        speed = speed.view(1,2)
        if (training): self.speed = nn.Parameter(speed) #è importante che siano parametri -> learnable
        else: self.speed = speed

        #layer globali e locali meteora
        self.globalode = SirenNet(self.dim_in,self.dim_hidden,self.dim_out,self.n_layers,final_activation = nn.Identity())
        self.localode = SirenNet(self.dim_in,self.dim_hidden,self.dim_out,self.n_layers,final_activation = nn.Identity())

    def forward(self,x,t,bkg): #da input a output

        #transformazione globale
        pix2d = x #mi serve per salvarla 
        if (not bkg): bkgcounts = self.globalode(pix2d) #prima passo in quella di fondo
        if (bkg): bkgcounts = bkg
        
        #transformazione locale
        localpix = self.timetransform(pix2d,t)
        metcounts = self.localode(localpix) #poi in quella 

        totcounts = metcounts + bkgcounts

        #sommo output 1 con output 2
        #if (normalization): totcounts = torch.sigmoid(totcounts)
        return totcounts 

    def timetransform(self,x,t): #transformazione con equazione differenziale
        x = x - self.x_in - self.speed*t
        return x

    def setspeed(self,vx,vy): #set parameter

        speed = torch.Tensor([vx,vy]) #inizializzo  le velocità
        speed = speed.view(1,2)
        self.speed = nn.Parameter(speed) #è importante che siano parametri -> learnable

    def getbkgcnts(self,x,t,bkg): 
        
        #transformazione globale
        pix2d = x #mi serve per salvarla 
        if (not bkg): bkgcounts = self.globalode(pix2d) #prima passo in quella di fondo
        if (bkg): bkgcounts = bkg
        return bkgcounts

    def getlocalcnts(self,x,t): 
        
        #transformazione locale
        pix2d = x #mi serve per salvarla 
        localpix = self.timetransform(pix2d,t)
        metcounts = self.localode(localpix) #poi in quella 
        return metcounts

#Funzioni Extra
def buildstring(amag,index): #costruisce etichetta file anche con i 0 (compatibile con Dario)

    if (index<10): filename = "amag_p%i0_000%i" % (amag,index)
    if (index>=10 and index<100): filename = "amag_p%i0_00%i" % (amag,index)
    if (index==100): filename = "amag_p%i0_0%i" % (amag,index)

    return filename

#legge dati simulati
def read_txt_data(amag,idx):

    txt = userdir + "Documenti/ETOS/Dati/SIM/list_amag_p%i0_new.txt" % (amag)
    linesplit_data = []

    if not os.path.isfile(txt): sys.exit()

    with open(txt) as f1:
        lines = f1.readlines()
        for iline,line in enumerate(lines):

            if (iline==0): continue #header file
            linesplit = line.split()
            if (not linesplit): continue
            if (int(linesplit[0])!=idx): continue
            for l in linesplit: linesplit_data.append(float(l))
            return linesplit_data

#animazione del video simulato e di quello ricostruito
def plot_video(amag,index,gtupix,hyperpars,modelvariables,stackvariables,basevariables,tuning):

    pixelcloud, ngtu = hyperpars[0],hyperpars[1]
    modeldir,modelname = hyperpars[2],hyperpars[3]

    datatot = modelvariables[0]
    datalocal,databkg = modelvariables[1],modelvariables[2]
    xmodel,ymodel = modelvariables[3],modelvariables[4]
    speedmodel,thetamodel = modelvariables[5],modelvariables[6]

    xpix,ypix = stackvariables[0],stackvariables[1]
    stackv,stackth = stackvariables[2],stackvariables[3]

    basexin,baseyin = basevariables[0],basevariables[1]
    basespeed,basetheta = basevariables[2],basevariables[3]

    filestring = buildstring(amag,index)
    if (tuning): pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Tuning/%s/" % (filestring)
    if (not tuning): pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Training/%s/" % (filestring)
    if(not os.path.exists(pngdir)): os.makedirs(pngdir)

    inputtext = userdir + 'Documenti/ETOS/Dati/SIM/%s.txt' % (filestring)
    if os.path.isfile(inputtext): a = np.loadtxt(inputtext) #crea array di numpy con dati in ordine
    else: sys.exit() 

    back = np.reshape(a,(128,48,48)) #fa un reshape dei dati nell'array a con lunghezza array nel primo, righe e colonne
    backmedian = np.zeros((128,48,48))
    for entry in range(128): back[entry,:,:] = np.rot90(back[entry,:,:],1) #i dati sono ruotati -> così è come su ETOS

    if (medianopt):
        bkgmap = np.median(back[:,0:48,0:48], axis=0) 
        for entry in range(128): backmedian[entry,:,:] = back[entry,:,:] / bkgmap 

    txt_data = read_txt_data(amag,index)
    xpixin,xpixfin = txt_data[28],txt_data[29]
    ypixin,ypixfin = txt_data[30],txt_data[31]
    realth,realv_pix = txt_data[32],txt_data[33] #in deg e pix/gtu
    realv = txt_data[5]*math.cos(math.radians(txt_data[13])) #in km/s

    dx = xpixfin - xpixin
    dy = ypixfin - ypixin
    dxreal = xpixfin-xpix
    dyreal = ypixfin-ypix

    #range plot
    xpixmin = max(xpix-pixelcloud,0)
    ypixmin = max(ypix-pixelcloud,0)
    xpixmax = min(xpix+pixelcloud,47)
    ypixmax = min(ypix+pixelcloud,47)

    if (xpixmin==0): xpixmax = pixelcloud*2 
    if (ypixmin==0): ypixmax = pixelcloud*2
    if (xpixmax==47): xpixmin = 47 - pixelcloud*2
    if (ypixmax==47): ypixmin = 47 - pixelcloud*2

    pixel2d = np.zeros((ngtu,48,48))
    for igtu,gtu in enumerate(range(gtupix,gtupix+ngtu)): 
        if (medianopt): pixel2d[igtu,:,:] = backmedian[gtu,:,:]
        if (not medianopt): pixel2d[igtu,:,:] = back[gtu,:,:]
    minvalue,maxvalue = np.min(pixel2d[:,47-ypixmax:48-ypixmin,xpixmin:xpixmax+1]),np.max(pixel2d[:,47-ypixmax:48-ypixmin,xpixmin:xpixmax+1])
    if (normalization): pixel2d = (pixel2d-minvalue)/(maxvalue-minvalue)

    #video simulato
    f = plt.figure()
    plt.xlabel('pix X')
    plt.ylabel('pix Y')
    plt.xlim(xpixmin,xpixmax+1)
    plt.ylim(ypixmin,ypixmax+1)
    labelsim = "Simulation - v = %.2f km/s, %s = %.0f °" % (realv,r"$\theta$",realth)
    plt.arrow(xpixin+0.5,ypixin+0.5,dx,dy,head_width=0.7, head_length=0.7, length_includes_head=True, color="y", label = labelsim)
    global iani
    iani = 0
    im = plt.imshow(pixel2d[0,:,:],animated=True,cmap="hot",extent=(0,48,0,48))
    if (not normalization): plt.clim(0.9*minvalue,1.1*maxvalue)
    if (normalization): plt.clim(0,1)
    plt.colorbar()
    plt.legend(loc="best")

    def updatefig(*args):
        global iani
        iani = iani + 1
        if (iani>ngtu-1): iani = 0
        titlevideo = " Simulated (M,idx) = (%i,%i) \n pix = (%i,%i), GTU = %i"  % (amag,index,xpixin,ypixin,gtupix+iani)
        plt.title(titlevideo)
        im.set_array(pixel2d[iani,:,:])
        if (not normalization): plt.clim(0.9*minvalue,1.1*maxvalue)
        if (normalization): plt.clim(0,1)
        return im,

    ani = animation.FuncAnimation(f, updatefig, blit=True)
    ani.save(pngdir + "etos_video_%i_%i.gif" % (gtupix,gtupix+ngtu-1), writer='pillow', fps=1.5)
    f.clear()
    plt.close(f)

    if (tuning): pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Tuning/%s/%s/" % (filestring,modeldir)
    if(not os.path.exists(pngdir)): os.makedirs(pngdir)

    for idata in range(3): 

        if (idata==0): data = datatot
        if (idata==1): data = databkg
        if (idata==2): data = datalocal

        minvalue,maxvalue = np.min(data),np.max(data)

        #figura ricostruita
        f = plt.figure()
        plt.xlabel('pix X')
        plt.ylabel('pix Y')
        labelsim = "Simulation - v = %.2f pix/GTU, %s = %.0f °" % (realv_pix,r"$\theta$",realth)
        if (idata==0): titlemodel = " %s - " % (modelname)
        if (idata==1): titlemodel = " %s - Background - " % (modelname)
        if (idata==2): titlemodel = " %s - Local - " % (modelname)
        labelmodel = "Model - v = %.2f pix/GTU, %s = %.0f °" % (speedmodel,r"$\theta$",thetamodel)
        labelstack = "Stack-CNN - v = %.2f pix/GTU, %s = %.0f °" % (stackv/pixelkm*gtusec,r"$\theta$",stackth)
        if (plotbaseline): labelbase = "Baseline - v = %.2f pix/GTU, %s = %.0f °" % (basespeed,r"$\theta$",basetheta)
        if (shiftpix): 
            plt.arrow(xpixin+0.5-xpixmin,ypixin+0.5-ypixmin,dx,dy,head_width=0.7, head_length=0.7, length_includes_head=True, color="y", label = labelsim)
            plt.arrow(xpix+0.5-xpixmin,ypix+0.5-ypixmin,stackv/pixelkm*gtusec*math.cos(math.radians(stackth))*ngtu,stackv/pixelkm*gtusec*math.sin(math.radians(stackth))*ngtu,head_width=0.7, head_length=0.7, length_includes_head=True, color = "magenta", label = labelstack)
            if (plotbaseline): plt.arrow(basexin+0.5-xpixmin,baseyin+0.5-ypixmin,basespeed*math.cos(math.radians(basetheta))*ngtu,basespeed*math.sin(math.radians(basetheta))*ngtu,head_width=0.7, head_length=0.7, length_includes_head=True, color="g", label = labelbase)
        if (not shiftpix): 
            plt.arrow(xpixin+0.5,ypixin+0.5,dx,dy,head_width=0.7, head_length=0.7, length_includes_head=True, color="y", label = labelsim)
            plt.arrow(xpix+0.5,ypix+0.5,stackv/pixelkm*gtusec*math.cos(math.radians(stackth))*ngtu,stackv/pixelkm*gtusec*math.sin(math.radians(stackth))*ngtu,head_width=0.7, head_length=0.7, length_includes_head=True, color = "magenta", label = labelstack)
            if (plotbaseline): plt.arrow(basexin+0.5,baseyin+0.5,basespeed*math.cos(math.radians(basetheta))*ngtu,basespeed*math.sin(math.radians(basetheta))*ngtu,head_width=0.7, head_length=0.7, length_includes_head=True, color = "g", label = labelbase)
        plt.arrow(xmodel+0.5,ymodel+0.5,speedmodel*math.cos(math.radians(thetamodel))*ngtu,speedmodel*math.sin(math.radians(thetamodel))*ngtu,head_width=0.7, head_length=0.7, length_includes_head=True, color="b", label = labelmodel)
        iani = 0
        if (shiftpix): im = plt.imshow(data[0,:,:],animated=True,cmap="hot",extent=(0,pixelcloud*2+1,0,pixelcloud*2+1))
        else: im = plt.imshow(data[0,:,:],animated=True,cmap="hot",extent=(xpixmin,xpixmax+1,ypixmin,ypixmax+1))
        plt.clim(0.9*minvalue,1.1*maxvalue)
        plt.colorbar()
        plt.legend(loc="best")

        def updatefig(*args):
            global iani
            iani = iani + 1
            if (iani>ngtu-1): iani = 0
            plt.title(titlemodel + "max = (%i,%i), GTU = %i" % (xmodel,ymodel,gtupix+iani))
            im.set_array(data[iani,:,:])
            plt.clim(0.9*minvalue,1.1*maxvalue)
            return im,

        ani = animation.FuncAnimation(f, updatefig, blit=True)
        ani.save(pngdir + "rec_video_%i.gif" % (idata), writer='pillow', fps=1.5)
        f.clear()
        plt.close(f)

#training
def train_net(magmet,idxmet,hyperpars,stackvariables,init,tuning):

    netname, n_epochs = hyperpars[0],hyperpars[1]
    learningrate,exprate = hyperpars[2],hyperpars[3]
    pixelcloud, ngtu = hyperpars[4],hyperpars[5]

    xmet,ymet,gtumet = stackvariables[0],stackvariables[1],stackvariables[2]
    stackv,stacktheta = stackvariables[3],stackvariables[4]

    #seed
    torch.manual_seed(seed) #fisso seed

    #prendo veri parametri simulati
    txt_data = read_txt_data(magmet,idxmet)
    xpixin,xpixfin = txt_data[28],txt_data[29]
    ypixin,ypixfin = txt_data[30],txt_data[31]
    realth,realv_pix = txt_data[32],txt_data[33] #in deg e pix/gtu
    realv = txt_data[5]*math.cos(math.radians(txt_data[13])) #in km/s

    #leggo dati veri simulati
    filename = buildstring(magmet,idxmet)
    inputtext = userdir + 'Documenti/ETOS/Dati/SIM/%s.txt' % (filename)
    if os.path.isfile(inputtext):
        a = np.loadtxt(inputtext) #crea array di numpy con dati in ordine
        #print("Leggendo il file %s" %(inputtext))
    else: sys.exit() 

    back = np.reshape(a,(128,48,48)) #fa un reshape dei dati nell'array a con lunghezza array nel primo, righe e colonne
    backmedian = np.zeros((128,48,48))
    for entry in range(128): back[entry,:,:] = np.rot90(back[entry,:,:],1) #i dati sono ruotati -> così è come su ETOS

    if (medianopt):
        bkgmap = np.median(back[:,0:48,0:48], axis=0) 
        for entry in range(128): backmedian[entry,:,:] = back[entry,:,:] / bkgmap 

    #range pixel
    xpixmin = max(xmet-pixelcloud,0)
    ypixmin = max(ymet-pixelcloud,0)
    xpixmax = min(xmet+pixelcloud,47)
    ypixmax = min(ymet+pixelcloud,47)

    if (xpixmin==0): xpixmax = pixelcloud*2 
    if (ypixmin==0): ypixmax = pixelcloud*2
    if (xpixmax==47): xpixmin = 47 - pixelcloud*2
    if (ypixmax==47): ypixmin = 47 - pixelcloud*2

    #angolo di baseline -> massimo iniziale e finale
    mettrack = []
    for gtuiter in range(127):
        maxiter = np.max(back[gtuiter,:,:])
        maxpositer = np.where(back[gtuiter,:,:] == maxiter)
        maxiternext = np.max(back[gtuiter+1,:,:])
        maxpositernext = np.where(back[gtuiter+1,:,:] == maxiternext)
        xpositer,ypositer = maxpositer[1][0],47-maxpositer[0][0]
        xpositernext,ypositernext = maxpositernext[1][0],47-maxpositernext[0][0]
        if not(xpositer>=xpixmin and xpositer<=xpixmax and ypositer>=ypixmin and ypositer<=ypixmax): continue
        if not(xpositernext>=xpixmin and xpositernext<=xpixmax and ypositernext>=ypixmin and ypositernext<=ypixmax): continue
        distanceiter = math.sqrt((xpositer-xpositernext)**2 + (ypositer-ypositernext)**2)
        if (distanceiter<=1): 
            if ([gtuiter,xpositer,ypositer] not in mettrack): mettrack.append([gtuiter,xpositer,ypositer])
            if ([gtuiter+1,xpositernext,ypositernext] not in mettrack): mettrack.append([gtuiter+1,xpositernext,ypositernext])

    #angolo di baseline senza reti
    basexin, basexfin = mettrack[0][1], mettrack[-1][1]
    baseyin, baseyfin = mettrack[0][2], mettrack[-1][2]
    gtuin = mettrack[0][0]
    basetheta = math.degrees(math.atan2(baseyfin-baseyin,basexfin - basexin))
    if (basetheta<0): basetheta = 360 + basetheta
    basespeed = math.sqrt((basexfin - basexin)**2 + (baseyfin-baseyin)**2) / len(mettrack)

    #plot dati reali
    metvideo = np.zeros((ngtu,pixelcloud*2+1,pixelcloud*2+1))
    for igtu,gtu in enumerate(range(gtuin,gtuin+ngtu)): 
        if (medianopt): metvideo[igtu,:,:] = backmedian[gtu,47-ypixmax:48-ypixmin,xpixmin:xpixmax+1]
        if (not medianopt): metvideo[igtu,:,:] = back[gtu,47-ypixmax:48-ypixmin,xpixmin:xpixmax+1]
    if (normalization): metvideo = (metvideo-np.min(metvideo))/(np.max(metvideo)-np.min(metvideo))
    if (bkgopt): bkg = np.median(metvideo)
    if (not bkgopt): bkg = None

    #inizializzazione parametri
    if (shiftpix): x_in = torch.Tensor([basexin-xpixmin,baseyin-ypixmin]) #lo sposto a 0 e pixelcloud
    else: x_in = torch.Tensor([basexin,baseyin])
    x_in = x_in.view(1,dim_in)

    #inizializzazione modello
    if (netname=="FullOde" and init == 1): net = FullOdeNet(dim_in,dim_out,hidden_size,n_fourier,x_in,stacktheta,stackv/pixelkm*gtusec,skipopt,True)
    if (netname=="SirenOde" and init == 1): net = SirenOdeNet(dim_in,dim_out,hidden_size,n_layers,x_in,stacktheta,stackv/pixelkm*gtusec,True)
    if (netname=="FullOde" and init == 2): net = FullOdeNet(dim_in,dim_out,hidden_size,n_fourier,x_in,basetheta,basespeed,skipopt,True)
    if (netname=="SirenOde" and init == 2): net = SirenOdeNet(dim_in,dim_out,hidden_size,n_layers,x_in,basetheta,basespeed,True)
    if (netname=="FullOde" and init == 0): net = FullOdeNet(dim_in,dim_out,hidden_size,n_fourier,x_in,None,None,skipopt,True)
    if (netname=="SirenOde" and init == 0): net = SirenOdeNet(dim_in,dim_out,hidden_size,n_layers,x_in,None,None,True)

    if (len(learningrate)==1):
        modeldir = netname + "_Adam_%.5f_%.2f_%i_%i" % (learningrate[0],exprate,pixelcloud,ngtu)
        modelname = netname + " - Adam (lr = %.5f, dr = %.2f) \n(%ix%ix%i)" % (learningrate[0],exprate,pixelcloud,pixelcloud,ngtu)
        optimizer = optim.Adam(net.parameters(),lr = learningrate[0])

    if (len(learningrate)==2):
        modeldir = netname + "_Adam_%.5f_%.5f_%.2f_%i_%i" % (learningrate[0],learningrate[1],exprate,pixelcloud,ngtu)
        modelname = netname + " - Adam (lr1 = %.5f, lr2 = %.5f, dr = %.2f) \n(%ix%ix%i)" % (learningrate[0],learningrate[1],exprate,pixelcloud,pixelcloud,ngtu)
        
        parnetlist = nn.ParameterList([par for name,par in net.named_parameters() if (par.requires_grad and name!="speed" and name!="x_in")])
        parphyslist = nn.ParameterList([par for name,par in net.named_parameters() if (par.requires_grad and (name=="speed" or name=="x_in"))])

        optimizer = optim.Adam([
            {'params': parnetlist},
            {'params': parphyslist, 'lr': learningrate[1]}
        ], lr=learningrate[0])

    if (len(learningrate)==3):
        modeldir = netname + "_Adam_%.5f_%.5f_%.5f_%.2f_%i_%i" % (learningrate[0],learningrate[1],learningrate[2],exprate,pixelcloud,ngtu)
        modelname = netname + " - Adam (lr1 = %.5f, lr2 = %.5f, lr3 = %.5f, dr = %.2f) \n(%ix%ix%i)" % (learningrate[0],learningrate[1],learningrate[2],exprate,pixelcloud,pixelcloud,ngtu)
        
        parnetlist = nn.ParameterList([par for name,par in net.named_parameters() if (par.requires_grad and name!="speed" and name!="x_in")])
        parspeedlist = nn.ParameterList([par for name,par in net.named_parameters() if (par.requires_grad and name=="speed")])
        parposlist = nn.ParameterList([par for name,par in net.named_parameters() if (par.requires_grad and name=="x_in")])

        optimizer = optim.Adam([
            {'params': parnetlist},
            {'params': parspeedlist, 'lr': learningrate[1]},
            {'params': parposlist, 'lr': learningrate[2]}
        ], lr=learningrate[0])

    print("---------------------")
    print("Modello Fisico: %s" % modelname) 

    #----------------------
    #------TRAINING--------
    #----------------------

    #funzione di loss e ottimizzatore
    if (losstype == 1): lossfunction = nn.L1Loss() #distanza in modulo
    if (losstype == 2): lossfunction = nn.MSELoss() #distanza al quadrato
    if (exprate<1): scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = exprate)

    losslist = [] #per plottare
    resthlist,resvlist = [],[] #residuo velocità e tempo
    vxlist,vylist = [],[] #vx e vy
    timelist = [] #plotto anche il tempo 
    ilosslist = -1

    print("Beginning Training!")
    net.train()
    print("Real Theta = %.2f °, Baseline = %.2f °, Stack-CNN = %.2f °" % (realth,basetheta,stacktheta))

    for e in range(n_epochs):

        lossepoch, resvepoch, resthepoch, vxepoch, vyepoch = 0.,0.,0.,0.,0.
        if (losslist and e%(n_epochs/10)==0): print("Epoch %i -> Loss %f, Res Theta %.2f °" % (e,losslist[ilosslist],resthlist[ilosslist]))
        starttime = time.time()

        #loop temporale
        for t in range(ngtu):
            #loop nell'immagine
            for x in range(xpixmin,xpixmax+1):
                for y in range(ypixmin,ypixmax+1):

                    #pixel shiftato
                    if (shiftpix): pixinput = torch.Tensor([x-xpixmin,y-ypixmin])
                    else: pixinput = torch.Tensor([x,y])
                    pixinput = pixinput.view(1,dim_in)

                    #azzero i parametri del gradiente
                    optimizer.zero_grad()

                    #forward
                    outputs = net(pixinput,t,bkg) #calcolo l'output
                    trueimage = metvideo[t,ypixmax-y,x-xpixmin] #l = 47 - y -> lmin = 47 - ymax -> l-lmin = ymax - y
                    trueimage = torch.Tensor([[trueimage]])
                    loss = lossfunction(outputs,trueimage) #calcolo loss
                    
                    #backward
                    loss.backward()

                    #optimizer
                    optimizer.step() #1 learning rate step 

                    #update list
                    lossepoch = lossepoch + loss.item()

                    #prendo residui ad ogni dato
                    for name, param in net.named_parameters(): 
                        if(param.requires_grad and name=="speed"): vx,vy = param.data[0][0],param.data[0][1]
                    vapp,theta = math.sqrt(vx*vx + vy*vy),math.degrees(math.atan2(vy,vx))
                    if (theta<0): theta = theta + 360
                    resvepoch = resvepoch + vapp-realv_pix

                    vxepoch = vxepoch + vapp*math.cos(math.radians(theta))
                    vyepoch = vyepoch + vapp*math.sin(math.radians(theta))
                    
                    restheta = theta - realth
                    if (restheta>180): restheta = restheta - 360
                    if (restheta<(-180)): restheta = 360 + restheta
                    resthepoch = resthepoch + restheta

        #learning rate viene aggiustato
        if (exprate<1): scheduler.step()

        stoptime = time.time()
        timelist.append(stoptime-starttime)

        resvepoch = resvepoch/(ngtu*(pixelcloud*2+1)*(pixelcloud*2+1))
        resthepoch = resthepoch/(ngtu*(pixelcloud*2+1)*(pixelcloud*2+1))
        vxepoch = vxepoch/(ngtu*(pixelcloud*2+1)*(pixelcloud*2+1))
        vyepoch = vyepoch/(ngtu*(pixelcloud*2+1)*(pixelcloud*2+1))

        resvlist.append(resvepoch)
        resthlist.append(resthepoch)
        vxlist.append(vxepoch)
        vylist.append(vyepoch)

        lossepoch = lossepoch/(ngtu*(pixelcloud*2+1)*(pixelcloud*2+1))
        losslist.append(lossepoch)
        ilosslist = ilosslist + 1

    print("Finished Training!")

    if (shiftpix): xmodel,ymodel = basexin-xpixmin,baseyin-ypixmin #le inizializzo in questo modo
    else: xmodel,ymodel = basexin,baseyin
    for name, param in net.named_parameters(): 
        #if(param.requires_grad and name=="speed"): vx,vy = param.data[0][0],param.data[0][1]
        if(param.requires_grad and name=="x_in"): xmodel,ymodel = int(param.data[0][0]+0.5),int(param.data[0][1]+0.5)
    #vapp,theta = math.sqrt(vx*vx + vy*vy),math.degrees(math.atan2(vy,vx))
    #if (theta<0): theta = theta + 360 #questo non è mediato nell'ultima epoch

    vapp = resvlist[-1] + realv_pix
    theta = resthlist[-1] + realth
    theta = theta - int(theta/360)*360
    if (theta<0): theta = theta + 360

    print("Posizione Iniziale = (%i,%i)" % (xmodel,ymodel))
    print("Velocità Modello = %.3f pix/GTU (%.3f,%.3f)" % (vapp,vx,vy))
    print("Direzione Modello = %.3f °" % (theta))
    print("---------------------")

    #video ricostruito
    recvideo2d = np.zeros((ngtu,pixelcloud*2+1,pixelcloud*2+1))
    recvideo2d_local = np.zeros((ngtu,pixelcloud*2+1,pixelcloud*2+1))
    recvideo2d_bkg = np.zeros((ngtu,pixelcloud*2+1,pixelcloud*2+1))

    #loop temporale
    for t in range(ngtu):

        #loop nell'immagine
        for x in range(xpixmin,xpixmax+1):
            for y in range(ypixmin,ypixmax+1):
                
                if (shiftpix): pixinput = torch.Tensor([x-xpixmin,y-ypixmin])
                else: pixinput = torch.Tensor([x,y])

                phi = net(pixinput,t,bkg)
                phi_local = net.getlocalcnts(pixinput,t)
                phi_bkg = net.getbkgcnts(pixinput,t,bkg)
                recvideo2d[t,ypixmax-y,x-xpixmin] = phi.data[0][0]
                recvideo2d_local[t,ypixmax-y,x-xpixmin] = phi_local.data[0][0]
                if (not bkgopt): recvideo2d_bkg[t,ypixmax-y,x-xpixmin] = phi_bkg.data[0][0]  
                if (bkgopt): recvideo2d_bkg[t,ypixmax-y,x-xpixmin] = phi_bkg 

    #plotting video
    hyperpars_video = [pixelcloud,ngtu,modeldir,modelname]
    modelvars_video = [recvideo2d,recvideo2d_local,recvideo2d_bkg,xmodel,ymodel,vapp,theta]
    stackvars_video = [xmet,ymet,stackv,stacktheta]
    basevars_video = [basexin,baseyin,basespeed,basetheta]
    if (drawopt): plot_video(magmet,idxmet,gtuin,hyperpars_video,modelvars_video,stackvars_video,basevars_video,tuning)

    #plotting results 
    if (tuning): pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Tuning/%s/%s/" % (filename,modeldir)
    if (not tuning): pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Training/%s/" % (filename)
    if(not os.path.exists(pngdir)): os.makedirs(pngdir)

    elist = [e for e in range(n_epochs)] #asse x comune a tutti i plot 

    #learning curve loss
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    if (losstype==1): plt.ylabel("Least Absolute Deviations [L1]")
    if (losstype==2): plt.ylabel("Mean Squared Error [MSE]")
    plt.yscale("log")
    plt.plot(elist,losslist,c = "b")
    plt.savefig(pngdir + "learningcurve_loss.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #learning curve res speeed
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Apparent Speed Residual [pix/GTU]")
    plt.axhline(stackv/pixelkm*gtusec-realv_pix,c="magenta",linestyle="dashdot",label="Stack-CNN")
    if (plotbaseline): plt.axhline(basespeed-realv_pix,c="g",linestyle="dashdot",label="Baseline")
    plt.plot(elist,resvlist,c = "b")
    plt.legend(loc="best")
    plt.savefig(pngdir + "learningcurve_speed.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #learning curve res theta
    stackresth = stacktheta-realth
    if (stackresth>180): stackresth = stackresth - 360
    if (stackresth<(-180)): stackresth = 360 + stackresth

    baseresth = basetheta - realth
    if (baseresth>180): baseresth = baseresth - 360
    if (baseresth<(-180)): baseresth = 360 + baseresth

    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Apparent Direction Residual [°]")
    plt.axhline(stackresth,c="magenta",linestyle="dashdot",label="Stack-CNN")
    if (plotbaseline): plt.axhline(baseresth,c="g",linestyle="dashdot",label="Baseline")
    plt.plot(elist,resthlist,c = "b")
    plt.legend(loc="best")
    plt.savefig(pngdir + "learningcurve_theta.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    timetot = np.sum(timelist)
    timetot = datetime.timedelta(seconds=timetot)

    #learning curve time
    f = plt.figure()
    plt.title("Training Time: %s" % (str(timetot)))
    plt.xlabel("Epoch Number")
    plt.ylabel("Training Time [s]")
    #plt.yscale("log")
    plt.plot(elist,timelist,c = "b")
    plt.savefig(pngdir + "learningcurve_time.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    return modelname,gtuin,[losslist,resvlist,resthlist,timelist],[basexin,baseyin,basespeed,basetheta]

#conversione in km
def pixel_to_km(xmax,ymax,height,cal): #prende thetax e phix da convertire
    infile = userdir + "Documenti/ETOS/Calibrazione/Angoli - Pixel.txt"

    with open(infile) as f: #lettura
        lines = f.readlines()
        for i,line in enumerate(lines): #loop su righe
            linesplit = line.split()
            if (not linesplit): continue
            if (cal == 0 and int(linesplit[0])==xmax and int(linesplit[1])==ymax): return height*math.tan(float(linesplit[2]))*math.cos(float(linesplit[3])), height*math.tan(float(linesplit[2]))*math.sin(float(linesplit[3]))
            if (cal == 1 and int(linesplit[0])==xmax and int(linesplit[1])==ymax): return height*math.tan(float(linesplit[4]))*math.cos(float(linesplit[5])), height*math.tan(float(linesplit[4]))*math.sin(float(linesplit[5]))
            if (cal == 2 and int(linesplit[0])==xmax and int(linesplit[1])==ymax): return height*math.tan(float(linesplit[6]))*math.cos(float(linesplit[7])), height*math.tan(float(linesplit[6]))*math.sin(float(linesplit[7]))
            if (cal == 3 and int(linesplit[0])==xmax and int(linesplit[1])==ymax): return height*math.tan(float(linesplit[8]))*math.cos(float(linesplit[9])), height*math.tan(float(linesplit[8]))*math.sin(float(linesplit[9]))
            if (cal == 4 and int(linesplit[0])==xmax and int(linesplit[1])==ymax): return height*math.tan(float(linesplit[10]))*math.cos(float(linesplit[11])), height*math.tan(float(linesplit[10]))*math.sin(float(linesplit[11]))

#lettura excel
def read_excel(txt_data):

    met_results = [] #risultati meteore riassunti qui
    fake_cnn_4,fake_cnn_5,fake_cnn_6 = [],[],[]
    fake_pix_4,fake_pix_5,fake_pix_6 = [],[],[]
    fake_rf_4,fake_rf_5,fake_rf_6 = [],[],[]

    #LETTURA EXCEL SECONDO TRIGGER
    excel = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Stack CNN Meteore - Immagini/SIM/Analisi SIM.xlsx"
    sheetname = "StackCNN_RF - Clean"

    if os.path.isfile(excel):
        #print("\nLeggendo il File Excel %s w/ Sheet %s" % (excel,sheetname))
        wb = openpyxl.load_workbook(excel, data_only=False) #workbook
    else:
        print("Non è Stato Trovato il File Excel %s w/ Sheet %s" % (excel,sheetname))
        sys.exit()	

    sheet = wb[sheetname]
    ncolumns = sheet.max_column
    nrows = sheet.max_row
    startrow = 3

    for i in range(startrow,nrows + 1):
        for j in range(1,ncolumns + 1):
            cellname = sheet.cell(row = 2, column = j).value #nome
            currentcell = sheet.cell(row = i, column = j).value #valore
            if(cellname =="FILE"): file = str(currentcell)
            if(cellname =="GTU"): gtu = int(currentcell)
            if(cellname =="X"): xmax = int(currentcell)
            if(cellname =="Y"): ymax = int(currentcell)
            if(cellname =="RF PROB [%]"): prf = float(currentcell)
            if(cellname =="CNN PROB [%]"): pcnn = float(currentcell)
            if(cellname =="MAX COUNTS"): cnts = float(currentcell)
            if(cellname =="V [km/s]"): vmet = float(currentcell)
            if(cellname =="THETA [°]"): thetamet = float(currentcell)
            if(cellname =="NOTES"): 
                if(currentcell): 
                    if (str(currentcell)=="Deltapix"): notes = 0
                    else: notes = 1
                else: notes = 1
        #fine loop su colonne excel

        #check se riga ha dati o no
        if (file == None): continue

        if (i==startrow): 
            fileback = file
            n_met_file = 0
            klist = []
            pcnnlist,prflist,cntslist = [],[],[]
            appvlist,appthlist = [],[]
            realvlist,realphilist,svlist,sphilist = [],[],[],[]
            xpixlist,ypixlist,gtulist = [],[],[]
            deltapixlist = [] #è 1 se il deltapix è <3 e 0 altrimenti

        if (file != fileback):

            if (n_met_file>1): print("Warning! %i Meteore @ %s" % (n_met_file,fileback))
            if (n_met_file!=0):

                #risolvo doppioni
                maxcnts,idxmax = 0,-1 #cerco tra quelli con deltapix < 3 quello piu intenso
                for itest,cnttest in enumerate(cntslist):
                    if (deltapixlist[itest]==0): continue #sarebbe soppressa perché deltapix è >2
                    if (cnttest>maxcnts):
                        maxcnts = cnttest
                        idxmax = itest

                if (1 not in deltapixlist): 
                    maxcnts = max(cntslist)
                    idxmax = cntslist.index(maxcnts)

                pcnnmax = pcnnlist[idxmax]
                prfmax = prflist[idxmax]
                kmax = klist[idxmax]
                v = realvlist[idxmax]
                phi = realphilist[idxmax]
                sv = svlist[idxmax]
                sphi = sphilist[idxmax]
                deltapixmax = deltapixlist[idxmax]
                appvmax = appvlist[idxmax]
                appthmax = appthlist[idxmax]
                xpixmax = xpixlist[idxmax]
                ypixmax = ypixlist[idxmax]
                gtumax = gtulist[idxmax]

                met_results.append([kmax,pcnnmax,prfmax,maxcnts,v,phi,sv,sphi,deltapixmax,appvmax,appthmax,xpixmax,ypixmax,gtumax])

            if (n_met_file==0): print("Warning! No Trig 2 in Meteore @ %s" % fileback)

            #serve a evitare doppioni
            n_met_file = 0
            klist = []
            pcnnlist,prflist,cntslist = [],[],[]
            appvlist,appthlist = [],[]
            realvlist,realphilist,svlist,sphilist = [],[],[],[]
            xpixlist,ypixlist,gtulist = [],[],[]
            deltapixlist = [] #è 1 se il deltapix è <3 e 0 altrimenti

        #leggo txt e cerco lo stesso file dati
        for k in range(len(txt_data)):

            idx,amag = txt_data[k][0],txt_data[k][2]
            iss_h,iss_v,iss_a = txt_data[k][15],txt_data[k][16],txt_data[k][17]
            met_hi,met_hf = txt_data[k][22],txt_data[k][23]
            xi,xf = txt_data[k][24],txt_data[k][25]
            yi,yf = txt_data[k][26],txt_data[k][27]

            xin,xfin = min(xi,xf),max(xi,xf)
            yin,yfin = min(yi,yf),max(yi,yf)
            hin,hfin = int(min(met_hi,met_hf)),int(max(met_hi,met_hf))

            #cerco file txt giusto
            file_txt = "amag_p%i0_00%i" % (amag,idx)
            if (idx<10): file_txt = "amag_p%i0_000%i" % (amag,idx)
            if (file!=file_txt): continue 

            #meteora nel range di pixel
            metpix = False
            step = 1

            for h in range(hin,hfin+step,step):

                xkm0,ykm0 = pixel_to_km(xmax,ymax,iss_h-h,0)
                xkm1,ykm1 = pixel_to_km(xmax,ymax,iss_h-h,1)
                xkm2,ykm2 = pixel_to_km(xmax,ymax,iss_h-h,2)
                xkm3,ykm3 = pixel_to_km(xmax,ymax,iss_h-h,3)
                xkm4,ykm4 = pixel_to_km(xmax,ymax,iss_h-h,4)

                halfpixel = (iss_h-h)*math.tan(math.radians(22))
                halfpixel = halfpixel / 24.
                halfpixel = halfpixel / 2.

                for ikm in range(5):
                    for jkm in range(5):

                        if (ikm==0): xkm = xkm0
                        if (ikm==1): xkm = xkm1
                        if (ikm==2): xkm = xkm2
                        if (ikm==3): xkm = xkm3
                        if (ikm==4): xkm = xkm4

                        if (jkm==0): ykm = ykm0
                        if (jkm==1): ykm = ykm1
                        if (jkm==2): ykm = ykm2
                        if (jkm==3): ykm = ykm3
                        if (jkm==4): ykm = ykm4

                        dmin = 100 #km
                        for xtest in range(int(xin),int(xfin+1),1):
                            for ytest in range(int(yin),int(yfin+1),1):
                                dist = (xtest-xkm)*(xtest-xkm) + (ytest-ykm)*(ytest-ykm)
                                dist = math.sqrt(dist)
                                if (dist < dmin): dmin = dist
                        
                        if (dmin<halfpixel*4. or (xkm>=xin and xkm<=xfin and ykm>=yin and ykm<=yfin)): 
                            
                            n_met_file = n_met_file + 1
                            metpix = True

                            klist.append(k)
                            prflist.append(prf)
                            pcnnlist.append(pcnn)
                            cntslist.append(cnts)
                            deltapixlist.append(notes)
                            xpixlist.append(xmax)
                            ypixlist.append(ymax)
                            gtulist.append(gtu)

                            appvlist.append(vmet)
                            appthlist.append(thetamet)

                            vxapp = vmet*math.cos(math.radians(thetamet))
                            vyapp = vmet*math.sin(math.radians(thetamet))
                            vxreal = vxapp
                            vyreal = vyapp - iss_v
                            realv = math.sqrt(vxreal*vxreal + vyreal*vyreal)
                            phiatan = math.degrees(math.atan2(-vxreal,-vyreal))

                            #mappo tra 0,360 e -360,0
                            phiatan = phiatan - int(phiatan/360)*360
                            iss_a = iss_a - int(iss_a/360)*360

                            #mappo i negativi nei positivi
                            if (phiatan<0): phiatan = phiatan + 360
                            if (iss_a<0): iss_a = iss_a + 360 

                            realphi = iss_a + phiatan
                            realphi = realphi - int(realphi/360)*360
                            if (realphi<0): realphi = 360 + realphi

                            #incertezze da stackCNN
                            svapp = 0.5 #km/s @ 370 km
                            svapp = svapp*newspeed
                            svtheta = 5 #gradi -> vanno convertiti in rad
                            svtheta = math.radians(svtheta)
                            costh = math.cos(math.radians(thetamet))
                            sinth = math.sin(math.radians(thetamet))
                            svx = math.sqrt(costh*costh*svapp*svapp + sinth*sinth*vmet*vmet*svtheta*svtheta)
                            svy = math.sqrt(sinth*sinth*svapp*svapp + costh*costh*vmet*vmet*svtheta*svtheta)
                            sv = math.sqrt(vxreal*vxreal*svx*svx + vyreal*vyreal*svy*svy) / realv
                            sphi = math.sqrt(vxreal*vxreal*svy*svy + vyreal*vyreal*svx*svx) / (realv*realv)
                            sphi = math.degrees(sphi) #va convertita in °

                            realvlist.append(realv)
                            realphilist.append(realphi)
                            svlist.append(sv)
                            sphilist.append(sphi)

                            break

                    if (metpix): break
                if (metpix): break

            if (not metpix): #un evento di meteora 
                if (amag==4): fake_cnn_4.append(pcnn)
                if (amag==5): fake_cnn_5.append(pcnn)
                if (amag==6): fake_cnn_6.append(pcnn)
                if (pcnn>cutcnn*100. and notes==1): 
                    if (amag==4): 
                        fake_rf_4.append(prf)
                        fake_pix_4.append(pcnn)
                    if (amag==5): 
                        fake_rf_5.append(prf)
                        fake_pix_5.append(pcnn)
                    if (amag==6): 
                        fake_rf_6.append(prf)
                        fake_pix_6.append(pcnn)
            
        fileback = file

    return met_results,fake_cnn_4,fake_cnn_5,fake_cnn_6,fake_pix_4,fake_pix_5,fake_pix_6,fake_rf_4,fake_rf_5,fake_rf_6

#plot analisi
def plot_residuals(simvariables,stackvariables,odevariables,basevariables,physvariables):

    pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Training/"
    if(not os.path.exists(pngdir)): os.makedirs(pngdir)

    #simulation variables
    realvlist = [sim[0] for sim in simvariables]
    realvapplist = [sim[1] for sim in simvariables]
    realvapplist_km = [sim[1]*pixelkm/gtusec for sim in simvariables]
    visslist = [sim[2] for sim in simvariables]
    phiisslist = [sim[3] for sim in simvariables]
    realthlist = [sim[4] for sim in simvariables]
    realphilist = [sim[5] for sim in simvariables]

    #stack-cnn variables
    stackvapplist = [stack[0] for stack in stackvariables] #in km/s
    stackvapplist_km = stackvapplist
    stackthlist = [stack[1] for stack in stackvariables]
    stackvlist = [stack[2] for stack in stackvariables] #in km/s
    stackphilist = [stack[3] for stack in stackvariables]

    #odenet variables 
    odenetlosslist = [ode[0] for ode in odevariables]
    odenettimelist = [ode[1] for ode in odevariables]
    odenetvapplist = [ode[2] for ode in odevariables]
    odenetthlist = [ode[3] for ode in odevariables]

    #va sostituito con fit con sigma su y pari alla sigma di quelli non calibrati
    mcal,bcal = np.polyfit(realvapplist,odenetvapplist,1)

    #Plot 2D Correlazione Velocità Modello e Reale
    f = plt.figure()
    plt.scatter(realvapplist,odenetvapplist,marker = "x",s = 30,color = "r", label = "Data")
    plt.plot(realvapplist,[(mcal*speed + bcal) for speed in realvapplist],c="b",label = "Calibration")
    plt.xlabel("Real Apparent Speed [pix/GTU]")
    plt.ylabel("OdeNet Apparent Speed [pix/GTU]")
    plt.title(" 2D OdeNet Speed Calibration \n y = %.3fx + %.3f" % (mcal,bcal))
    plt.grid()
    plt.legend(loc = "best")
    metpng = pngdir + "distro2d_odenet_speed.png"
    plt.savefig(metpng)
    f.clear()
    plt.close(f)

    #calibrated with apparent real speed
    odenetvapplist_cal = [(v-bcal)/mcal for v in odenetvapplist]
    odenetvapplist_km = [v*pixelkm/gtusec for v in odenetvapplist_cal] #in km/s

    #baseline (not ML) variables
    basevapplist = [base[0] for base in basevariables]
    basevapplist_km = [base[0]*pixelkm/gtusec for base in basevariables]
    basethlist = [base[1] for base in basevariables] 

    #azimuth and real speed 
    odenetphilist, odenetvlist = [],[]
    basephilist, basevlist = [],[]

    for iphi in range(len(realvlist)):

        iss_a, iss_v = phiisslist[iphi],visslist[iphi]
        iss_a = iss_a - int(iss_a/360)*360
        if (iss_a<0): iss_a = iss_a + 360 

        #odenet variables
        vxphi_ode = odenetvapplist_km[iphi]*math.cos(math.radians(odenetthlist[iphi]))
        vyphi_ode = odenetvapplist_km[iphi]*math.sin(math.radians(odenetthlist[iphi]))
        phiatan_ode = math.degrees(math.atan2(-vxphi_ode,-vyphi_ode + iss_v))
        phiatan_ode = phiatan_ode - int(phiatan_ode/360)*360
        realphi_ode = iss_a + phiatan_ode
        realphi_ode = realphi_ode - int(realphi_ode/360)*360
        if (realphi_ode<0): realphi_ode = 360 + realphi_ode
        realv_ode = math.sqrt(vxphi_ode**2 + (vyphi_ode - iss_v)**2)

        odenetphilist.append(realphi_ode)
        odenetvlist.append(realv_ode)

        #baseline variables
        vxphi_base = basevapplist_km[iphi]*math.cos(math.radians(basethlist[iphi]))
        vyphi_base = basevapplist_km[iphi]*math.sin(math.radians(basethlist[iphi]))
        phiatan_base = math.degrees(math.atan2(-vxphi_base,-vyphi_base + iss_v))
        phiatan_base = phiatan_base - int(phiatan_base/360)*360
        realphi_base = iss_a + phiatan_base
        realphi_base = realphi_base - int(realphi_base/360)*360
        if (realphi_base<0): realphi_base = 360 + realphi_base
        realv_base = math.sqrt(vxphi_base**2 + (vyphi_base - iss_v)**2)

        basephilist.append(realphi_base)
        basevlist.append(realv_base)

    #computing residuals
    odenetresvapplist, odenetresvlist = [],[]
    odenetresthlist, odenetresphilist = [],[]
    stackresvapplist, stackresvlist = [],[]
    stackresthlist, stackresphilist = [],[]
    baseresvapplist, baseresvlist = [],[]
    baseresthlist, baseresphilist = [],[]

    for ires in range(len(realvlist)):

        #odenet residuals
        odenetresvapp = odenetvapplist_km[ires] - realvapplist_km[ires]
        odenetresv = odenetvlist[ires] - realvlist[ires]
        odenetresth = odenetthlist[ires] - realthlist[ires]
        odenetresphi = odenetphilist[ires] - realphilist[ires]

        if (odenetresth>180): odenetresth = odenetresth - 360
        if (odenetresth<(-180)): odenetresth = 360 + odenetresth
        if (odenetresphi>180): odenetresphi = odenetresphi - 360
        if (odenetresphi<(-180)): odenetresphi = 360 + odenetresphi

        odenetresvapplist.append(odenetresvapp)
        odenetresvlist.append(odenetresv)
        odenetresthlist.append(odenetresth)
        odenetresphilist.append(odenetresphi)

        #stack-CNN residuals
        stackresvapp = stackvapplist_km[ires] - realvapplist_km[ires]
        stackresv = stackvlist[ires] - realvlist[ires]
        stackresth = stackthlist[ires] - realthlist[ires]
        stackresphi = stackphilist[ires] - realphilist[ires]

        if (stackresth>180): stackresth = stackresth - 360
        if (stackresth<(-180)): stackresth = 360 + stackresth
        if (stackresphi>180): stackresphi = stackresphi - 360
        if (stackresphi<(-180)): stackresphi = 360 + stackresphi

        stackresvapplist.append(stackresvapp)
        stackresvlist.append(stackresv)
        stackresthlist.append(stackresth)
        stackresphilist.append(stackresphi)

        #baseline residuals
        baseresvapp = basevapplist_km[ires] - realvapplist_km[ires]
        baseresv = basevlist[ires] - realvlist[ires]
        baseresth = basethlist[ires] - realthlist[ires]
        baseresphi = basephilist[ires] - realphilist[ires]

        if (baseresth>180): baseresth = baseresth - 360
        if (baseresth<(-180)): baseresth = 360 + baseresth
        if (baseresphi>180): baseresphi = baseresphi - 360
        if (baseresphi<(-180)): baseresphi = 360 + baseresphi

        baseresvapplist.append(baseresvapp)
        baseresvlist.append(baseresv)
        baseresthlist.append(baseresth)
        baseresphilist.append(baseresphi)

    #Distribuzione temporale
    hstart = min(odenettimelist)
    hstop = max(odenettimelist)
    nbins = 100
    stephist = (hstop-hstart) / nbins
    stephist = round(stephist,3)
    metpng = pngdir + "distro1d_time.png"
    xtitlehist = "OdeNet Training Time [s]"
    ytitlehist = "Events / %.3f s" % (stephist)
    nbins = int((hstop-hstart)/stephist)

    f = plt.figure()
    plt.hist(odenettimelist,bins = nbins,range = (hstart,hstop), color = "b")
    plt.title("ODENET TRAINING TIME DISTRIBUTION")
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.savefig(metpng,bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #Distribuzione loss
    hstart = min(odenetlosslist)
    hstop = max(odenetlosslist)
    nbins = 100
    stephist = (hstop-hstart) / nbins
    stephist = round(stephist,7)
    metpng = pngdir + "distro1d_loss.png"
    if (losstype==1): xtitlehist = "Least Absolute Deviations [L1]"
    if (losstype==2): xtitlehist = "Mean Squared Error [MSE]"
    ytitlehist = "Events / %.7f" % (stephist)
    nbins = int((hstop-hstart)/stephist)

    f = plt.figure()
    plt.hist(odenetlosslist,bins = nbins,range = (hstart,hstop), color = "b")
    plt.title("ODENET LOSS DISTRIBUTION")
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.savefig(metpng,bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #APPARENT THETA RESIDUAL
    stephist = 5
    resmax = 180
    hstop = resmax - stephist/2.
    hstart = -resmax + stephist/2.
    nbins = int((hstop-hstart)/stephist)
    metpng = pngdir + "distro1d_restheta_odenet.png"
    xtitlehist = "OdeNet Apparent Direction Residual [°]"
    ytitlehist = "Events / %i °" % (stephist)

    restharr = np.array(odenetresthlist)
    resth_mu = np.mean(restharr)
    resth_sig = np.std(restharr)

    restharr_stack = np.array(stackresthlist)
    resth_mu_stack = np.mean(restharr_stack)
    resth_sig_stack = np.std(restharr_stack)

    restharr_base = np.array(baseresthlist)
    resth_mu_base = np.mean(restharr_base)
    resth_sig_base = np.std(restharr_base)

    f = plt.figure()
    plt.hist(odenetresthlist, bins = nbins, range = (hstart,hstop), color = "b")
    titlehist = "ODENET APPARENT DIRECTION RESIDUAL DISTRIBUTION\n" + r"$\mu$" + (" = %.2f °, " % resth_mu) + r"$\sigma$" + (" = %.2f °" % resth_sig)
    plt.title(titlehist)
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.savefig(pngdir + "distro1d_restheta_odenet.png")
    f.clear()
    plt.close(f)

    f = plt.figure()
    plt.hist(odenetresthlist, bins = nbins, range = (hstart,hstop), color = "b",label="OdeNet",linestyle="solid",edgecolor="black")
    plt.hist(stackresthlist, bins = nbins, range = (hstart,hstop), color = "magenta",label="Stack-CNN",linestyle="dashdot",alpha=0.5,edgecolor ="black")
    if (plotbaseline): plt.hist(baseresthlist, bins = nbins, range = (hstart,hstop), color = "g",label="Baseline",linestyle="dashed",alpha=0.5,edgecolor ="black")
    titlehist = " APPARENT DIRECTION RESIDUAL DISTRIBUTION \n OdeNet: " + r"$\mu$" + (" = %.2f °, " % resth_mu) + r"$\sigma$" + (" = %.2f °" % resth_sig)
    titlehist = titlehist + " \n Stack-CNN: " + r"$\mu$" + (" = %.2f °, " % resth_mu_stack) + r"$\sigma$" + (" = %.2f °" % resth_sig_stack) 
    if (plotbaseline):  titlehist = titlehist + " \n Baseline: " + r"$\mu$" + (" = %.2f °, " % resth_mu_base) + r"$\sigma$" + (" = %.2f °" % resth_sig_base) 
    plt.title(titlehist)
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.legend(loc = "best")
    plt.savefig(pngdir + "distro1d_restheta_odenet_stackcnn_baseline.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #AZIMUTH RESIDUAL
    stephist = 5
    resmax = 180
    hstop = resmax - stephist/2.
    hstart = -resmax + stephist/2.
    nbins = int((hstop-hstart)/stephist)
    metpng = pngdir + "distro1d_resphi_odenet.png"
    xtitlehist = "OdeNet Azimuth Residual [°]"
    ytitlehist = "Events / %i °" % (stephist)

    resphiarr = np.array(odenetresphilist)
    resphi_mu = np.mean(resphiarr)
    resphi_sig = np.std(resphiarr)

    resphiarr_stack = np.array(stackresphilist)
    resphi_mu_stack = np.mean(resphiarr_stack)
    resphi_sig_stack = np.std(resphiarr_stack)

    resphiarr_base = np.array(baseresphilist)
    resphi_mu_base = np.mean(resphiarr_base)
    resphi_sig_base = np.std(resphiarr_base)

    f = plt.figure()
    plt.hist(odenetresphilist, bins = nbins, range = (hstart,hstop), color = "b")
    titlehist = "ODENET AZIMUTH RESIDUAL DISTRIBUTION\n" + r"$\mu$" + (" = %.2f °, " % resphi_mu) + r"$\sigma$" + (" = %.2f °" % resphi_sig)
    plt.title(titlehist)
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.savefig(pngdir + "distro1d_resphi_odenet.png")
    f.clear()
    plt.close(f)

    f = plt.figure()
    plt.hist(odenetresphilist, bins = nbins, range = (hstart,hstop), color = "b",label="OdeNet",linestyle="solid",edgecolor="black")
    plt.hist(stackresphilist, bins = nbins, range = (hstart,hstop), color = "magenta",label="Stack-CNN",linestyle="dashdot",alpha=0.5,edgecolor ="black")    
    if (plotbaseline): plt.hist(baseresphilist, bins = nbins, range = (hstart,hstop), color = "g",label="Baseline",linestyle="dashed",alpha=0.5,edgecolor ="black")
    titlehist = " AZIMUTH RESIDUAL DISTRIBUTION \n OdeNet: " + r"$\mu$" + (" = %.2f °, " % resphi_mu) + r"$\sigma$" + (" = %.2f °" % resphi_sig)
    titlehist = titlehist + " \n Stack-CNN: " + r"$\mu$" + (" = %.2f °, " % resphi_mu_stack) + r"$\sigma$" + (" = %.2f °" % resphi_sig_stack) 
    if (plotbaseline): titlehist = titlehist + " \n Baseline: " + r"$\mu$" + (" = %.2f °, " % resphi_mu_base) + r"$\sigma$" + (" = %.2f °" % resphi_sig_base) 
    plt.title(titlehist)
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.legend(loc = "best")
    plt.savefig(pngdir + "distro1d_resphi_odenet_stackcnn_baseline.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #APPARENT SPEED RESIDUAL
    stephist = 2.
    resmax = 50
    hstop = resmax - stephist/2.
    hstart = -resmax + stephist/2.
    nbins = int((hstop-hstart)/stephist)
    metpng = pngdir + "distro1d_resvapp_odenet.png"
    xtitlehist = "OdeNet Apparent Speed Residual [km/s]"
    ytitlehist = "Events / %.1f km/s" % (stephist)

    resvapparr = np.array(odenetresvapplist)
    resvapp_mu = np.mean(resvapparr)
    resvapp_sig = np.std(resvapparr)

    resvapparr_stack = np.array(stackresvapplist)
    resvapp_mu_stack = np.mean(resvapparr_stack)
    resvapp_sig_stack = np.std(resvapparr_stack)

    resvapparr_base = np.array(baseresvapplist)
    resvapp_mu_base = np.mean(resvapparr_base)
    resvapp_sig_base = np.std(resvapparr_base)

    f = plt.figure()
    plt.hist(odenetresvapplist, bins = nbins, range = (hstart,hstop), color = "b")
    titlehist = "ODENET APPARENT SPEED RESIDUAL DISTRIBUTION\n" + r"$\mu$" + (" = %.2f km/s, " % resvapp_mu) + r"$\sigma$" + (" = %.2f km/s" % resvapp_sig)
    plt.title(titlehist)
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.savefig(pngdir + "distro1d_resvapp_odenet.png")
    f.clear()
    plt.close(f)

    f = plt.figure()
    plt.hist(odenetresvapplist, bins = nbins, range = (hstart,hstop), color = "b",label="OdeNet",linestyle="solid",edgecolor="black")
    plt.hist(stackresvapplist, bins = nbins, range = (hstart,hstop), color = "magenta",label="Stack-CNN",linestyle="dashdot",alpha=0.5,edgecolor ="black")
    if (plotbaseline): plt.hist(baseresvapplist, bins = nbins, range = (hstart,hstop), color = "g",label="Baseline",linestyle="dashed",alpha=0.5,edgecolor ="black")
    titlehist = " APPARENT SPEED RESIDUAL DISTRIBUTION \n OdeNet: " + r"$\mu$" + (" = %.2f km/s, " % resvapp_mu) + r"$\sigma$" + (" = %.2f km/s" % resvapp_sig)
    titlehist = titlehist + " \n Stack-CNN: " + r"$\mu$" + (" = %.2f km/s, " % resvapp_mu_stack) + r"$\sigma$" + (" = %.2f km/s" % resvapp_sig_stack) 
    if (plotbaseline): titlehist = titlehist + " \n Baseline: " + r"$\mu$" + (" = %.2f km/s, " % resvapp_mu_base) + r"$\sigma$" + (" = %.2f km/s" % resvapp_sig_base) 
    plt.title(titlehist)
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.legend(loc = "best")
    plt.savefig(pngdir + "distro1d_resvapp_odenet_stackcnn_baseline.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #REAL SPEED RESIDUAL
    stephist = 2.
    resmax = 50
    hstop = resmax - stephist/2.
    hstart = -resmax + stephist/2.
    nbins = int((hstop-hstart)/stephist)
    metpng = pngdir + "distro1d_resv_odenet.png"
    xtitlehist = "OdeNet Real Speed Residual [km/s]"
    ytitlehist = "Events / %.1f km/s" % (stephist)

    resvarr = np.array(odenetresvlist)
    resv_mu = np.mean(resvarr)
    resv_sig = np.std(resvarr)

    resvarr_stack = np.array(stackresvlist)
    resv_mu_stack = np.mean(resvarr_stack)
    resv_sig_stack = np.std(resvarr_stack)

    resvarr_base = np.array(baseresvlist)
    resv_mu_base = np.mean(resvarr_base)
    resv_sig_base = np.std(resvarr_base)

    f = plt.figure()
    plt.hist(odenetresvlist, bins = nbins, range = (hstart,hstop), color = "b")
    titlehist = "ODENET REAL SPEED RESIDUAL DISTRIBUTION\n" + r"$\mu$" + (" = %.2f km/s, " % resv_mu) + r"$\sigma$" + (" = %.2f km/s" % resv_sig)
    plt.title(titlehist)
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.savefig(pngdir + "distro1d_resv_odenet.png")
    f.clear()
    plt.close(f)

    f = plt.figure()
    plt.hist(odenetresvlist, bins = nbins, range = (hstart,hstop), color = "b",label="OdeNet",linestyle="solid",edgecolor="black")
    plt.hist(stackresvlist, bins = nbins, range = (hstart,hstop), color = "magenta",label="Stack-CNN",linestyle="dashdot",alpha=0.5,edgecolor ="black")
    if (plotbaseline): plt.hist(baseresvlist, bins = nbins, range = (hstart,hstop), color = "g",label="Baseline",linestyle="dashed",alpha=0.5,edgecolor ="black")
    titlehist = " REAL SPEED RESIDUAL DISTRIBUTION \n OdeNet: " + r"$\mu$" + (" = %.2f km/s, " % resv_mu) + r"$\sigma$" + (" = %.2f km/s" % resv_sig)
    titlehist = titlehist + " \n Stack-CNN: " + r"$\mu$" + (" = %.2f km/s, " % resv_mu_stack) + r"$\sigma$" + (" = %.2f km/s" % resv_sig_stack) 
    if (plotbaseline): titlehist = titlehist + " \n Baseline: " + r"$\mu$" + (" = %.2f km/s, " % resv_mu_base) + r"$\sigma$" + (" = %.2f km/s" % resv_sig_base) 
    plt.title(titlehist)
    plt.xlabel(xtitlehist)
    plt.ylabel(ytitlehist)
    plt.legend(loc = "best")
    plt.savefig(pngdir + "distro1d_resv_odenet_stackcnn_baseline.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #Plot 2D Correlazione residuo theta e loss
    f = plt.figure()
    plt.scatter([abs(res) for res in odenetresthlist],odenetlosslist,marker = "o",s = 20,color = "b")
    plt.xlabel("Apparent Theta Residual [°]")
    if (losstype==1): plt.ylabel("Least Absolute Deviations [L1]")
    if (losstype==2): plt.ylabel("Mean Squared Error [MSE]")
    plt.title("2D OdeNet Loss - Residual")
    plt.grid()
    metpng = pngdir + "distro2d_loss_resth.png"
    plt.savefig(metpng)
    f.clear()
    plt.close(f)

    #Plot 2D Correlazione residuo vapp e loss
    f = plt.figure()
    plt.scatter([abs(res) for res in odenetresvapplist],odenetlosslist,marker = "o",s = 20,color = "b")
    plt.xlabel("Apparent Speed Residual [km/s]")
    if (losstype==1): plt.ylabel("Least Absolute Deviations [L1]")
    if (losstype==2): plt.ylabel("Mean Squared Error [MSE]")
    plt.title("2D OdeNet Loss - Residual")
    plt.grid()
    metpng = pngdir + "distro2d_loss_resvapp.png"
    plt.savefig(metpng)
    f.clear()
    plt.close(f)

    #Plot 2D Correlazione residuo theta e loss
    f = plt.figure()
    plt.scatter([abs(res) for res in odenetresphilist],odenetlosslist,marker = "o",s = 20,color = "b")
    plt.xlabel("Azimuth Residual [°]")
    if (losstype==1): plt.ylabel("Least Absolute Deviations [L1]")
    if (losstype==2): plt.ylabel("Mean Squared Error [MSE]")
    plt.title("2D OdeNet Loss - Residual")
    plt.grid()
    metpng = pngdir + "distro2d_loss_resphi.png"
    plt.savefig(metpng)
    f.clear()
    plt.close(f)

    #Plot 2D Correlazione residuo vapp e loss
    f = plt.figure()
    plt.scatter([abs(res) for res in odenetresvlist],odenetlosslist,marker = "o",s = 20,color = "b")
    plt.xlabel("Real Speed Residual [km/s]")
    if (losstype==1): plt.ylabel("Least Absolute Deviations [L1]")
    if (losstype==2): plt.ylabel("Mean Squared Error [MSE]")
    plt.title("2D OdeNet Loss - Residual")
    plt.grid()
    metpng = pngdir + "distro2d_loss_resv.png"
    plt.savefig(metpng)
    f.clear()
    plt.close(f)

    #Plot 2D Correlazione variabili odenet e variabili fisiche
    modelvariables = [odenetlosslist,odenetresvapplist,odenetresvlist,odenetresthlist,odenetresphilist]
    modeltitles = ["OdeNet Loss","OdeNet Apparent Speed Residual [km/s]","OdeNet Longitudinal Speed Residual [km/s]","OdeNet Apparent Direction Residual [°]","OdeNet Azimuth Residual [°]"]
    phystitles = ["Type [0 = Asteroid, 1 = Comet]","Absolute Magnitude","Background [D1 Counts]","Mass [kg]","Pre-Atmospheric Speed [km/s]","Asteroid Density [kg/"+r"$m^3$"+"]","Ablation Coefficient ["+r"${s}^2/{km}^2$"+"]","Luminous Efficiency","mu","fp","mu0","fp0","Inclination [°]","Azimuth [°]","Length [s]","Max Counts","Mean Height [km]","Apparent Speed [km/s]","Apparent Direction [°]"]
    nphys = len(physvariables[0])
    nmodel = len(modelvariables)

    #distribuzione 2d trig2 -> variabili fisiche vs variabili modello
    for i in range(nphys):

        physlist = []
        for nsample in range(len(physvariables)): physlist.append(physvariables[nsample][i])

        for j in range(nmodel):

            modellist = modelvariables[j]

            f = plt.figure()
            plt.scatter(modellist,physlist,marker = "o",s = 20)
            plt.ylabel(phystitles[i])
            plt.xlabel(modeltitles[j])
            plt.title("2D Meteor Distribution")
            plt.grid()
            metpng = pngdir + "distro2d_phys_odenet_%i_%i.png" % (i,j)
            plt.savefig(metpng)
            f.clear()
            plt.close(f)

            plt.close('all')

    #distribuzione 1d in funzione inclinazione
    incllist = []
    for nsample in range(len(physvariables)): incllist.append(physvariables[nsample][12])

    for j in range(1,nmodel):

        modellist = modelvariables[j]
        modellist_0_30 = [modellist[i] for i,incl in enumerate(incllist) if (incl>=0 and incl<30)]
        modellist_30_45 = [modellist[i] for i,incl in enumerate(incllist) if (incl>=30 and incl<45)]
        modellist_45_60 = [modellist[i] for i,incl in enumerate(incllist) if (incl>=45 and incl<60)]
        modellist_60_90 = [modellist[i] for i,incl in enumerate(incllist) if (incl>=60 and incl<=90)]

        if (j==1 or j==2): #speed
            stephist = 2.
            resmax = 50
            ytitlehist = "Events / %.1f km/s" % (stephist)

        if (j==3 or j==4): #angle
            stephist = 5.
            resmax = 180
            ytitlehist = "Events / %i °" % (stephist)

        hstop = resmax - stephist/2.
        hstart = -resmax + stephist/2.
        nbins = int((hstop-hstart)/stephist)
        
        if (j==1): xtitlehist = "OdeNet Apparent Speed Residual [km/s]"
        if (j==2): xtitlehist = "OdeNet Real Speed Residual [km/s]"
        if (j==3): xtitlehist = "OdeNet Apparent Direction Residual [°]"
        if (j==4): xtitlehist = "OdeNet Azimuth Residual [°]"

        if (j==1 or j==2): unit = "km/s"
        if (j==3 or j==4):  unit = "°"

        res_mu_0_30,res_sig_0_30 = np.mean(modellist_0_30),np.std(modellist_0_30)
        res_mu_30_45,res_sig_30_45 = np.mean(modellist_30_45),np.std(modellist_30_45)
        res_mu_45_60,res_sig_45_60 = np.mean(modellist_45_60),np.std(modellist_45_60)
        res_mu_60_90,res_sig_60_90 = np.mean(modellist_60_90),np.std(modellist_60_90)

        if (j==1): titlehist = "OdeNet Apparent Speed Residual Distribution"
        if (j==2): titlehist = "OdeNet Real Speed Residual Distribution"
        if (j==3): titlehist = "OdeNet Apparent Direction Residual Distribution"
        if (j==4): titlehist = "OdeNet Azimuth Residual Distribution"

        titlehist_0_30 = titlehist + " - Inclination [0,30] °\n" + r"$\mu$" + (" = %.2f %s, " % (res_mu_0_30,unit)) + r"$\sigma$" + (" = %.2f %s" % (res_sig_0_30,unit))
        titlehist_30_45 = titlehist + " - Inclination [30,45] °\n" + r"$\mu$" + (" = %.2f %s, " % (res_mu_30_45,unit)) + r"$\sigma$" + (" = %.2f %s" % (res_sig_30_45,unit))
        titlehist_45_60 = titlehist + " - Inclination [45,60] °\n" + r"$\mu$" + (" = %.2f %s, " % (res_mu_45_60,unit)) + r"$\sigma$" + (" = %.2f %s" % (res_sig_45_60,unit))
        titlehist_60_90 = titlehist + " - Inclination [60,90] °\n" + r"$\mu$" + (" = %.2f %s, " % (res_mu_60_90,unit)) + r"$\sigma$" + (" = %.2f %s" % (res_sig_60_90,unit))

        f = plt.figure()
        plt.hist(modellist_0_30, bins = nbins, range = (hstart,hstop), color = "b")
        plt.title(titlehist_0_30)
        plt.xlabel(xtitlehist)
        plt.ylabel(ytitlehist)
        plt.savefig(pngdir + "distro1d_%i_incl_0_30.png" % (j))
        f.clear()
        plt.close(f)

        f = plt.figure()
        plt.hist(modellist_30_45, bins = nbins, range = (hstart,hstop), color = "b")
        plt.title(titlehist_30_45)
        plt.xlabel(xtitlehist)
        plt.ylabel(ytitlehist)
        plt.savefig(pngdir + "distro1d_%i_incl_30_45.png" % (j))
        f.clear()
        plt.close(f)

        f = plt.figure()
        plt.hist(modellist_45_60, bins = nbins, range = (hstart,hstop), color = "b")
        plt.title(titlehist_45_60)
        plt.xlabel(xtitlehist)
        plt.ylabel(ytitlehist)
        plt.savefig(pngdir + "distro1d_%i_incl_45_60.png" % (j))
        f.clear()
        plt.close(f)

        f = plt.figure()
        plt.hist(modellist_60_90, bins = nbins, range = (hstart,hstop), color = "b")
        plt.title(titlehist_60_90)
        plt.xlabel(xtitlehist)
        plt.ylabel(ytitlehist)
        plt.savefig(pngdir + "distro1d_%i_incl_60_90.png" % (j))
        f.clear()
        plt.close(f)

        plt.close('all')