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
import numpy as np
import os,math,sys,time,datetime,openpyxl
from siren_pytorch import SirenNet

torch.manual_seed(42) #fisso seed
userdir = "/mnt/c/Users/leona/"
if(not os.path.exists(userdir)): userdir = "/media/leonardo/HDD/"
drawopt = True #disegno dataset
gtuvalue = 1. #i secondi di una gtu
iani = 0 #serve per gif

#hyperparameters
dim_in, dim_out = 2,1 #input pixel2d, output cnts/pixel/gtu
hidden_size, n_fourier = 128,6 #neuroni e numero di mappe fourier 
n_layers = 8 #8 layer
skipopt = True #implementa skip connection

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
        x = torch.sigmoid(self.fc8(x)) #nel finale c'è sigmoide

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

    def __init__(self,dim_in,dim_out,dim_hidden,dim_fourier,x_in,skipopt = False): #costruttore 

        super(FullOdeNet,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.dim_fourier = dim_fourier
        self.skipopt = skipopt
        self.x_in = x_in #posizione iniziale meteora -> deve essere 2d e tensore torch

        speed = torch.Tensor([0.,0.]) #inizializzo a 1 le velocità 
        self.speed = nn.Parameter(speed) #è importante che siano parametri -> learnable

        #layer globali e locali meteora
        self.globalode = OdeNet(self.dim_in,self.dim_out,self.dim_hidden,self.dim_fourier,self.skipopt)
        self.localode = OdeNet(self.dim_in,self.dim_out,self.dim_hidden,self.dim_fourier,self.skipopt)

    def forward(self,x,t): #da input a output

        #transformazione globale
        pix2d = x #mi serve per salvarla 
        bkgcounts = self.globalode(pix2d) #prima passo in quella di fondo
        
        #transformazione locale
        localpix = self.timetransform(pix2d,t)
        metcounts = self.localode(localpix) #poi in quella 

        #sommo output 1 con output 2
        totcounts = bkgcounts + metcounts
        return totcounts 

    def timetransform(self,x,t): #transformazione con equazione differenziale
        x = x - self.x_in - self.speed*t*gtuvalue
        return x

class SirenOdeNet(nn.Module): #inerita da nn.Module

    def __init__(self,dim_in,dim_out,dim_hidden,n_layers,x_in): #costruttore 

        super(SirenOdeNet,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers
        self.x_in = x_in #posizione iniziale meteora -> deve essere 2d e tensore torch

        speed = torch.Tensor([0.,0.]) #inizializzo a 1 le velocità 
        self.speed = nn.Parameter(speed) #è importante che siano parametri -> learnable

        #layer globali e locali meteora
        self.globalode = SirenNet(self.dim_in,self.dim_hidden,self.dim_out,self.n_layers,final_activation = nn.Sigmoid())
        self.localode = SirenNet(self.dim_in,self.dim_hidden,self.dim_out,self.n_layers,final_activation = nn.Sigmoid())

    def forward(self,x,t): #da input a output

        #transformazione globale
        pix2d = x #mi serve per salvarla 
        bkgcounts = self.globalode(pix2d) #prima passo in quella di fondo
        
        #transformazione locale
        localpix = self.timetransform(pix2d,t)
        metcounts = self.localode(localpix) #poi in quella 

        #sommo output 1 con output 2
        totcounts = bkgcounts + metcounts
        return totcounts 

    def timetransform(self,x,t): #transformazione con equazione differenziale
        x = x - self.x_in - self.speed*t*gtuvalue
        return x

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
def plot_video(data,amag,index,xpix,ypix,gtupix,pixelcloud,ngtu,speedmodel,thetamodel,modeldir,modelname):

    filestring = buildstring(amag,index)
    pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Training/%s/%s/" % (filestring,modeldir)
    if(not os.path.exists(pngdir)): os.makedirs(pngdir)

    inputtext = userdir + 'Documenti/ETOS/Dati/SIM/%s.txt' % (filestring)
    if os.path.isfile(inputtext):
        a = np.loadtxt(inputtext) #crea array di numpy con dati in ordine
        #print("Leggendo il file %s" %(inputtext))
    else: sys.exit() 

    back = np.reshape(a,(128,48,48)) #fa un reshape dei dati nell'array a con lunghezza array nel primo, righe e colonne
    for entry in range(128): back[entry,:,:] = np.rot90(back[entry,:,:],1) #i dati sono ruotati -> così è come su ETOS

    txt_data = read_txt_data(amag,index)
    xpixin,xpixfin = txt_data[28],txt_data[29]
    ypixin,ypixfin = txt_data[30],txt_data[31]
    realth,realv_pix = txt_data[32],txt_data[33] #in deg e pix/gtu
    realv = txt_data[5]*math.cos(math.radians(txt_data[13])) #in km/s

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
    for igtu,gtu in enumerate(range(gtupix,gtupix+ngtu)): pixel2d[igtu,:,:] = back[gtu,:,:]
    minvalue,maxvalue = np.min(pixel2d[:,47-ypixmax:48-ypixmin,xpixmin:xpixmax+1]),np.max(pixel2d[:,47-ypixmax:48-ypixmin,xpixmin:xpixmax+1])
    
    #video simulato
    f = plt.figure()
    plt.xlabel('pix X')
    plt.ylabel('pix Y')
    plt.xlim(xpixmin,xpixmax+1)
    plt.ylim(ypixmin,ypixmax+1)
    labelsim = "Simulation - v = %.2f km/s, %s = %.0f °" % (realv,r"$\theta$",realth)
    plt.arrow(xpix+0.5,ypix+0.5,dxreal,dyreal,head_width=0.7, head_length=0.7, length_includes_head=True, color="g", label = labelsim)
    global iani
    iani = 0
    im = plt.imshow(pixel2d[0,:,:],animated=True,cmap="hot",extent=(0,48,0,48))
    plt.clim(0.9*minvalue,1.1*maxvalue)
    plt.colorbar()
    plt.legend(loc="best")

    def updatefig(*args):
        global iani
        iani = iani + 1
        if (iani>ngtu-1): iani = 0
        titlevideo = " Simulated (M,idx) = (%i,%i) \n max = (%i,%i), GTU = %i"  % (amag,index,xpix,ypix,gtupix+iani)
        plt.title(titlevideo)
        im.set_array(pixel2d[iani,:,:])
        plt.clim(minvalue,maxvalue)
        return im,

    ani = animation.FuncAnimation(f, updatefig, blit=True)
    ani.save(pngdir + "etos_video_%i_%i.gif" % (gtupix,gtupix+ngtu-1), writer='pillow', fps=1.5)
    f.clear()
    plt.close(f)

    #figura ricostruita
    f = plt.figure()
    plt.xlabel('pix X')
    plt.ylabel('pix Y')
    labelsim = "Simulation - v = %.2f km/s, %s = %.0f °" % (realv,r"$\theta$",realth)
    titlemodel = " %s \n " % (modelname)
    labelmodel = "Model - v = %.2f ?, %s = %.0f °" % (speedmodel,r"$\theta$",thetamodel)
    plt.arrow(xpix+0.5-xpixmin,ypix+0.5-ypixmin,dxreal,dyreal,head_width=0.7, head_length=0.7, length_includes_head=True, color="g", label = labelsim)
    plt.arrow(xpix+0.5-xpixmin,ypix+0.5-ypixmin,realv_pix*math.cos(math.radians(thetamodel))*ngtu,realv_pix*math.sin(math.radians(thetamodel))*ngtu,head_width=0.7, head_length=0.7, length_includes_head=True, color="b", label = labelmodel)
    #qui ci va seconda freccia dopo aver trovato fattore di conversione
    iani = 0
    im = plt.imshow(data[0,:,:],animated=True,cmap="hot",extent=(0,pixelcloud*2+2,0,pixelcloud*2+2))
    plt.clim(0.9*minvalue,1.1*maxvalue)
    plt.colorbar()
    plt.legend(loc="best")

    def updatefig(*args):
        global iani
        iani = iani + 1
        if (iani>ngtu-1): iani = 0
        plt.title(titlemodel + "max = (%i,%i), GTU = %i" % (xpix-xpixmin,ypix-ypixmin,gtupix+iani))
        im.set_array(data[iani,:,:])
        plt.clim(0.9*minvalue,1.1*maxvalue)
        return im,

    ani = animation.FuncAnimation(f, updatefig, blit=True)
    ani.save(pngdir + "rec_video.gif", writer='pillow', fps=1.5)
    f.clear()
    plt.close(f)

def train_net(netname,trainmode,n_epochs,learningrate,pixelcloud,ngtu,xmet,ymet,gtumet,magmet,idxmet,stacktheta):

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
    for entry in range(128): back[entry,:,:] = np.rot90(back[entry,:,:],1) #i dati sono ruotati -> così è come su ETOS

    #range pixel
    xpixmin = max(xmet-pixelcloud,0)
    ypixmin = max(ymet-pixelcloud,0)
    xpixmax = min(xmet+pixelcloud,47)
    ypixmax = min(ymet+pixelcloud,47)

    if (xpixmin==0): xpixmax = pixelcloud*2 
    if (ypixmin==0): ypixmax = pixelcloud*2
    if (xpixmax==47): xpixmin = 47 - pixelcloud*2
    if (ypixmax==47): ypixmin = 47 - pixelcloud*2

    x_in = torch.Tensor([xmet-xpixmin,ymet-ypixmin]) #lo sposto a 0 e pixelcloud
    x_in = x_in.view(1,dim_in)

    #inizializzazione modello
    if (netname=="FullOde"): net = FullOdeNet(dim_in,dim_out,hidden_size,n_fourier,x_in,skipopt)
    if (netname=="SirenOde"): net = SirenOdeNet(dim_in,dim_out,hidden_size,n_layers,x_in)

    if (len(learningrate)==1):
        modeldir = netname + "_Adam_%f_%i_%i_tmode%i" % (learningrate[0],pixelcloud,ngtu,trainmode)
        modelname = netname + " - Adam (lr = %f) - (%ix%ix%i) - Train %i" % (learningrate[0],pixelcloud,pixelcloud,ngtu,trainmode)
    if (len(learningrate)==2):
        modeldir = netname + "_Adam_%f_%f_%i_%i_tmode%i" % (learningrate[0],learningrate[1],pixelcloud,ngtu,trainmode)
        modelname = netname + " - Adam (lr = %f, lr = %f) - (%ix%ix%i) - Train %i" % (learningrate[0],learningrate[1],pixelcloud,pixelcloud,ngtu,trainmode)

    print("---------------------")
    print("Modello Fisico: %s" % modelname) 

    metvideo = np.zeros((ngtu,pixelcloud*2+1,pixelcloud*2+1))
    for igtu,gtu in enumerate(range(gtumet,gtumet+ngtu)): metvideo[igtu,:,:] = back[gtu,47-ypixmax:48-ypixmin,xpixmin:xpixmax+1]

    #----------------------
    #------TRAINING--------
    #----------------------

    #funzione di loss e ottimizzatore
    lossfunction = nn.MSELoss() #distanza al quadrato
    #if (len(learningrate)==1): optimizer = optim.Adam(net.parameters(),lr = learningrate[0],betas=[0.99954,0.99999])
    #if (len(learningrate)==2): optimizer = optim.Adam([{'params': [par for par in net.parameters() if par.name!="speed"]},{'params': [par for par in net.parameters() if par.name=="speed"], 'lr': learningrate[1], 'betas': [0.95,0.99999]}], lr = learningrate[0], betas=[0.99954,0.99999])
    if (len(learningrate)==1): optimizer = optim.Adam(net.parameters(),lr = learningrate[0])
    if (len(learningrate)==2): optimizer = optim.Adam([{'params': [par for par in net.parameters() if par.name!="speed"]},{'params': [par for par in net.parameters() if par.name=="speed"], 'lr': learningrate[1]}], lr = learningrate[0])

    losslist = [] #per plottare
    resthlist,resvlist = [],[] #residuo velocità e tempo
    timelist = [] #plotto anche il tempo 
    ilosslist = -1

    print("Beginning Training!")

    if (trainmode==0): #video è una epoca totale

        for e in range(n_epochs):

            lossepoch = 0.
            if (losslist and e%(n_epochs/10)==0): print("Epoch %i -> Loss %f" % (e,losslist[ilosslist]))
            starttime = time.time()

            #loop temporale
            for t in range(ngtu):
                #loop nell'immagine
                for x in range(xpixmin,xpixmax+1):
                    for y in range(ypixmin,ypixmax+1):

                        #pixel shiftato
                        pixinput = torch.Tensor([x-xpixmin,y-ypixmin])
                        pixinput = pixinput.view(1,dim_in)

                        #azzero i parametri del gradiente
                        optimizer.zero_grad()

                        #forward
                        outputs = net(pixinput,t) #calcolo l'output
                        trueimage = metvideo[t,ypixmax-y,x-xpixmin] #l = 47 - y -> lmin = 47 - ymax -> l-lmin = ymax - y
                        trueimage = torch.Tensor([[trueimage]])
                        loss = lossfunction(outputs,trueimage) #calcolo loss
                        
                        #backward
                        loss.backward()

                        #optimizer
                        optimizer.step() #1 learning rate step 

                        #update list
                        lossepoch = lossepoch + loss.item()

            stoptime = time.time()
            timelist.append(stoptime-starttime)

            #prendo residuo ad ogni epoch
            for name, param in net.named_parameters(): 
                if(param.requires_grad and name=="speed"): vx,vy = param.data[0],param.data[1]
            vapp,theta = math.sqrt(vx*vx + vy*vy),math.degrees(math.atan2(vy,vx))
            if (theta<0): theta = theta + 360

            resvlist.append(abs(vapp-realv))
            
            restheta = theta-realth
            if (restheta>180): restheta = restheta - 360
            if (restheta<(-180)): restheta = 360 + restheta
            resthlist.append(abs(restheta))

            lossepoch = lossepoch/(ngtu*(pixelcloud*2+1)*(pixelcloud*2+1))
            losslist.append(lossepoch)
            ilosslist = ilosslist + 1
    
    if (trainmode==1): #training frame per frame 

        #loop temporale
        for t in range(ngtu):

            if (losslist): print("Frame %i -> Loss %f" % (t,losslist[ilosslist]))
            for e in range(n_epochs):

                if (e%ngtu==0):
                    lossepoch = 0.
                    starttime = time.time()

                #loop nell'immagine
                for x in range(xpixmin,xpixmax+1):
                    for y in range(ypixmin,ypixmax+1):

                        #pixel shiftato
                        pixinput = torch.Tensor([x-xpixmin,y-ypixmin])
                        pixinput = pixinput.view(1,dim_in)

                        #azzero i parametri del gradiente
                        optimizer.zero_grad()

                        #forward
                        outputs = net(pixinput,t) #calcolo l'output
                        trueimage = metvideo[t,ypixmax-y,x-xpixmin] #l = 47 - y -> lmin = 47 - ymax -> l-lmin = ymax - y
                        trueimage = torch.Tensor([[trueimage]])
                        loss = lossfunction(outputs,trueimage) #calcolo loss
                        
                        #backward
                        loss.backward()

                        #optimizer
                        optimizer.step() #1 learning rate step 

                        #update list
                        lossepoch = lossepoch + loss.item()

                if (e%ngtu==ngtu-1):
                    stoptime = time.time()
                    timelist.append(stoptime-starttime)

                    #prendo residuo ad ogni epoch
                    for name, param in net.named_parameters(): 
                        if(param.requires_grad and name=="speed"): vx,vy = param.data[0],param.data[1]
                    vapp,theta = math.sqrt(vx*vx + vy*vy),math.degrees(math.atan2(vy,vx))
                    if (theta<0): theta = theta + 360

                    resvlist.append(abs(vapp-realv))
                    
                    restheta = theta-realth
                    if (restheta>180): restheta = restheta - 360
                    if (restheta<(-180)): restheta = 360 + restheta
                    resthlist.append(abs(restheta))

                    lossepoch = lossepoch/(ngtu*(pixelcloud*2+1)*(pixelcloud*2+1))
                    losslist.append(lossepoch)
                    ilosslist = ilosslist + 1


    print("Finished Training!")
    for name, param in net.named_parameters(): 
        if(param.requires_grad and name=="speed"): vx,vy = param.data[0],param.data[1]
    vapp,theta = math.sqrt(vx*vx + vy*vy),math.degrees(math.atan2(vy,vx))
    if (theta<0): theta = theta + 360

    print("Velocità Modello = %.3f ? (%.3f,%.3f)" % (vapp,vx,vy))
    print("Direzione Modello = %.3f °" % (theta))
    print("---------------------")

    #video ricostruito
    recvideo2d = np.zeros((ngtu,pixelcloud*2+1,pixelcloud*2+1))

    #loop temporale
    for t in range(ngtu):

        #loop nell'immagine
        for x in range(xpixmin,xpixmax+1):
            for y in range(ypixmin,ypixmax+1):
                pixinput = torch.Tensor([x-xpixmin,y-ypixmin])
                phi = net(pixinput,t)
                recvideo2d[t,ypixmax-y,x-xpixmin] = phi.data[0][0]

    if (drawopt): plot_video(recvideo2d,magmet,idxmet,xmet,ymet,gtumet,pixelcloud,ngtu,vapp,theta,modeldir,modelname)

    #plotting results 
    pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Training/%s/%s/" % (filename,modeldir)
    if(not os.path.exists(pngdir)): os.makedirs(pngdir)

    elist = [e for e in range(n_epochs)] #asse x comune a tutti i plot 

    #learning curve loss
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Mean Squared Error [MSE]")
    plt.yscale("log")
    plt.plot(elist,losslist,c = "b",linestyle="dashed",marker="o")
    plt.savefig(pngdir + "learningcurve_loss.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #learning curve res speeed
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Apparent Speed Residual [km/s]")
    plt.plot(elist,resvlist,c = "b",linestyle="dashed",marker="o")
    plt.savefig(pngdir + "learningcurve_speed.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #learning curve res theta
    stackresth = stacktheta-realth
    if (stackresth>180): stackresth = stackresth - 360
    if (stackresth<(-180)): stackresth = 360 + stackresth

    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Apparent Direction Residual [°]")
    plt.axhline(abs(stackresth),c="g",linestyle="dashdot")
    plt.plot(elist,resthlist,c = "b",linestyle="dashed",marker="o")
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
    plt.plot(elist,timelist,c = "b",linestyle="dashed",marker="o")
    plt.savefig(pngdir + "learningcurve_time.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    return losslist,resvlist,resthlist,timelist,modelname