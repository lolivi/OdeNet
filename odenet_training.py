import odenet_pytorch
from odenet_pytorch import *

firstwrite = True #true se si vuole sovrascrivere txt

#hyperparameters tuned
tuning = False #training directory
n_epochs = 50 #epoche -> taglio quando i residui sono piatti
init = 2 #inizializzazione angolo e velocità (0 = No init, 1 = StackCNN, 2 = Baseline)
tmode = 0 #0 considera come epoca tutto il video, 1 divide epoche per numero di frame addestra frame per frame 
model = "FullOde" #due architetture
ngtu = 10 #numero frame video 
pixelcloud = 5 #frame immagine
exprate = 1 #update esponenziale learning rate
lr = [0.001] #learning rate
hyperpars_train = [model,n_epochs,lr,exprate,pixelcloud,ngtu]

#plotting results 
pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Training/"
if(not os.path.exists(pngdir)): os.makedirs(pngdir)
outfile = pngdir + "Net_Training.txt"

#lettura txt
txt_data = []
for m in range(4,7):
    for i in range(100):
        if (i%10==0): print("Leggendo Dati (%i,%i)" % (m,i))
        txt_data.append(read_txt_data(m,i))

#lettura excel
met_results,fake_cnn_4,fake_cnn_5,fake_cnn_6,fake_pix_4,fake_pix_5,fake_pix_6,fake_rf_4,fake_rf_5,fake_rf_6 = read_excel(txt_data)
fake_cnn = fake_cnn_4 + fake_cnn_5 + fake_cnn_6
fake_rf = fake_rf_4 + fake_rf_5 + fake_rf_6

#simulated variables and reconstructed variables
simvariables = [] #realv, realvapp, viss, phiiss, theta, phi
stackvariables = [] #vapp, theta, v, phi
odevariables = [] #loss, time, vapp, theta
basevariables = [] #vapp, theta
physvariables = [] #tutte le variabili simulate per scatter 2d

#odenet variables
bettercombo = 0 #conta residui odenet minori stackCNN
wrongfiles = [] #lista indici eventi falliti
cntfile = 0

for i in range(len(met_results)):
    
    kidx = met_results[i][0]
    pcnn,prf = met_results[i][1],met_results[i][2]
    deltapix = met_results[i][8]
    mdata = txt_data[kidx][2]
    if (mdata==4): idxdata = kidx
    if (mdata==5): idxdata = kidx - 100
    if (mdata==6): idxdata = kidx - 200

    print("\n- Evento %i / %i" % (i+1,len(met_results)))
    cntfile = cntfile + 1

    #residui stackCNN
    if not(pcnn>cutcnn*100 and deltapix==1 and prf>cutrf*100): continue 

    resvapp_stackcnn = met_results[i][9] - txt_data[kidx][33]*pixelkm/gtusec
    resth_stackcnn = met_results[i][10] - txt_data[kidx][32]
    if (resth_stackcnn>180): resth_stackcnn = resth_stackcnn - 360
    if (resth_stackcnn<(-180)): resth_stackcnn = 360 + resth_stackcnn

    #dati meteora triggerati 
    xdata,ydata = met_results[i][11],met_results[i][12]
    gtudata = met_results[i][13]
    filedir = buildstring(mdata,idxdata)

    #prendo veri parametri simulati
    xpixin,xpixfin = txt_data[kidx][28],txt_data[kidx][29]
    ypixin,ypixfin = txt_data[kidx][30],txt_data[kidx][31]
    realth,realv_pix = txt_data[kidx][32],txt_data[kidx][33] #in deg e pix/gtu
    realv,realphi = txt_data[kidx][5]*math.cos(math.radians(txt_data[kidx][13])), txt_data[kidx][14] #in km/s
    viss, phiiss = txt_data[kidx][16], txt_data[kidx][17]
    simvariables.append([realv,realv_pix,viss,phiiss,realth,realphi])
    physvariables.append([txt_data[kidx][1],txt_data[kidx][2],txt_data[kidx][3],txt_data[kidx][4],txt_data[kidx][5],txt_data[kidx][6],txt_data[kidx][7],txt_data[kidx][8],txt_data[kidx][9],txt_data[kidx][10],txt_data[kidx][11],txt_data[kidx][12],txt_data[kidx][13],txt_data[kidx][14],txt_data[kidx][18],txt_data[kidx][19],(txt_data[kidx][22]+txt_data[kidx][23])/2.,txt_data[kidx][33]*pixelkm/gtusec,txt_data[kidx][32]])

    #leggo dati veri simulati
    inputtext = userdir + 'Documenti/ETOS/Dati/SIM/%s.txt' % (filedir)
    if os.path.isfile(inputtext): a = np.loadtxt(inputtext) #crea array di numpy con dati in ordine
    else: sys.exit() 

    back = np.reshape(a,(128,48,48)) #fa un reshape dei dati nell'array a con lunghezza array nel primo, righe e colonne
    backmedian = np.zeros((128,48,48))
    for entry in range(128): back[entry,:,:] = np.rot90(back[entry,:,:],1) #i dati sono ruotati -> così è come su ETOS

    if (medianopt):
        for entry in range(128): 
            tmin,tmax = max(entry-4,0),min(entry+5,128)
            backmedian[entry,:,:] = back[entry,:,:] / np.median(back[tmin:tmax,0:48,0:48],axis = 0) 

    #range pixel
    xpixmin = max(xdata-pixelcloud,0)
    ypixmin = max(ydata-pixelcloud,0)
    xpixmax = min(xdata+pixelcloud,47)
    ypixmax = min(ydata+pixelcloud,47)

    if (xpixmin==0): xpixmax = pixelcloud*2 
    if (ypixmin==0): ypixmax = pixelcloud*2
    if (xpixmax==47): xpixmin = 47 - pixelcloud*2
    if (ypixmax==47): ypixmin = 47 - pixelcloud*2

    #stack variables
    stackth = met_results[i][10] #theta stackcnn
    stackvapp = met_results[i][9] #v apparente [km/s]
    stackphi = met_results[i][5] #phi stackcnn
    stackv = met_results[i][4] #real v (km/s)
    stackvariables.append([stackvapp,stackth,stackv,stackphi])
    stackvars_train = [xdata,ydata,gtudata,stackvapp,stackth]
    
    print("Training (M,idx) = (%i,%i)" % (mdata,idxdata))
    label,gtuin,odevars,basevars = train_net(mdata,idxdata,hyperpars_train,stackvars_train,init,tuning)
    basev,baseth = basevars[2],basevars[3]

    #baseline variables
    basevariables.append([basev,baseth])
    
    #odenet variables
    lossfin = odevars[0][n_epochs-1]
    resvfin = odevars[1][n_epochs-1]
    resthfin = odevars[2][n_epochs-1]
    timetot = np.sum(odevars[3]) 
    
    vapp_odenet = resvfin + realv_pix
    th_odenet = resthfin + realth
    th_odenet = th_odenet - int(th_odenet/360)*360
    if (th_odenet<0): th_odenet = th_odenet + 360
    odevariables.append([lossfin,timetot,vapp_odenet,th_odenet]) #loss, time, vapp, theta

    #if (lossfin>0.0005): print("Warning! Video Ricostruito Erroneamente!") 
 
    if (abs(resthfin)<abs(resth_stackcnn)): bettercombo = bettercombo + 1
    if (abs(resthfin)>100): wrongfiles.append(idxdata)
    print("Percentuale Res OdeNet < Res Stack-CNN = %.2f %s" % (bettercombo*100./cntfile,"%"))

    if (not firstwrite): 
        opt = "a" #append
        header = False
    if (firstwrite):
        opt = "w"
        firstwrite = False
        header = True

    with open(outfile,opt) as fout: #scrittura

        if (header): fout.write("Mag  Idx  Loss  Time  OdeNet Theta  OdeNet Speed  Baseline Theta  Baseline Speed\n")
        writecontent = "%i %i %f %f %f %f %f %f\n" % (mdata,idxdata,lossfin,timetot,th_odenet,vapp_odenet,baseth,basev)
        fout.write(writecontent)
    
    fout.close()

    if (cntfile>10): 
        print(wrongfiles)
        plot_residuals(simvariables,stackvariables,odevariables,basevariables,physvariables)

if (cntfile>10): plot_residuals(simvariables,stackvariables,odevariables,basevariables,physvariables)
print("Wrong Files: ")
print(wrongfiles)

