import odenet_pytorch
from odenet_pytorch import *

#hyperparameters to be tuned
tuning = True #opzione directory
n_epochs = 50 #epoche -> taglio quando i residui sono piatti
init_list = [2] #inizializzazione angolo e velocità (0 = No init, 1 = StackCNN, 2 = Baseline)
nets_list = ["FullOde","SirenOde"] #due architetture
ngtu_list = [10] #numero frame video 
pixelcloud_list = [5] #frame immagine
expratelist = [1]
lrlist = [[0.01],[0.001],[0.0001]]
ncombo = len(init_list)*len(nets_list)*len(ngtu_list)*len(pixelcloud_list)*len(lrlist)*len(expratelist)

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

best_model_index = np.zeros(ncombo) #indice miglior modello
weights = np.zeros(ncombo)
resthdistro,resvdistro = [],[]
lossdistro = []

#inizio addestramento
for i in range(len(met_results)):
    
    kidx = met_results[i][0]
    pcnn,prf = met_results[i][1],met_results[i][2]
    deltapix = met_results[i][8]
    mdata = txt_data[kidx][2]
    if (mdata==4): idxdata = kidx
    if (mdata==5): idxdata = kidx - 100
    if (mdata==6): idxdata = kidx - 200

    if not(pcnn>cutcnn*100 and deltapix==1 and prf>cutrf*100): continue
    if (mdata!=4): continue

    #dati meteora triggerati 
    xdata,ydata = met_results[i][11],met_results[i][12]
    gtudata = met_results[i][13]
    filedir = buildstring(mdata,idxdata)

    #prendo veri parametri simulati
    xpixin,xpixfin = txt_data[kidx][28],txt_data[kidx][29]
    ypixin,ypixfin = txt_data[kidx][30],txt_data[kidx][31]
    realth,realv_pix = txt_data[kidx][32],txt_data[kidx][33] #in deg e pix/gtu
    realv = txt_data[kidx][5]*math.cos(math.radians(txt_data[kidx][13])) #in km/s

    print("Training (M,idx) = (%i,%i)" % (mdata,idxdata))

    stackth = met_results[i][10] #theta stackcnn
    stackvapp = met_results[i][9]
    stackvars = [xdata,ydata,gtudata,stackvapp,stackth]

    lossplot = [] #loss per ogni config
    resvplot,resthplot = [],[] #residui per ogni config
    timeplot = [] #tempo per ogni config
    labels = [] #nome di ogni combo
    icombo = 0 #conteggi combo

    for init in init_list:
        for model in nets_list:
            for ngtu in ngtu_list:
                for pixcloud in pixelcloud_list:
                    for exprate in expratelist:
                        for lr in lrlist:

                            icombo = icombo + 1
                            print("- Combo %i / %i (%.2f %s)" % (icombo,ncombo,icombo/ncombo*100.,"%"))
                            
                            hyperpars = [model,n_epochs,lr,exprate,pixcloud,ngtu]
                            label,gtuin,odevars,basevars = train_net(mdata,idxdata,hyperpars,stackvars,init,tuning)

                            lossplot.append(odevars[0])
                            resvplot.append(odevars[1])
                            resthplot.append(odevars[2])
                            timeplot.append(odevars[3])
                            labels.append(label)    

    #plotting results 
    pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Tuning/%s/" % filedir
    if(not os.path.exists(pngdir)): os.makedirs(pngdir)

    cm = plt.get_cmap("gist_rainbow")
    ncol = ncombo
    colors = [cm(1.*i/ncol) for i in range(ncol)]

    elist = [e for e in range(n_epochs)] #asse x comune a tutti i plot 
    minresth = [resthlist[n_epochs-1] for resthlist in resthplot if resthlist[n_epochs-1]!=0] #quelli identici a 0 non sono neanche addestrati
    minresv = [resvlist[n_epochs-1] for resvlist in resvplot]
    minloss = [losslist[n_epochs-1] for losslist in lossplot]
    full_sort_index = np.argsort(minresth) #l'ultimo residuo

    #minloss = [min(loss) for loss in lossplot]
    #full_sort_index = np.argsort(minloss)
    sort_index = full_sort_index[0:ncol].tolist() #lista contenente in ordine gli indici di quelli con res minore
    for icol,reali in enumerate(sort_index): 
        best_model_index[reali] = best_model_index[reali] + resthplot[reali][n_epochs-1]*icol #reali è l'indice del modello, icol è la posizione!
        weights[reali] = weights[reali] + resthplot[reali][n_epochs-1]

    resthdistro.append(minresth)
    resvdistro.append(minresv)
    lossdistro.append(minloss)
    sigresth = np.std(resthdistro,axis=0)

    #learning curve loss
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    if (losstype==1): plt.ylabel("Least Absolute Deviations [L1]")
    if (losstype==2): plt.ylabel("Mean Squared Error [MSE]")
    plt.yscale("log")
    for icol,reali in enumerate(sort_index): plt.plot(elist,lossplot[reali],c = cm(1.*icol/ncol),label=labels[reali])
    plt.legend(bbox_to_anchor=(1.04,1.1), loc="upper left")
    plt.savefig(pngdir + "learningcurve_loss.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #learning curve res speeed
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Apparent Speed Residual [pix/GTU]")
    plt.axhline(abs(stackvapp/pixelkm*gtusec-realv_pix),c="g",linestyle="dashdot",label="Stack-CNN")
    #plt.yscale("log")
    for icol,reali in enumerate(sort_index): plt.plot(elist,resvplot[reali],c = cm(1.*icol/ncol),label=labels[reali])
    plt.legend(bbox_to_anchor=(1.04,1.1), loc="upper left")
    plt.savefig(pngdir + "learningcurve_speed.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #learning curve res speeed
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Apparent Direction Residual [°]")
    #plt.yscale("log")
    plt.axhline(abs(stackth-realth),c="g",linestyle="dashdot",label="Stack-CNN")
    for icol,reali in enumerate(sort_index): plt.plot(elist,resthplot[reali],c = cm(1.*icol/ncol),label=labels[reali])
    plt.legend(bbox_to_anchor=(1.04,1.1), loc="upper left")
    plt.savefig(pngdir + "learningcurve_theta.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #learning curve time
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Training Time [s]")
    #plt.yscale("log")
    for icol,reali in enumerate(sort_index): plt.plot(elist,timeplot[reali],c = cm(1.*icol/ncol),label=labels[reali])
    plt.legend(bbox_to_anchor=(1.04,1.1), loc="upper left")
    plt.savefig(pngdir + "learningcurve_time.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #Plot 2D Correlazione residuo theta e loss
    f = plt.figure()
    plt.scatter([item for sublist in resthdistro for item in sublist],[item for sublist in lossdistro for item in sublist],marker = "o",s = 20,color = "b")
    plt.xlabel("Apparent Theta Residual [°]")
    if (losstype==1): plt.ylabel("Least Absolute Deviations [L1]")
    if (losstype==2): plt.ylabel("Mean Squared Error [MSE]")
    plt.title("2D OdeNet Loss - Residual")
    plt.grid()
    metpng = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Tuning/distro2d_loss_resth.png"
    plt.savefig(metpng)
    f.clear()
    plt.close(f)

    #Plot 2D Correlazione residuo vapp e loss
    f = plt.figure()
    plt.scatter([item for sublist in resvdistro for item in sublist],[item for sublist in lossdistro for item in sublist],marker = "o",s = 20,color = "b")
    plt.xlabel("Apparent Speed Residual [pix/GTU]")
    if (losstype==1): plt.ylabel("Least Absolute Deviations [L1]")
    if (losstype==2): plt.ylabel("Mean Squared Error [MSE]")
    plt.title("2D OdeNet Loss - Residual")
    plt.grid()
    metpng = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Tuning/distro2d_loss_resvapp.png"
    plt.savefig(metpng)
    f.clear()
    plt.close(f)

    #best_model = np.argsort(best_model_index/weights)
    best_model = np.argsort(sigresth)
    best_combo = int(best_model[0])

    print("----------------------------------------")
    print("Best Model (Min Direction Residual)")
    print(labels[best_combo])
    print("Best Models in Order")
    for ipos,imodel in enumerate(best_model): print("- Pos%i: %s w/ sigma = %f" % (ipos,labels[imodel],sigresth[imodel]))
    print("Numeratore: ")
    print(best_model_index)
    print("Denominatore: ")
    print(weights)
    print("----------------------------------------")

#best_model = np.argsort(best_model_index/weights)
best_model = np.argsort(sigresth)
best_combo = int(best_model[0])

print("----------------------------------------")
print("Best Model (Min Direction Residual)")
print(labels[best_combo])
print("Best Models in Order")
for ipos,imodel in enumerate(best_model): print("- Pos%i: %s w/ sigma = %f" % (ipos,labels[imodel],sigresth[imodel]))
print("----------------------------------------")