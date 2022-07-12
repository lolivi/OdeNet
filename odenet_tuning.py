import odenet_pytorch
from odenet_pytorch import *

#hyperparameters to be tuned
n_epochs = 100 #epoche -> taglio quando i residui sono piatti
trainmodes = [0] #0 considera come epoca tutto il video, 1 divide epoche per numero di frame addestra frame per frame 
nets_list = ["FullOde","SirenOde"] #due architetture
ngtu_list = [10] #numero frame video 
pixelcloud_list = [5] #frame immagine
lrlist = [[0.00001],[0.00001,0.01]]
ncombo = len(nets_list)*len(ngtu_list)*len(pixelcloud_list)*len(lrlist)*len(trainmodes)

#dataset training
#amag_list = [4,4,4,4,4,4,4,4]
#idx_list = [67,85,83,94,10,98,9,47]
amag_list = [4]
idx_list = [83]
n_events = len(amag_list)
gtu_list = [None for i in range(n_events)]
stacktheta = [None for i in range(n_events)]
stackspeed = [None for i in range(n_events)]
pix_list = [[None,None] for i in range(n_events)]

#leggo dati da excel
excel = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Stack CNN Meteore - Immagini/SIM - Solo Trigger Finale/Analisi SIM.xlsx"
sheetname = "StackCNN_RF - Clean"

if os.path.isfile(excel):
    print("Leggendo il File Excel %s w/ Sheet %s" % (excel,sheetname))
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
        if(cellname =="GTU"): gtumax = currentcell
        if(cellname =="X"): xmax = currentcell
        if(cellname =="Y"): ymax = currentcell
        if(cellname =="V [km/s]"): vmet = currentcell
        if(cellname =="THETA [°]"): thetamet = currentcell
        if(cellname =="RF PROB [%]"): prf = currentcell
        if(cellname =="CNN PROB [%]"): pcnn = currentcell
    #fine loop su colonne excel

    #check se riga ha dati o no
    if (file == None): continue

    magfile,idxfile = -1,-1
    for m in range(4,7):
        for i in range(100):
            if (i<10): file_test = "amag_p%i0_000%i" % (m,i)
            else: file_test = "amag_p%i0_00%i" % (m,i)
            if (file_test == file): magfile,idxfile = m,i

    if (magfile in amag_list and idxfile in idx_list): 
        idxgtu = idx_list.index(idxfile)
        stacktheta[idxgtu] = thetamet
        stackspeed[idxgtu] = vmet
        gtu_list[idxgtu] = gtumax
        pix_list[idxgtu] = [xmax,ymax]

#fine lettura dataset training

#inizio addestramento
for data in range(n_events):

    #dati meteora triggerati 
    xdata,ydata = pix_list[data][0],pix_list[data][1]
    gtudata = gtu_list[data]
    mdata,idxdata = amag_list[data],idx_list[data]
    filedir = buildstring(mdata,idxdata)

    print("Training (M,idx) = (%i,%i)" % (mdata,idxdata))

    stackth = stacktheta[data] #theta stackcnn

    lossplot = [] #loss per ogni config
    resvplot,resthplot = [],[] #residui per ogni config
    timeplot = [] #tempo per ogni config
    labels = [] #nome di ogni combo
    icombo = 0 #conteggi combo

    for model in nets_list:
        for tmode in trainmodes:
            for ngtu in ngtu_list:
                for pixcloud in pixelcloud_list:
                    for lr in lrlist:

                        icombo = icombo + 1
                        print("Combo %i / %i (%.2f %s)" % (icombo,ncombo,icombo/ncombo*100.,"%"))

                        losscombo,resvcombo,resthcombo,timecombo,label = train_net(model,tmode,n_epochs,lr,pixcloud,ngtu,xdata,ydata,gtudata,mdata,idxdata,stackth)

                        lossplot.append(losscombo)
                        resvplot.append(resvcombo)
                        resthplot.append(resthcombo)
                        timeplot.append(timecombo)
                        labels.append(label)    

    #plotting results 
    pngdir = userdir + "Documenti/Università Fisica/Magistrale - Nucleare e Subnucleare/Tesi Magistrale - Detriti/Modello Fisico Meteore - Immagini/Dataset Training/%s/" % filedir
    if(not os.path.exists(pngdir)): os.makedirs(pngdir)

    cm = plt.get_cmap("gist_rainbow")
    ncol = 5 #plotto le migliori 5
    ncol = min(ncol,ncombo)
    colors = [cm(1.*i/ncol) for i in range(ncol)]

    elist = [e for e in range(n_epochs)] #asse x comune a tutti i plot 
    minresth = [resthlist[n_epochs-1] for resthlist in resthplot if resthlist[n_epochs-1]!=0] #quelli identici a 0 non sono neanche addestrati
    full_sort_index = np.argsort(minresth) #l'ultimo residuo

    #minloss = [min(loss) for loss in lossplot]
    #full_sort_index = np.argsort(minloss)
    sort_index = full_sort_index[0:ncol].tolist()

    #learning curve loss
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Mean Squared Error [MSE]")
    plt.yscale("log")
    for icol,reali in enumerate(sort_index): plt.plot(elist,lossplot[reali],c = cm(1.*icol/ncol),label=labels[reali],linestyle="dashed",marker="o")
    plt.legend(bbox_to_anchor=(1.04,1.1), loc="upper left")
    plt.savefig(pngdir + "learningcurve_loss.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)

    #learning curve res speeed
    f = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Epoch Number")
    plt.ylabel("Apparent Speed Residual [km/s]")
    #plt.yscale("log")
    for icol,reali in enumerate(sort_index): plt.plot(elist,resvplot[reali],c = cm(1.*icol/ncol),label=labels[reali],linestyle="dashed",marker="o")
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
    for icol,reali in enumerate(sort_index): plt.plot(elist,resthplot[reali],c = cm(1.*icol/ncol),label=labels[reali],linestyle="dashed",marker="o")
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
    for icol,reali in enumerate(sort_index): plt.plot(elist,timeplot[reali],c = cm(1.*icol/ncol),label=labels[reali],linestyle="dashed",marker="o")
    plt.legend(bbox_to_anchor=(1.04,1.1), loc="upper left")
    plt.savefig(pngdir + "learningcurve_time.png",bbox_inches = "tight")
    f.clear()
    plt.close(f)