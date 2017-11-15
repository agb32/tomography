#mpirun -np 1 -hostlist n1-c437  /usr/local/bin/mpipython $PWD/thisfile.py
#Python code created using the simulation setup GUI...
#Order of execution may not be quite optimal - you can always change by hand
#for large simulations - typically, the order of sends and gets may not be
#quite right.  Anyway, enjoy...
import numpy
import util.Ctrl
import base.mpiGet
import base.mpiSend
import base.shmGet
import base.shmSend
import science.tomoRecon
import science.xinterp_dm
import science.wfscent
ctrl=util.Ctrl.Ctrl(globals=globals())
print "Rank %d imported modules"%ctrl.rank
#Set up the science modules...
newMPIGetList=[]
newMPISendList=[]
newSHMGetList=[]
newSHMSendList=[]
reconList=[]
dmList=[]
wfscentList=[]
if not "nopoke" in ctrl.userArgList:ctrl.doInitialPoke()
#Add any personal code after this line and before the next, and it won't get overwritten
if ctrl.rank==0:
    dmList.append(science.xinterp_dm.dm(None,ctrl.config,args={},idstr="dm0path1"))
    wfscentList.append(science.wfscent.wfscent(dmList[0],ctrl.config,args={},idstr="1"))
    dmList.append(dmList[0].addNewIdObject(None,"dm0path2"))
    wfscentList.append(science.wfscent.wfscent(dmList[1],ctrl.config,args={},idstr="2"))
    dmList.append(dmList[0].addNewIdObject(None,"dm0path3"))
    wfscentList.append(science.wfscent.wfscent(dmList[2],ctrl.config,args={},idstr="3"))
    reconList.append(science.tomoRecon.recon({"1":wfscentList[0],"2":wfscentList[1],"3":wfscentList[2],},ctrl.config,args={},idstr="recon"))
    dmList[0].newParent(reconList[0],"dm0path1")
    dmList[1].newParent(reconList[0],"dm0path2")
    dmList[2].newParent(reconList[0],"dm0path3")
    execOrder=[dmList[0],wfscentList[0],dmList[1],wfscentList[1],dmList[2],wfscentList[2],reconList[0],]
    ctrl.mainloop(execOrder)
print "Simulation finished..."
#Add any personal code after this, and it will not get overwritten
