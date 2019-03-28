#get Rcpi results from folder CSVs

import os

sFolder = "Rcpi"
no=0
list = os.listdir(sFolder) # dir is your directory path
print "No,CanSMILES,ALogP,ALogp2,AMR,apol,nA,nR,nN,nD,nC,nF,nQ,nE,nG,nH,nI,nP,nL,nK,nM,nS,nT,nY,nV,nW,naAromAtom,nAromBond,nAtom,ATSc1,ATSc2,ATSc3,ATSc4,ATSc5,ATSm1,ATSm2,ATSm3,ATSm4,ATSm5,ATSp1,ATSp2,ATSp3,ATSp4,ATSp5,BCUTw.1l,BCUTw.1h,BCUTc.1l,BCUTc.1h,BCUTp.1l,BCUTp.1h,bpol,nB,PPSA.1,PPSA.2,PPSA.3,PNSA.1,PNSA.2,PNSA.3,DPSA.1,DPSA.2,DPSA.3,FPSA.1,FPSA.2,FPSA.3,FNSA.1,FNSA.2,FNSA.3,WPSA.1,WPSA.2,WPSA.3,WNSA.1,WNSA.2,WNSA.3,RPCG,RNCG,RPCS,RNCS,THSA,TPSA,RHSA,RPSA,C1SP1,C2SP1,C1SP2,C2SP2,C3SP2,C1SP3,C2SP3,C3SP3,C4SP3,SCH.3,SCH.4,SCH.5,SCH.6,SCH.7,VCH.3,VCH.4,VCH.5,VCH.6,VCH.7,SC.3,SC.4,SC.5,SC.6,VC.3,VC.4,VC.5,VC.6,SPC.4,SPC.5,SPC.6,VPC.4,VPC.5,VPC.6,SP.0,SP.1,SP.2,SP.3,SP.4,SP.5,SP.6,SP.7,VP.0,VP.1,VP.2,VP.3,VP.4,VP.5,VP.6,VP.7,ECCEN,FMF,fragC,GRAV.1,GRAV.2,GRAV.3,GRAVH.1,GRAVH.2,GRAVH.3,GRAV.4,GRAV.5,GRAV.6,nHBAcc,nHBDon,HybRatio,MolIP,Kier1,Kier2,Kier3,khs.sLi,khs.ssBe,khs.ssssBe,khs.ssBH,khs.sssB,khs.ssssB,khs.sCH3,khs.dCH2,khs.ssCH2,khs.tCH,khs.dsCH,khs.aaCH,khs.sssCH,khs.ddC,khs.tsC,khs.dssC,khs.aasC,khs.aaaC,khs.ssssC,khs.sNH3,khs.sNH2,khs.ssNH2,khs.dNH,khs.ssNH,khs.aaNH,khs.tN,khs.sssNH,khs.dsN,khs.aaN,khs.sssN,khs.ddsN,khs.aasN,khs.ssssN,khs.sOH,khs.dO,khs.ssO,khs.aaO,khs.sF,khs.sSiH3,khs.ssSiH2,khs.sssSiH,khs.ssssSi,khs.sPH2,khs.ssPH,khs.sssP,khs.dsssP,khs.sssssP,khs.sSH,khs.dS,khs.ssS,khs.aaS,khs.dssS,khs.ddssS,khs.sCl,khs.sGeH3,khs.ssGeH2,khs.sssGeH,khs.ssssGe,khs.sAsH2,khs.ssAsH,khs.sssAs,khs.sssdAs,khs.sssssAs,khs.sSeH,khs.dSe,khs.ssSe,khs.aaSe,khs.dssSe,khs.ddssSe,khs.sBr,khs.sSnH3,khs.ssSnH2,khs.sssSnH,khs.ssssSn,khs.sI,khs.sPbH3,khs.ssPbH2,khs.sssPbH,khs.ssssPb,nAtomLC,nAtomP,LOBMAX,LOBMIN,nAtomLAC,MDEC.11,MDEC.12,MDEC.13,MDEC.14,MDEC.22,MDEC.23,MDEC.24,MDEC.33,MDEC.34,MDEC.44,MDEO.11,MDEO.12,MDEO.22,MDEN.11,MDEN.12,MDEN.13,MDEN.22,MDEN.23,MDEN.33,MLogP,MOMI.X,MOMI.Y,MOMI.Z,MOMI.XY,MOMI.XZ,MOMI.YZ,MOMI.R,PetitjeanNumber,topoShape,geomShape,nRotB,LipinskiFailures,TopoPSA,VABC,VAdjMat,Wlambda1.unity,Wlambda2.unity,Wlambda3.unity,Wnu1.unity,Wnu2.unity,Wgamma1.unity,Wgamma2.unity,Wgamma3.unity,Weta1.unity,Weta2.unity,Weta3.unity,WT.unity,WA.unity,WV.unity,WK.unity,WG.unity,WD.unity,MW,WTPT.1,WTPT.2,WTPT.3,WTPT.4,WTPT.5,WPATH,WPOL,XLogP,Zagreb"
for item in list:
    if item[-3:]=="csv":
        sParts = item.split(".")
        f=open(os.path.join(sFolder,item),"r")
        lines = f.readlines()
        print str((item.split(".")[0]).split("-")[1])+","+lines[1][:-1]
        f.close()
        no +=1

#print "CSV files =", no
