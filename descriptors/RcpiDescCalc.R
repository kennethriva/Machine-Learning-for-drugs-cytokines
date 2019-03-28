# calculate the RCpi descriptors for a list of SMILES

options( java.parameters = "-Xmx5g" )
library(Rcpi)

# FUNCTIONS ----------------------------------------------------------------------------------------------------
extractDrugAIOwithParams <- function (molecules, descNames, silent = TRUE, warn = TRUE) 
{
  if (warn == TRUE) {
    warning("Note that we need 3-D coordinates of the molecules to calculate some of the descriptors, if not provided, these descriptors will be NA")
  }
  
  x = rcdk::eval.desc(molecules, descNames, verbose = !silent)
  return(x)
}
# ---------------------------------------------------------------------------------------------------------------


ptm <- proc.time()

print("Rcpi calculation of all molecular descriptors")

# set the descriptors to calculate
descNames = c("org.openscience.cdk.qsar.descriptors.molecular.ALOGPDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.APolDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.AminoAcidCountDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.AromaticAtomsCountDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.AromaticBondsCountDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.AtomCountDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.AutocorrelationDescriptorCharge", 
              "org.openscience.cdk.qsar.descriptors.molecular.AutocorrelationDescriptorMass", 
              "org.openscience.cdk.qsar.descriptors.molecular.AutocorrelationDescriptorPolarizability", 
              # "org.openscience.cdk.qsar.descriptors.molecular.BCUTDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.BPolDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.BondCountDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.CPSADescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.CarbonTypesDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.ChiChainDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.ChiClusterDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.ChiPathClusterDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.ChiPathDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.EccentricConnectivityIndexDescriptor", 
              # "org.openscience.cdk.qsar.descriptors.molecular.FMFDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.FragmentComplexityDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.GravitationalIndexDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.HBondAcceptorCountDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.HBondDonorCountDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.HybridizationRatioDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.IPMolecularLearningDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.KappaShapeIndicesDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.KierHallSmartsDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.LargestChainDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.LargestPiSystemDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.LengthOverBreadthDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.LongestAliphaticChainDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.MDEDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.MannholdLogPDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.MomentOfInertiaDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.PetitjeanNumberDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.PetitjeanShapeIndexDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.RotatableBondsCountDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.RuleOfFiveDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.TPSADescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.VABCDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.VAdjMaDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.WHIMDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.WeightDescriptor", 
              # "org.openscience.cdk.qsar.descriptors.molecular.WeightedPathDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.WienerNumbersDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.XLogPDescriptor", 
              "org.openscience.cdk.qsar.descriptors.molecular.ZagrebIndexDescriptor")


# folders with the input and output files
inFolder ="smi"
outFolder="Rcpi"
################################
iSmi = 1
fSmi = 1
pos  = 1
################################

for (n in iSmi:fSmi) {
  inFile  = paste(c("smi/chembl_can-",n,".txt"), collapse = "")
  print(inFile)
  delta_i = (n-1)*10000 # the previous number of drug; change the first value!
  
  # read all SMILES as MOL
  x.mol <- NULL
  x.mol = readMolFromSmi(inFile,type = 'mol')
  
  for (i in pos:length(x.mol)) { # length(x.mol)
    outFile = paste(c(outFolder,"/","chembl_can-",i+delta_i,".csv"), collapse = "") # output descriptors
    cat(i," ",i+delta_i," ",date()) # print number of molecule and chembl id
    if(file.exists(outFile) == FALSE){ # only if the CSV file is not present, run the calculation!
      # calculate the descriptors and write the output into a file with name columns, tab-separated
      print(try(write.csv(suppressWarnings(extractDrugAIOwithParams(x.mol[i],descNames,silent = FALSE, warn = FALSE)),file=outFile,quote=F),FALSE))
      # Descr = suppressWarnings(extractDrugAIO(x.mol[i]))
      # print(extractDrugAIO(x.mol[i],silent = FALSE, warn = FALSE))
    } else {
      print(paste(c(outFile,"already exists! Skipped!"),collapse = ""))
    }
  }
  cat("\n")
}

# calculate descriptors
# des =extractDrugAIOwithParams(x.mol,descNames,silent = FALSE, warn = FALSE) 

print(proc.time() - ptm) # get the time
