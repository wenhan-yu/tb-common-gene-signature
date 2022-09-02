###################################################################
#### Function to convert ensembl ID to gene symbol
###################################################################

#- The function maps each ensembl ID to gene symbol with the highest gene expression sum across all samples in the matrix. 
#  Only consider the conditions one-to-one or multiple-to-one (ensembl ID to symbol) 
ensembl2genesymbol <- function(counts,ginfo) {
    MapName <- as.vector(ginfo[rownames(counts),'hgnc_symbol'])
    
    #For each duplicated gene, find the probe ID with the highest sum expression
    uni <- which(duplicated(MapName)==FALSE)
    New <- counts[uni,]
    rownames(New)=as.vector(MapName[uni])
    for (e in unique(MapName[which(duplicated(MapName)==TRUE)])){
        if (!is.na(e) && nchar(e)>0){ 
            ids <- which(MapName==e)
            #Select the ID showing the highest values summed up across the samples
            tmp <- rowSums(counts[ids,]) #Sum up each probes across all samples 
            New[e,] <- counts[ids[match(max(tmp),tmp)],]#Replace by the probe id with maximum values
        }
    }
    return(New)
}