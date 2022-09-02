############################
###Function to gene alias mapping
############################

aliasMapping <- function(gset){
    ori.n <-rownames(gset)
    Symbol <- alias2SymbolTable(ori.n, species="Hs")
    Symbol[is.na(Symbol)]<-ori.n[is.na(Symbol)]#Replace NA which don't alias matched back to original name
    #Remove duplicated gene symbol after replacement 
    uni <- which(duplicated(Symbol)==FALSE)
	New <- gset[uni,]
    rownames(New)<-Symbol[uni]
	for (e in unique(Symbol[which(duplicated(Symbol)==TRUE)])){
        ids <- which(Symbol==e)
        #1. Select the probe showing the highest values summed up across the samples
        tmp <- rowSums(gset[ids,]) #Sum up each probes across all samples 
        New[e,] <- gset[ids[match(max(tmp),tmp)],]#Replace by the probe id with maximum values
	}
    gset <- New
    return(gset)
}
