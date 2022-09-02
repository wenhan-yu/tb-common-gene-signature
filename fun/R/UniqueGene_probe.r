#From multiple microarray probe IDs mapped into a single gene, 
#OR
#Multiple ensemble ID mapping into a single gene symbol or replicated symbols 
#select the one with the highest expression 
#Annotation format : A table format. The probe ID or ensemble ID should be used in row names and the gene symbol should be in one of the columns  
UniqueGene_probe=function(GeneExp,Annotation,name='Symbol'){
	if(is.null(rownames(Annotation))){
		MapName <- as.vector(Annotation[rownames(GeneExp)])
	}else{
		MapName <- as.vector(Annotation[rownames(GeneExp),name])
	}
	uni <- which(duplicated(MapName)==FALSE)
	New <- GeneExp[uni,]
	rownames(New)=as.vector(MapName[uni])
	for (e in unique(MapName[which(duplicated(MapName)==TRUE)])){
		if (!is.na(e) && nchar(e)>0){ 
			ids <- which(MapName==e)
			#1. Select the probe showing the highest values summed up across the samples
			tmp <- rowSums(GeneExp[ids,]) #Sum up each probes across all samples 
			New[e,] <- GeneExp[ids[match(max(tmp),tmp)],]#Replace by the probe id with maximum values
			#2. Use the median values of all probes mapped into a gene to represent that gene expression
			#New[e,] <- colMedians(GeneExp[ids,]) #Sum up each probes across all samples 
		}
	}
	if (length(which(is.na(rownames(New)))) > 0){
		New <- New[-which(is.na(rownames(New))),]#deal with some of probe ids matching to NA
	}
	if (length(which(rownames(New)=='')) > 0){
		New <- New[-which(rownames(New)==''),]#deal with some of probe ids no mathcing to any symbol
	}
	return(New)
}
