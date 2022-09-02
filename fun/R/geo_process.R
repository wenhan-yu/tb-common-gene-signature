#use the function- getGEOData from MetaIntegrator 
geo_process <- function(GSE_ID,stage,platform='') {
    re <- list()
    
    if ('GSE31348_GSE36238' %in% GSE_ID){#for the case with multiple GSE IDs
        SA <- getGEOData(c('GSE31348','GSE36238'),qNorm=FALSE)
        re$gset <- cbind(SA$originalData$GSE31348$expr,SA$originalData$GSE36238$expr[rownames(SA$originalData$GSE31348$expr),])
        re$gprobe <- SA$originalData$GSE36238$keys
        match <- intersect(colnames(SA$originalData$GSE31348$pheno),colnames(SA$originalData$GSE36238$pheno))
        re$pheno <- rbind(SA$originalData$GSE31348$pheno[,match],SA$originalData$GSE36238$pheno[,match])
        re$platform <- data$originalData$GSE31348$platform
        re$diseaseTerms<-unique(re$pheno[,stage])
        re$comment <- data$originalData$GSE31348$exp_comment #inform taking log transformation or not
    }else{
        data <- getGEOData(GSE_ID,qNorm=FALSE)
        if (platform !=''){
            GSE_ID=paste0(GSE_ID,'_',platform)
        }
        re$gset <- data$originalData[[GSE_ID]]$expr
        re$gprobe <- data$originalData[[GSE_ID]]$keys 
        re$platform <- data$originalData[[GSE_ID]]$platform
        re$pheno <- data$originalData[[GSE_ID]]$pheno
        #print(re$pheno[1:2,])
        re$diseaseTerms<-unique(data$originalData[[GSE_ID]]$pheno[,stage])
        re$comment <- data$originalData[[GSE_ID]]$exp_comment #inform taking log transformation or not
    }
    return(re)
    
}