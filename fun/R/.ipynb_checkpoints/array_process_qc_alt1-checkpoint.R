############################
###Function to microarray data process
############################

array_process_qc_alt1 <- function(GSE_ID, stage, savepath, qnorm, platform='') {#if > 1 platforms in GSE, need to specify here
    
   
    #Download data using MetaIntegrator package
    dl <- getGEOData(GSE_ID,qNorm=FALSE)
    if (platform !=''){
        GSE_ID=paste0(GSE_ID,'_',platform)
    }
    data <- list()
    data$gset <- dl$originalData[[GSE_ID]]$expr
    data$gprobe <- dl$originalData[[GSE_ID]]$keys 
    data$platform <- dl$originalData[[GSE_ID]]$platform
    data$pheno <- dl$originalData[[GSE_ID]]$pheno
    
    
    #Replace NA to 0
    data$gset[is.na(data$gset)]=0

    # perform quantile normalization
    #btitle <- paste(GSE_ID, '/', data$platform, " samples", sep ='')
    #inds <- order(data$pheno[,stage])
    #labels <- unique(data$pheno[inds,stage])
    #fl <- as.factor(data$pheno[inds,stage])
    if (qnorm == TRUE) {
        #Draw boxplot before normalization
        #draw_boxplot(data$gset[,inds],paste('Prior to normalization :: ',btitle, sep =''),labels,fl)
        data$gset <- array_normalize(data$gset)
        #btitle <- paste('After normalization :: ',btitle, sep ='')
    }
    #Draw boxplot to assess data quality
    #draw_boxplot(data$gset[,inds],btitle,labels,fl)
    
    # Reduce rows to one probe per gene (keep probes with the highest expression sum across samples)
    data <- probe2genesymbol(data)
    
    
    #Remap gene symbol for gene aliases
    #data$gset <-aliasMapping(data$gset)
    
    write.csv(data$gset, paste0(savepath , GSE_ID, '_array_Exp_EachGene', '.csv'))
    write.csv(data$pheno, paste0(savepath , GSE_ID, '_array_Exp_Info', '.csv'))
    return(data)
}

