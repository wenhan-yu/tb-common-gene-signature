############################
###Function to microarray data process
############################

array_process_qc <- function(GSE_ID, stage, savepath, qnorm, platform='') {#if > 1 platforms in GSE, need to specify here
    
    if (GSE_ID=='E-MTAB-6845'){
        data <- EMTAB6845(savepath)
    }else{
        #Download data using MetaIntegrator package
        data <- geo_process(GSE_ID,stage,platform)
    }
    #Replace NA to 0
    data$gset[is.na(data$gset)]=0

    # perform quantile normalization
    btitle <- paste(GSE_ID, '/', data$platform, " samples", sep ='')
    inds <- order(data$pheno[,stage])
    labels <- unique(data$pheno[inds,stage])
    fl <- as.factor(data$pheno[inds,stage])
    if (qnorm == TRUE) {
        #Draw boxplot before normalization
        draw_boxplot(data$gset[,inds],paste('Prior to normalization :: ',btitle, sep =''),labels,fl)
        data$gset <- array_normalize(data$gset)
        btitle <- paste('After normalization :: ',btitle, sep ='')
    }
    #Draw boxplot to assess data quality
    draw_boxplot(data$gset[,inds],btitle,labels,fl)
    
    if (GSE_ID=='E-MTAB-6845'){
        #Integrate multiple splice junctions into a single gene and retrive gene information from BioMart
        #ginfo <- geneLength(rownames(data$gset),'ensembl')
        ginfo <- idsmap(rownames(data$gset))
        geneID <- intersect(rownames(ginfo),rownames(data$gset))
        sampleID <- intersect(colnames(data$gset),rownames(data$pheno))
        data$gset <- data$gset[geneID,sampleID]
        data$gset <- ensembl2genesymbol(data$gset,ginfo)
        data$pheno <- data$pheno[sampleID,]
    }else{
        # Reduce rows to one probe per gene (keep probes with the highest expression sum across samples)
        data <- probe2genesymbol(data)
    }
    
    #Remap gene symbol for gene aliases
    #data$gset <-aliasMapping(data$gset)
    
    write.csv(data$gset, paste0(savepath , GSE_ID, '_array_Exp_EachGene', '.csv'))
    write.csv(data$pheno, paste0(savepath , GSE_ID, '_array_Exp_Info', '.csv'))
    return(data)
}

EMTAB6845 <- function (savepath){
    data<-list()
    #Array process specific to ArrayExpress dataset
    Pheno <- read.delim(paste(savepath,'E-MTAB-6845.sdrf.txt',sep=''), header = FALSE, sep = "\t")
    Pheno <- Pheno[,c(1,7,11,12,13,14,15)]
    colnames(Pheno) <- c('subject_ID','age','gender','interferon gamma release assa','preventative tuberculosis treatment','days_to_tb_diagnosis','status')
    Pheno <- Pheno[2:dim(Pheno)[1],]
    #Convert to numeric from factor and make it negative
    Pheno$days_to_tb_diagnosis <- 0-as.numeric(as.character(Pheno$days_to_tb_diagnosis))
    #Remove duplicated subject_id
    Pheno <-Pheno[which(!duplicated(Pheno$subject_ID)),]
    rownames(Pheno) <- Pheno$subject_ID
    data$pheno <- Pheno
    
    geSet <- read.delim(paste(savepath,'E-MTAB-6845_ProcessedDataMatrix_counts.csv',sep=''), header = TRUE, sep = ",")
    rownames(geSet) <- geSet$X 
    geSet <- geSet[,2:dim(geSet)[2]]
    data$gset <- log2(as.matrix(geSet)+1)
    
    return(data)
}