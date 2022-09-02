############################
###Function to microarray data process
############################

array_process_qc <- function(GSE_ID, platform_id, savepath, qnorm) {
    
    #Download microarray data from GEO
    dl <- getGEOData(GSE_ID,qNorm=FALSE)
    platform_ids <- strsplit(platform_id,",")[[1]]
    
    #preprocess files of multiple platforms in one GSE ID
    for(i in seq(length(dl$originalData))){
        data <- list()
        if(length(dl$originalData) > 1){
            df_expr <- dl$originalData[[i]]$expr
            data$gset <- df_expr
            df_keys <- dl$originalData[[i]]$keys    
            df_platform <- dl$originalData[[i]]$platform
            data$platform <- df_platform 
            df_pheno <- dl$originalData[[i]]$pheno
            data$pheno <- df_pheno        
            platform <- dl$originalData[[i]]$platform        
        }else{
            data$gset <- dl$originalData[[GSE_ID]]$expr
            data$gprobe <- dl$originalData[[GSE_ID]]$keys 
            data$platform <- dl$originalData[[GSE_ID]]$platform
            data$pheno <- dl$originalData[[GSE_ID]]$pheno
            platform <- dl$originalData[[GSE_ID]]$platform
        }
        data_file_path <- paste0(savepath , GSE_ID, "_", platform, '_array_Exp_EachGene', '.csv')
        pheno_file_path <- paste0(savepath , GSE_ID, "_", platform, '_array_Exp_Info', '.csv')
        
        #Replace NA to 0
        data$gset[is.na(data$gset)]=0
        
        # perform quantile normalization
        if (qnorm == TRUE) {
            data$gset <- array_normalize(data$gset)
        }
           
        # Reduce rows to one probe per gene (keep probes with the highest expression sum across samples)
        data <- probe2genesymbol(data)
        
        # Save processed microarray data
        if(platform %in% platform_ids){
            write.csv(data$gset, data_file_path)
            write.csv(data$pheno, pheno_file_path)
        }

    }
  
  
}

