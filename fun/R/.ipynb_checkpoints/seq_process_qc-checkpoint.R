############################
###Function to microarray data process
############################

seq_process_qc <- function(GSE_ID, stage, rawpath, savepath, process=1, pheno_fetch=1) {
    #pheno_fetch=0: Do not collect phenotype data
    #process=1: Return read_count 
    #process=2: Return log2(TPM)
    
    if (pheno_fetch==1){#Some of GSE dataset doesn't have pheno data
        #Download data using MetaIntegrator package
        data <- geo_process(GSE_ID,stage)
    }else{
        data <- list()
    }
    
    if (process==1){
        #Load the RNAseq raw data (RNAseq data should have associated gsm IDs as the column names )
        counts <- read.csv(file = paste0(rawpath,GSE_ID,'.salmon.genes.read_count.tximport.gsm.csv')) 
        #Add geneID to index
        rownames(counts) <- counts$gene_id 
        counts <-  counts[,2:dim(counts)[2]]
        #Convert ensemble ID to gene symbol
        ginfo <- idsmap(rownames(counts))
        ginfo <- ginfo[intersect(rownames(counts),rownames(ginfo)),]
        counts <- counts[rownames(ginfo),]#Filter out non-mapping ensembl IDs
        #Add counts back in data
        counts <- counts[,intersect(colnames(counts),rownames(data$pheno))]
        data$gset <- counts
        data$ginfo <- ginfo
        #Make sure pheno and counts have the same samples since two matrices came from two different sources
        data$pheno <- data$pheno[colnames(counts),]

        # perform edgeR TMM normalization for the purpose of storing normalized data
        d <- DGEList(as.matrix(counts))
        d <- calcNormFactors(d)
        dn <- cpm(d)
        # Filter out genes that don't have at least 1 count-per-million in at least number of samples / 2
        isexpr <- rowSums(dn > 1) >= dim(dn)[2] / 5
        dn <- dn[isexpr,]
        #Convert ensembl ID to gene symbol (after normalization and removing low expressed genes)
        dn <-ensembl2genesymbol(dn,data$ginfo)

        #Draw boxplot to assess data quality
        inds <- order(data$pheno[,stage])
        labels <- unique(data$pheno[inds,stage])
        fl <- as.factor(data$pheno[inds,stage])
        btitle <- paste0(GSE_ID, " samples")
        draw_boxplot(log2(data$gset[,inds]+1),paste0('Prior to normalization :: ',btitle),labels,fl)
        draw_boxplot(log2(dn[,inds]+1),paste0('After normalization :: ',btitle),labels,fl)
        write.csv(log2(dn+1), paste0(savepath , GSE_ID, '_seq_Exp_EachGene', '.csv')) #Store normalized data
        
    }else{#Use TPM
        #Load the RNAseq raw data (RNAseq data should have associated gsm IDs as the column names )
        counts <- read.csv(file = paste0(rawpath,GSE_ID,'.salmon.genes.TPM.tximport.gsm.csv')) 
        #Add geneID to index
        rownames(counts) <- counts$gene_id 
        counts <-  counts[,2:dim(counts)[2]]
        #Convert ensemble ID to gene symbol
        ginfo <- idsmap(rownames(counts))
        ginfo <- ginfo[intersect(rownames(counts),rownames(ginfo)),]
        counts <- counts[rownames(ginfo),]#Filter out non-mapping ensembl IDs
        if (pheno_fetch==1){
            counts <- counts[,intersect(colnames(counts),rownames(data$pheno))]
        }
        
        #Convert ensembl ID to gene symbol 
        counts <-ensembl2genesymbol(counts,ginfo)
        data$gset <- log2(counts+1)
        data$ginfo <- ginfo
        #Make sure pheno and counts have the same samples since two matrices came from two different sources
        data$pheno <- data$pheno[colnames(counts),]
        
        btitle <- paste0(GSE_ID, " samples")
        if (pheno_fetch==1){
            #Draw boxplot to assess data quality
            inds <- order(data$pheno[,stage])
            labels <- unique(data$pheno[inds,stage])
            fl <- as.factor(data$pheno[inds,stage])
            draw_boxplot(data$gset[,inds],paste0('TPM :: ',btitle),labels,fl)
        }else{
            draw_boxplot(data$gset,paste0('TPM :: ',btitle),'','')
        }

        write.csv(data$gset, paste0(savepath , GSE_ID, '_seq_Exp_EachGene', '.csv')) 
    }
    if (pheno_fetch==1){
        write.csv(data$pheno, paste0(savepath , GSE_ID, '_seq_Exp_Info', '.csv'))
    }
    return(data)
}
