require(edgeR)
diff_exp_seq_analysis <- function(Name, data, def, diff_exp_data, norm_exp_data) {
    
    # Differential expression analysis with limma and voom
    # Voom is a function in the limma package that modifies RNA-Seq data for use with limma.
    
    df <- list()
    df$dsets <- c()
    #Four pairwise comparisons 
    #ATB_v_HC  ATB_v_LTBI  ATB_v_OD  ATB_v_Tret  LTBI_v_HC
    min=3 #minimual number of a group size before pairwise comparison 
    if ('ATB' %in% names(def$labels) && 'HC' %in% names(def$labels)){
        C0 <- 'HC'
        C1 <- 'ATB'
        if (length(which(data$pheno[,def$stage] %in% def$labels[[C0]]))>min && length(which(data$pheno[,def$stage] %in% def$labels[[C1]]))>min){
            df$ATB_HC <- linearModel(data, def, def$labels[[C0]], def$labels[[C1]], C0, C1, Name)
            write.csv(df$ATB_HC, paste0(diff_exp_data,'ATB_v_HC/', Name, '_seq.csv'))
            
            #Save the comparison in list and subsets in the folder
            df$dsets <- rbind(df$dsets, c(paste0(C1,'_v_',C0),Name, C1, C0, 'seq', paste0('ATB_v_HC/', Name, '_seq.csv')))
            partition_gset(data, def, norm_exp_data, def$labels[[C0]], def$labels[[C1]], Name, C0, C1)
        }
    }
    if ('ATB' %in% names(def$labels) && 'LTBI' %in% names(def$labels)){
        C0 <- 'LTBI'
        C1 <- 'ATB'
        if (length(which(data$pheno[,def$stage] %in% def$labels[[C0]]))>min && length(which(data$pheno[,def$stage] %in% def$labels[[C1]]))>min){
            df$ATB_LTBI <- linearModel(data, def, def$labels[[C0]], def$labels[[C1]], C0, C1, Name)
            write.csv(df$ATB_LTBI, paste0(diff_exp_data,'ATB_v_LTBI/', Name, '_seq.csv'))
            
            #Save the comparison in list and subsets in the folder
            df$dsets <- rbind(df$dsets, c(paste0(C1,'_v_',C0),Name, C1, C0, 'seq', paste0('ATB_v_LTBI/', Name, '_seq.csv')))
            partition_gset(data, def, norm_exp_data, def$labels[[C0]], def$labels[[C1]], Name, C0, C1)
        }
    }
    if ('ATB' %in% names(def$labels) && 'OD' %in% names(def$labels)){
        C0 <- 'OD'
        C1 <- 'ATB'
        if (length(def$labels[[C0]])>1){
            c=1
            for (e in def$labels[[C0]]){
                if (length(which(data$pheno[,def$stage] %in% e))>min && length(which(data$pheno[,def$stage] %in% def$labels[[C1]]))>min){
                    df[[paste0('ATB_OD',c)]] <- linearModel(data, def, e, def$labels[[C1]], paste0(C0,':',e), C1, Name)
                    write.csv(df[[paste0('ATB_OD',c)]], paste0(diff_exp_data,'ATB_v_OD/', Name, '_compare', c, '_seq.csv'))
                    c=c+1
                    #Save the comparison in list and subsets in the folder
                    df$dsets <- rbind(df$dsets, c(paste0(C1,'_v_',C0),Name, C1, e, 'seq', paste0('ATB_v_OD/', Name, '_seq.csv')))
                    partition_gset(data, def, norm_exp_data, e, def$labels[[C1]], Name, e, C1)
                }
            }
        }else{
            if (length(which(data$pheno[,def$stage] %in% def$labels[[C0]]))>min && length(which(data$pheno[,def$stage] %in% def$labels[[C1]]))>min){
                df$ATB_OD <- linearModel(data, def, def$labels[[C0]], def$labels[[C1]], C0, C1, Name)
                write.csv(df$ATB_OD, paste0(diff_exp_data,'ATB_v_OD/', Name, '_seq.csv'))
                
                #Save the comparison in list and subsets in the folder
                df$dsets <- rbind(df$dsets, c(paste0(C1,'_v_',C0),Name, C1, C0, 'seq', paste0('ATB_v_OD/', Name, '_seq.csv')))
                partition_gset(data, def, norm_exp_data, def$labels[[C0]], def$labels[[C1]], Name, C0, C1)
            }
        }
    }
    if ('ATB' %in% names(def$labels) && 'Tret' %in% names(def$labels)){
        C0 <- 'Tret' 
        C1 <- 'ATB'
        if (length(def$labels[[C0]])>1){
            c=0
            for (e in def$labels[[C0]]){
                if (length(which(data$pheno[,def$stage] %in% e))>min && length(which(data$pheno[,def$stage] %in% def$labels[[C1]]))>min){
                    df[[paste0('ATB_Tret',c)]] <- linearModel(data, def, e, def$labels[[C1]], paste0(C0,':',e), C1, Name)
                    write.csv(df[[paste0('ATB_Tret',c)]], paste0(diff_exp_data,'ATB_v_Tret/', Name, '_compare', c, '_seq.csv'))
                    c=c+1
                    #Save the comparison in list and subsets in the folder
                    df$dsets <- rbind(df$dsets, c(paste0(C1,'_v_',C0),Name, C1, e, 'seq', paste0('ATB_v_Tret/', Name, '_seq.csv')))
                    partition_gset(data, def, norm_exp_data, e, def$labels[[C1]], Name, e, C1)
                }
            }
        }else{
            if (length(which(data$pheno[,def$stage] %in% def$labels[[C0]]))>min && length(which(data$pheno[,def$stage] %in% def$labels[[C1]]))>min){
                df$ATB_Tret <- linearModel(data, def, def$labels[[C0]], def$labels[[C1]], C0, C1, Name)
                write.csv(df$ATB_Tret, paste0(diff_exp_data,'ATB_v_Tret/', Name, '_seq.csv'))
                
                #Save the comparison in list and subsets in the folder
                df$dsets <- rbind(df$dsets, c(paste0(C1,'_v_',C0),Name, C1, C0, 'seq', paste0('ATB_v_Tret/', Name, '_seq.csv')))
                partition_gset(data, def, norm_exp_data, def$labels[[C0]], def$labels[[C1]], Name, C0, C1)
            }
        }
    }
    if ('LTBI' %in% names(def$labels) && 'HC' %in% names(def$labels)){
        C0 <- 'HC'
        C1 <- 'LTBI'
        if (length(which(data$pheno[,def$stage] %in% def$labels[[C0]]))>min && length(which(data$pheno[,def$stage] %in% def$labels[[C1]]))>min){
            df$LTBI_HC <- linearModel(data, def, def$labels[[C0]], def$labels[[C1]], C0, C1, Name)
            write.csv(df$LTBI_HC, paste0(diff_exp_data,'LTBI_v_HC/', Name, '_seq.csv'))
            
            #Save the comparison in list and subsets in the folder
            df$dsets <- rbind(df$dsets, c(paste0(C1,'_v_',C0),Name, C1, C0, 'seq', paste0('LTBI_v_HC/', Name, '_seq.csv')))
            partition_gset(data, def, norm_exp_data, def$labels[[C0]], def$labels[[C1]], Name, C0, C1)
        }
    }
    return(df)
}
    

linearModel <- function(data, def, mat0, mat1, lab0, lab1, Name){
    #Define the groups -G0 and G1 and find associated datasets
    G0 <- which(data$pheno[,def$stage] %in% mat0)
    G1 <- which(data$pheno[,def$stage] %in% mat1)
    #print(paste0(lab0,':',mat0));print(G0)
    #print(data$pheno[colnames(data$gset)[G0],def$stage])
    #print(paste0(lab1,':',mat1));print(G1)
    #print(data$pheno[colnames(data$gset)[G1],def$stage])
    col0 <- paste0('G0_',rownames(data$pheno)[G0])
    col1 <- paste0('G1_',rownames(data$pheno)[G1])
    subset <- data$gset[,c(rownames(data$pheno)[G0],rownames(data$pheno)[G1])]
    colnames(subset) <- c(col0,col1)
    
    d0 <- DGEList(as.matrix(subset))
    d0 <- calcNormFactors(d0)
    #Calculate normalize factor before filtering out low-expressed genes
    # Filter out genes that don't have at least 1 count-per-million in at least number of samples / 2
    isexpr <- rowSums(cpm(d0) > 1) >= dim(subset)[2] / 1.5
    d <- d0[isexpr,]
    #Convert ensembl ID to gene symbol (after normalization and removing low expressed genes)
    d$counts <-ensembl2genesymbol(d$counts,data$ginfo)
    
    groups = gsub("_.*", "", colnames(subset))
    groups <- factor(groups, levels = c('G0','G1') )
    design <- model.matrix(~ groups + 0)
    colnames(design) <- c('G0','G1')
    rownames(design) <- colnames(subset)
    # Run voom
    print(paste0('Diff expression analysis in ',lab0,'-',lab1))
    voomResults <- voom(d, design,plot=T)
    
    #Run limma
    fit <- lmFit(voomResults, design)
    cont.matrix <- makeContrasts(G1-G0, levels=design)
    fit2 <- contrasts.fit(fit, cont.matrix)
    fit2 <- eBayes(fit2, 0.01)
    plotSA(fit2, main="Final model: Mean-variance trend")
    tT <- topTable(fit2, adjust="fdr", sort="none" , n=Inf)
    #volcano plot
    options(repr.plot.width=11, repr.plot.height=8)
    plot(tT$logFC,-log10(tT$adj.P.Val), main=paste0(Name,':',lab1,' vs ', lab0),xlab='logFC', ylab='-log10(adj pvalue)')
    abline(v=1,col='red')
    abline(v=-1,col='red')
    abline(h=2,col='red')
    return(tT)
}
    
partition_gset <- function(data, def, norm_exp_data, mat0, mat1, Name, gName0, gName1){
    #Convert ensembl ID to gene symbol (after normalization and removing low expressed genes)
    data$gset <-ensembl2genesymbol(data$gset,data$ginfo)
    
    G0 <- which(data$pheno[,def$stage] %in% mat0)
    G1 <- which(data$pheno[,def$stage] %in% mat1)
    write.csv(log2(data$gset[,row.names(data$pheno)[G0]]+1),paste0(norm_exp_data, Name,'_',gName0,'_seq_Exp_EachGene.csv'))
    write.csv(log2(data$gset[,row.names(data$pheno)[G1]]+1),paste0(norm_exp_data, Name,'_',gName1,'_seq_Exp_EachGene.csv'))
    return
}
    
    