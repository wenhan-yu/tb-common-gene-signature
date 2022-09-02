require(biomaRt) 
idsmap <- function(geneID){
    human <- useEnsembl(biomart="ensembl", dataset="hsapiens_gene_ensembl")
    retrive=getBM(attributes=c("hgnc_symbol","ensembl_gene_id"), 
                  filters='ensembl_gene_id', values=geneID, mart=human,useCache = FALSE) #add useCache = FALSE because of incompability of Bioconductor 
    #Remove repeats
    genes <- retrive[which(duplicated(retrive[,'ensembl_gene_id'])==FALSE),]
    rownames(genes) <- genes[,'ensembl_gene_id']
    return(genes)
}