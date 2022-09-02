require(biomaRt) 
geneLength <- function(geneID,group)
{
	human <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
	if (group=='ensembl'){
		fil="ensembl_gene_id"
	}else if (group=='symbol'){
		fil="hgnc_symbol"
	}
	retrive=getBM(attributes=c("hgnc_symbol","ensembl_gene_id", "start_position","end_position","transcript_length"), filters=fil, values=geneID, mart=human,useCache = FALSE) #add useCache = FALSE because of incompability of Bioconductor 
	## list the available datasets in this Mart
    #listAttributes(mart = human)
    ## we can search for a term of interest to filter this e.g. 'start'
    #searchAttributes(mart = human, pattern = "length")
	#print(retrive)
    # Take longest/mean length among all transcripts from a single gene
    gene_coords <- retrive[which(duplicated(retrive[,fil])==FALSE),]
    rownames(gene_coords) <- gene_coords[,fil]
    for (e in rownames(gene_coords)){
        gene_coords[e,'transcript_length'] <- max(retrive$transcript_length[grep(e,retrive[,fil])])
    }
	return(gene_coords)
}