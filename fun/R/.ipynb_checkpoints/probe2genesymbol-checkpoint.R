###################################################################
#### Function to convert probe ID to gene symbol
###################################################################

#- The function maps each gene symbol to the probe with the highest gene expression sum across all samples in the matrix. 

probe2genesymbol <- function(data) {
    #get the sum for each row/probe from expression matrix
    probe_sums <- rowSums(data$gset)
    
    #Establish probe id - gene symbol
    map <- c()
    for (probe_i in names(data$gprobe)) {
        gene_i=data$gprobe[[probe_i]]
        probe_i_sum = probe_sums[[probe_i]]
        #multiple genes to a single probe ID
        if (!is.na(gene_i) & grepl(',', gene_i, fixed = TRUE)) {
            for (e in unlist(strsplit(gene_i, ',', fixed = TRUE))) {
                map <-rbind(map,c(probe_i,trimws(e),probe_i_sum))
            }
        #one gene to a single probe ID
        } else if (!is.na(gene_i)) {
            map <-rbind(map,c(probe_i,trimws(gene_i),probe_i_sum))
        }
    }
    
    #For each duplicated gene, find the probe ID with the highest sum expression
    uni <- which(duplicated(map[,2])==FALSE)
    new_map <- map[uni,1:2]
	for (e in unique(map[which(duplicated(map[,2])==TRUE),2])){
        tmp <- map[which(map[,2] == e),]
        #Select the probe showing the highest values summed up across the samples
        map_p <- tmp[match(max(tmp[,3]),tmp[,3]),1]
        #Replace by the probe id with maximum values
        new_map[which(new_map[,2]==e),1] <- map_p
    }   
	
    #Replace probe with gene
    data$gset <- data$gset[new_map[,1],]
    rownames(data$gset) <- new_map[,2]
    
    return(data)
}