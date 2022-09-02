############################
###Function to draw boxplot
############################
#library("RColorBrewer")
require("viridis")  
draw_boxplot<- function(gset,title,labels,fl){
    # set parameters and draw the plot
    palette(c("#dfeaf4","#dff4e4", "#AABBCC"))
    options(repr.plot.width=4+dim(gset)[[2]]/5, repr.plot.height=6)
    par(mar=c(2+round(max(nchar(colnames(gset)))/2),4.1,4.1,10.1),xpd=TRUE)
    if (length(levels(fl))>=3){
        #colors = brewer.pal(n = length(levels(fl)), name = 'Spectral')
        colors = viridis(length(levels(fl)))
    }else if (length(levels(fl))==2){
        colors = c("#F4A582","#92C5DE")
    }else{
        colors = c("#F4A582")
    }
    boxplot(gset, boxwex=0.6, notch=T, main=title, outline=FALSE, las=2, col=rep(colors,table(fl)))
    legend("topright", labels, fill=colors, bty="n")
}
