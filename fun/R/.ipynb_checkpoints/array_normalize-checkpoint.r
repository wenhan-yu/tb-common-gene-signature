library(preprocessCore)

array_normalize <- function(expr){
    re_expr <- preprocessCore::normalize.quantiles(expr)
    colnames(re_expr)<-colnames(expr)
    rownames(re_expr)<-rownames(expr)
    return(re_expr)
}
