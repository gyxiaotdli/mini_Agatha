args <- commandArgs(trailingOnly=TRUE)
if(length(args)>0){
    n <- args[1]
}else{
    n <- 'pimen'
}
fs <- list.files('.')
for(f in fs){
    f1 <- gsub('HD39091',args[1],f)
    cmd <- paste('mv',f,f1)
    cat(cmd,'\n')
    system(cmd)
}
