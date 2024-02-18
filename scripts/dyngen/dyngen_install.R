depp <- c("dyngen", "anndata", "dplyr", "reticulate")

depp.new<-depp[!(depp%in%installed.packages())]
if (length(depp.new)) {
  install.packages(depp.new, repos='http://cran.us.r-project.org', Ncpus = 8)
}

reticulate::install_miniconda()
anndata::install_anndata()
