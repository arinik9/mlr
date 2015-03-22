library(mlr)  # http://berndbischl.github.io/mlr/tutorial/html/index.html
library(ParamHelpers)
library(ggplot2)
library(kernlab)  # for the algorithm SVM
require("clusterSim")
require(clValid)  # for the method DB index
library("parallelMap")  # for the function parallelStartMulticore()

GetNumerics = function(i, d){
  #  Computation of DB Index for each number in the range 2:8
  #  The goal is to obtain numerical values in train.data
  #  for allow to run kmeans algortihm
  #  For achieve it, we will analyze just 1st column of train.data
  #  So we will find the indices that are numericable, i.g. "2" or 2
  #
  # Args:
  #   i: index
  #   d: train.data
  #
  # Returns:
  #   numericable value of indice

  f = d[1,i]
  if (is.factor(d[1,i])){
    return(as.numeric(levels(f))[f])
  } else {
    return(as.numeric(f))
  }
}


ComputeDbIndex = function(n, task){
  #  Computation of DB Index for each number in the range 2:8
  #  The goal is to find the optimal value of n, which is number of cluster,
  #  The minimum value of DB Index  give us the optimal value of n
  #  the measure "db": Davies-Bouldin cluster separation measure
  #
  # Args:
  #   n: number of in the range 2:8
  #   task: task object from Task fontion
  #
  # Returns:
  #   The value of DB Index computed belongs to n

  parallelStartMulticore(nb.core, show.info = verbose)
  # train.data is the half size of dataset
  # So, split quotient will be (1/2)*(2/5)=1/5 of data
  rdesc = makeResampleDesc("Subsample", iters = nb.iteration, split = 2/5)
  r = resample("cluster.kmeans", task, rdesc, measures = list(db),
               centers = n, show.info = verbose)
  parallelStop()
  return(r$aggr)
}


SvmKmeansSimpleLearner <- function(dataset=NA, target=NA, C=NA, sigma=NA, 
                                   n=NA, verbose=FALSE){
  #  Simple learner that uses kernelized SVM and $K$-Means for binary (or not) 
  #  classification.
  #  ref: https://github.com/berndbischl/mlr/wiki/GSOC-2015:-Implement-several-ensemble-SVMs-in-mlr
  #
  # Args:
  #   dataset: the dataset that will be used
  #   target: the one of the column of the dataset that inform the classification label
  #   C: the regularization parameter for the SVM.
  #   sigma:  the RBF kernel parameter.
  #   n: number of cluster expected in the algorithm kmeans
  #   verbose: If TRUE, prints some outputs of the functions.
  #
  # Returns:
  #   The label the SVM predicted.


  #######################
  # PARAMETERS HANDLING #
  #######################

  if (all(is.na(dataset))){
    stop(" Arguments Dataset must not have missing values.")
  } else if (!is.na(dataset) && !is.data.frame(dataset)){
    stop(" Arguments Dataset must have a type data.frame.")
  }

  if (!is.character(target)){
    stop(" Arguments Target must be character.")
  }

  if (is.null(dataset[[target]])){
    stop(" Dataset do not contain the column Target.")
  }

  if (is.na(target)){
    #obtaining the name of last column.
    #In general, the information of class is in the last column in datasets
    target = names(dataset)[ncol(dataset)]
  }

  #################################
  # CREATION OF TEST & TRAIN DATA #
  #################################

  dataset        = na.omit(dataset)
  target.index   = which(names(dataset) %in% target)
  size.dataset   = nrow(dataset)
  random.indices = sample(size.dataset)
  middle         = round(size.dataset/2)
  train.data     = dataset[random.indices[1:middle], ]
  test.data      = dataset[random.indices[(middle+1):size.dataset], ]
  train.data$Id  = NULL
  test.data$Id   = NULL


  options(warn = -1)  # Character'lerden NA cikmasini engellemek icin
  # it does not matter wheter it is test.data or train.data in sapply
  # GetNumerics function is defined above
  indices = sapply(1:ncol(train.data), GetNumerics, train.data) 
  options(warn = 0)
  numeric.indices = which(is.na(indices) == FALSE)

  # Configuration of Train & Test Data for kmeans algorithm
  for (i in numeric.indices){
    train.data[ ,i] = as.numeric(train.data[ ,i])
    test.data[ ,i]  = as.numeric(test.data[ ,i])
  }

  ############################
  # SOME OPTIONAL PARAMETERS #
  ############################

  nb.core      = 2L  # number of core to work simultaneously in parallel works
  nb.iteration = 5L  # number of iteration in makeResampleDesc() functions

  #####################
  # KMEANS CLUSTERING #
  #####################

  train.cl.task = makeClusterTask(id = 'kmeans', data = train.data[ ,numeric.indices])

  # Optimization of the paramater "centers" of KMEANS
  # Checking the parameter that user entered
  # If it has not entered, we will do optimization for find  values optimals

  if (is.na(n)){
    min.n   = 2L
    max.n   = 8L
    n.range = c(min.n:max.n)
    # ComputeDbIndex function is defined above
    X       = sapply(n.range, ComputeDbIndex, train.cl.task)
    n       = n.range[which.min(X)]
  }

  if (verbose){
    cat("number of cluster: ", n, "\n")
  }

  kmeans.learn = setHyperPars(makeLearner('cluster.kmeans'), centers = n)
  kmeans.model = train(kmeans.learn, train.cl.task)
  kmeans.label = predict(kmeans.model, task = train.cl.task)

  ######################
  # SVM CLASSIFICATION #
  ######################

  # Checking the parameters that user entered
  # If at least one of them has not entered, we will do optimization 
  # for find  values optimals
  vals = list()
  train.svm.task = makeClassifTask(id = "classif", data = train.data, 
                                   target = target)
  ctrl = makeTuneControlGrid()
  # 5-fold cross-validation
  rdesc = makeResampleDesc("CV", iters = nb.iteration)

  # I used parallelStartMulticore function on ubuntu.
  # parallelStartSocket() function is an another choice.
  parallelStartMulticore(nb.core, show.info = verbose)
  # if just C is known
  if (!is.na(C) && is.na(sigma)){
    ps = makeParamSet(
      makeDiscreteParam("sigma", values = 2^(-2:2))
    )
    res = tuneParams("classif.ksvm", task = train.svm.task, resampling = rdesc, 
                     par.set = ps, control = ctrl, show.info = verbose)
    vals=res$x
    vals[["C"]]=C
  } else if (is.na(C) && !is.na(sigma)){
    # If just sigma is known
    ps = makeParamSet(
      makeDiscreteParam("C", values = 2^(-2:2))
    )
    res  = tuneParams("classif.ksvm", task = train.svm.task, resampling = rdesc, 
                     par.set = ps, control = ctrl, show.info = verbose)
    vals = res$x
    vals[["sigma"]] = sigma
  } else{
    ps   = makeParamSet(
      makeDiscreteParam("C", values = 2^(-2:2)),
      makeDiscreteParam("sigma", values = 2^(-2:2))
    )
    res  = tuneParams("classif.ksvm", task = train.svm.task, resampling = rdesc, 
                     par.set = ps, control = ctrl, show.info = verbose)
    vals = res$x
  }
  parallelStop()

  # C and Sigma are the values optimizated if user does not entered them
  if (verbose){
    cat("C: ", vals[["C"]], "sigma: ", vals[["sigma"]], "\n")
  }

  ####################
  # TRAINING LEARNER #
  ####################

  svm.learner = setHyperPars(makeLearner("classif.ksvm"), par.vals = vals)
  svm.trained.model = list()

  trained.cluster.id = vector()
  for (i in 1:n){
    clustered.train.data = train.data[which(kmeans.label$data$response == i), ]
    if (verbose){
      cat("clustered train data: ", clustered.train.data[ ,ncol(train.data)], "\n")
    }
    # If it exists more than one label in the cluster. Else, Svm does not support it
    # Notice that sometimes clusters only have one label. So I'm interested in the label coming from SVM
    # How can train the learner et task corresponding to cluster that has juste one label? Like "Diabet", "Diabet", "Diabet" etc ..
    # For this reason, I took juste the labels that hava more than 1 label
    if (length(unique(clustered.train.data[ ,ncol(train.data)])) > 1){
      trained.cluster.id = c(trained.cluster.id, i)
      # First, we are creating a task belongs to each clustered train data (n times)
      train.svm.task = makeClassifTask(id = paste0("classif",i), data = clustered.train.data,
                                       target = target, fixup.data = "quiet")
      svm.trained.model[[i]]=train(svm.learner, train.svm.task)
                                   #subset = which(kmeans.label$data$response == i))
    }
  }

  ##############
  # PREDICTION #
  ##############

  # I realize the 2nd clustering for test data. As Test data and Train data are separeted,
  # I will have a chance to predict the test data via the SVM model obtaining by train data

  test.cl.task = makeClusterTask(id = 'kmeans', data = test.data[ ,numeric.indices])
  kmeans.learn = setHyperPars(makeLearner('cluster.kmeans'), centers = n)
  kmeans.model = train(kmeans.learn, test.cl.task)
  kmeans.label = predict(kmeans.model, task = test.cl.task)

  svm.label = list()
  for (i in trained.cluster.id){
    data = test.data[which(kmeans.label$data$response == i), ]
    pred.class.task = makeClassifTask(data = data, target = target, fixup.data = "quiet")
    svm.label[[as.character(i)]] = predict(svm.trained.model[[i]], task = pred.class.task)
  }

  return(svm.label)
}
