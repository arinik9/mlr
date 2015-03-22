library(mlbench)

# Dataset no:1
data(iris)
result1.1 = SvmKmeansSimpleLearner(dataset=iris, target="Species", C=NA, sigma=NA, 
                                   n=3, verbose=FALSE)
result1.2 = SvmKmeansSimpleLearner(dataset=iris, target="Species", C=NA, sigma=NA, 
                                   n=2, verbose=TRUE)
result1.3 = SvmKmeansSimpleLearner(dataset=iris, target="Species", C=0.5, sigma=1, 
                                   n=3, verbose=FALSE)

# Dataset no:2
data(BreastCancer)
result2.1 = SvmKmeansSimpleLearner(data=BreastCancer, target="Class", C=NA, sigma=NA,
                                   n=2, verbose=FALSE)
result2.2 = SvmKmeansSimpleLearner(data=BreastCancer, target="Class", C=NA, sigma=NA,
                                   n=3, verbose=TRUE)
result2.3 = SvmKmeansSimpleLearner(data=BreastCancer, target="Class", C=0.5, sigma=1,
                                   n=2, verbose=FALSE)

# Dataset no:3
data(PimaIndiansDiabetes)
result3.1 = SvmKmeansSimpleLearner(data=PimaIndiansDiabetes, target="diabetes", C=NA,
                                   sigma=NA, n=NA, verbose=FALSE)
result3.2 = SvmKmeansSimpleLearner(data=PimaIndiansDiabetes, target="diabetes", C=NA,
                                   sigma=NA, n=3, verbose=TRUE)
result3.3 = SvmKmeansSimpleLearner(data=PimaIndiansDiabetes, target="diabetes", C=0.5,
                                   sigma=2, n=2, verbose=FALSE)
