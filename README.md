# PenalizedNR

## Cox Non-proportional Hazards model with penalization
<!-- badges: start -->
<!-- badges: end -->

A penalized Newton's method for the time-varying effects model.

## Installation

You can install the released version of PenalizedNR from [Github](https://github.com/umich-biostatistics/PenalizedNR) with:

``` r
install.packages("devtools") # you need devtools to install packages from Github
devtools::install_github("umich-biostatistics/PenalizedNR")
```

You can install directly from CRAN with:

``` r
install.packages("PenalizedNR")
```

## Example

This tutorial simulates a data set to demonstrate the functions provided by FRprovideR.

```{r example, eval=FALSE}
# load the package
library(PenalizedNR)
# other imports
library(mvtnorm)
library(splines)
library(survival)
```

To simulate a data set, use the following code chunk:

```{r example.simuate.data, eval=FALSE}
# Simulate a data set
p=1
K=10
N=1000 #sample size
F=1 ###number of facility
n_f = rep(N, F) #sample size for each facility
N=sum(n_f)
gamma = rep(0, F)
range(gamma)
gamma_subject=rep(gamma,n_f)
F_pre=1:F
facility=rep(F_pre, n_f)

############generate data########################
Sigma_z1<-AR1(0.6,p)

#z= rmvnorm(N, mean=rep(0,p), sigma=Sigma_z1)
z = rnorm(N, mean = 0, 1)

z_012_rare=function(x){
  
  U=runif(1, 0.85, 0.95)
  
  x2=quantile(x,prob=U)
  x3=x
  x3[x<x2]=0
  x3[x>x2]=1
  return(x3)
}

z = z_012_rare(z)

U=runif(N, 0,1)

pre_time=rep(0, N)
for (i in 1:(N)) {
  f=function(t) {
    integrand <- function(x) {0.5*exp(gamma_subject[i] +sin(3*pi*x/4)*(x<3)*z[i])}
    
    Lambda=integrate(integrand, lower = 0, upper = t)$value
    Lambda+log(1-U[i])
  }
  r1 <- suppressWarnings(try(uniroot(f,  lower = 0, upper = 4), silent=TRUE))
  if (class(r1) == "try-error"){    
    pre_time[i]=4
  }
  else pre_time[i]=uniroot(f,  lower = 0, upper = 4)$root
}

pre_censoring=runif(N,0,3)
pre_censoring=pre_censoring*(pre_censoring<3)+3*(pre_censoring>=3)
tcens=(pre_censoring<pre_time) # censoring indicator
delta=1-tcens
time=pre_time*(delta==1)+pre_censoring*(delta==0)

delta = delta[order(time)]
facility=facility[order(time)]
z = z[order(time)]
time = time[order(time)]
```

This data is also available in the included data sets that come with the package.
To use the included data, run:

```{r, eval=FALSE}
          # raw data
          # processed data
```

Now, set relevant parameters and fit a model to the prepared data:

```{r example.fit, eval=FALSE}
stratum=rep(1, length(time))
  data_NR <- data.frame(event=delta, time=time, z, strata=stratum, stringsAsFactors=F)
  #Z.char <- paste0("X", 1:p)
  Z.char <- "z"
  fmla <- formula(paste0("Surv(time, event)~",
                         paste(c(paste0("tv(", Z.char, ")"), "strata(strata)"), collapse="+")))
lambda_all <- c(1:30)
index <- 1
for(lambda_index in 1:length(lambda_all)){
  model_all[[index]] <- surtiver(fmla, data_NR, nsplines=K, spline ="P-spline", ties="none", tau=0.5, stop="ratch",
                     method = "Newton", btr = btr_tmp, iter.max = 15, threads = 4, parallel = TRUE,
                     lambda_spline = lambda_all[lambda_index],TIC_prox = FALSE, ord = 4, degree = 3, 
                     fixedstep = FALSE,
                     penalizestop = FALSE,
                     ICLastOnly = TRUE)
  index = index + 1
}
```


