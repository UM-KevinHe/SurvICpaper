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

Load a simple simulation data set with sample size of 2,000.

```{r example.simuate.data, eval=FALSE}
load("simulN2kOP2.RData")
```

<!-- This data is also available in the included data sets that come with the package.
To use the included data, run:
```{r, eval=FALSE}
          # raw data
          # processed data
``` -->


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


