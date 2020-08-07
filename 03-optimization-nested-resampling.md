## Nested Resampling {#nested-resampling}

In order to obtain unbiased performance estimates for learners, all parts of the model building (preprocessing and model selection steps) should be included in the resampling, i.e., repeated for every pair of training/test data.
For steps that themselves require resampling like hyperparameter tuning or feature-selection (via the wrapper approach) this results in two nested resampling loops.


\begin{center}\includegraphics[width=0.98\linewidth]{images/nested_resampling} \end{center}

The graphic above illustrates nested resampling for parameter tuning with 3-fold cross-validation in the outer and 4-fold cross-validation in the inner loop.

In the outer resampling loop, we have three pairs of training/test sets.
On each of these outer training sets parameter tuning is done, thereby executing the inner resampling loop.
This way, we get one set of selected hyperparameters for each outer training set.
Then the learner is fitted on each outer training set using the corresponding selected hyperparameters.
Subsequently, we can evaluate the performance of the learner on the outer test sets.

In [mlr3](https://mlr3.mlr-org.com), you can run nested resampling for free without programming any loops by using the [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class.
This works as follows:

1. Generate a wrapped Learner via class [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) or `mlr3filters::AutoSelect` (not yet implemented).
2. Specify all required settings - see section ["Automating the Tuning"](#autotuner) for help.
3. Call function [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) with the created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html).

You can freely combine different inner and outer resampling strategies.

A common setup is prediction and performance evaluation on a fixed outer test set.
This can be achieved by passing the [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html) strategy (`rsmp("holdout")`) as the outer resampling instance to either [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html).

The inner resampling strategy could be a cross-validation one (`rsmp("cv")`) as the sizes of the outer training sets might differ.
Per default, the inner resample description is instantiated once for every outer training set.

Note that nested resampling is computationally expensive.
For this reason we use relatively small search spaces and a low number of resampling iterations in the examples shown below.
In practice, you normally have to increase both.
As this is computationally intensive you might want to have a look at the section on [Parallelization](#parallelization).

### Execution {#nested-resamp-exec}

To optimize hyperparameters or conduct feature selection in a nested resampling you need to create learners using either:

* the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class, or
* the `mlr3filters::AutoSelect` class (not yet implemented)

We use the example from section ["Automating the Tuning"](#autotuner) and pipe the resulting learner into a [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call.


```r
library("mlr3tuning")
task = tsk("iris")
learner = lrn("classif.rpart")
resampling = rsmp("holdout")
measure = msr("classif.ce")
param_set = paradox::ParamSet$new(
  params = list(paradox::ParamDbl$new("cp", lower = 0.001, upper = 0.1)))
terminator = trm("evals", n_evals = 5)
tuner = tnr("grid_search", resolution = 10)

at = AutoTuner$new(learner, resampling, measure = measure,
  param_set, terminator, tuner = tuner)
```

Now construct the [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call:


```r
resampling_outer = rsmp("cv", folds = 3)
rr = resample(task = task, learner = at, resampling = resampling_outer)
```

```
## INFO  [08:29:26.788] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [08:29:26.817] Evaluating 1 configuration(s) 
## INFO  [08:29:26.885] Result of batch 1: 
## INFO  [08:29:26.888]     cp classif.ce      resample_result 
## INFO  [08:29:26.888]  0.056     0.0303 <ResampleResult[18]> 
## INFO  [08:29:26.890] Evaluating 1 configuration(s) 
## INFO  [08:29:26.959] Result of batch 2: 
## INFO  [08:29:26.961]   cp classif.ce      resample_result 
## INFO  [08:29:26.961]  0.1     0.0303 <ResampleResult[18]> 
## INFO  [08:29:26.963] Evaluating 1 configuration(s) 
## INFO  [08:29:27.011] Result of batch 3: 
## INFO  [08:29:27.012]     cp classif.ce      resample_result 
## INFO  [08:29:27.012]  0.012     0.0303 <ResampleResult[18]> 
## INFO  [08:29:27.014] Evaluating 1 configuration(s) 
## INFO  [08:29:27.057] Result of batch 4: 
## INFO  [08:29:27.059]     cp classif.ce      resample_result 
## INFO  [08:29:27.059]  0.023     0.0303 <ResampleResult[18]> 
## INFO  [08:29:27.060] Evaluating 1 configuration(s) 
## INFO  [08:29:27.103] Result of batch 5: 
## INFO  [08:29:27.104]     cp classif.ce      resample_result 
## INFO  [08:29:27.104]  0.045     0.0303 <ResampleResult[18]> 
## INFO  [08:29:27.110] Finished optimizing after 5 evaluation(s) 
## INFO  [08:29:27.111] Result: 
## INFO  [08:29:27.112]     cp learner_param_vals  x_domain classif.ce 
## INFO  [08:29:27.112]  0.056          <list[2]> <list[1]>     0.0303 
## INFO  [08:29:27.165] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [08:29:27.174] Evaluating 1 configuration(s) 
## INFO  [08:29:27.219] Result of batch 1: 
## INFO  [08:29:27.220]     cp classif.ce      resample_result 
## INFO  [08:29:27.220]  0.045    0.09091 <ResampleResult[18]> 
## INFO  [08:29:27.222] Evaluating 1 configuration(s) 
## INFO  [08:29:27.266] Result of batch 2: 
## INFO  [08:29:27.267]     cp classif.ce      resample_result 
## INFO  [08:29:27.267]  0.078    0.09091 <ResampleResult[18]> 
## INFO  [08:29:27.269] Evaluating 1 configuration(s) 
## INFO  [08:29:27.314] Result of batch 3: 
## INFO  [08:29:27.315]     cp classif.ce      resample_result 
## INFO  [08:29:27.315]  0.023    0.09091 <ResampleResult[18]> 
## INFO  [08:29:27.317] Evaluating 1 configuration(s) 
## INFO  [08:29:27.366] Result of batch 4: 
## INFO  [08:29:27.368]     cp classif.ce      resample_result 
## INFO  [08:29:27.368]  0.056    0.09091 <ResampleResult[18]> 
## INFO  [08:29:27.370] Evaluating 1 configuration(s) 
## INFO  [08:29:27.413] Result of batch 5: 
## INFO  [08:29:27.414]     cp classif.ce      resample_result 
## INFO  [08:29:27.414]  0.089    0.09091 <ResampleResult[18]> 
## INFO  [08:29:27.420] Finished optimizing after 5 evaluation(s) 
## INFO  [08:29:27.420] Result: 
## INFO  [08:29:27.422]     cp learner_param_vals  x_domain classif.ce 
## INFO  [08:29:27.422]  0.045          <list[2]> <list[1]>    0.09091 
## INFO  [08:29:27.470] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [08:29:27.473] Evaluating 1 configuration(s) 
## INFO  [08:29:27.519] Result of batch 1: 
## INFO  [08:29:27.521]     cp classif.ce      resample_result 
## INFO  [08:29:27.521]  0.001    0.06061 <ResampleResult[18]> 
## INFO  [08:29:27.523] Evaluating 1 configuration(s) 
## INFO  [08:29:27.571] Result of batch 2: 
## INFO  [08:29:27.572]     cp classif.ce      resample_result 
## INFO  [08:29:27.572]  0.067    0.06061 <ResampleResult[18]> 
## INFO  [08:29:27.574] Evaluating 1 configuration(s) 
## INFO  [08:29:27.618] Result of batch 3: 
## INFO  [08:29:27.619]     cp classif.ce      resample_result 
## INFO  [08:29:27.619]  0.089    0.06061 <ResampleResult[18]> 
## INFO  [08:29:27.621] Evaluating 1 configuration(s) 
## INFO  [08:29:27.667] Result of batch 4: 
## INFO  [08:29:27.668]     cp classif.ce      resample_result 
## INFO  [08:29:27.668]  0.034    0.06061 <ResampleResult[18]> 
## INFO  [08:29:27.670] Evaluating 1 configuration(s) 
## INFO  [08:29:27.715] Result of batch 5: 
## INFO  [08:29:27.716]     cp classif.ce      resample_result 
## INFO  [08:29:27.716]  0.078    0.06061 <ResampleResult[18]> 
## INFO  [08:29:27.727] Finished optimizing after 5 evaluation(s) 
## INFO  [08:29:27.728] Result: 
## INFO  [08:29:27.729]     cp learner_param_vals  x_domain classif.ce 
## INFO  [08:29:27.729]  0.001          <list[2]> <list[1]>    0.06061
```

### Evaluation {#nested-resamp-eval}

With the created [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) we can now inspect the executed resampling iterations more closely.
See the section on [Resampling](#resampling) for more detailed information about [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) objects.

For example, we can query the aggregated performance result:


```r
rr$aggregate()
```

```
## classif.ce 
##       0.08
```

Check for any errors in the folds during execution (if there is not output, warnings or errors recorded, this is an empty `data.table()`:


```r
rr$errors
```

```
## Empty data.table (0 rows and 2 cols): iteration,msg
```

Or take a look at the confusion matrix of the joined predictions:


```r
rr$prediction()$confusion
```

```
##             truth
## response     setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         44         6
##   virginica       0          6        44
```
