## Hyperparameter Tuning {#tuning}

Hyperparameters are second-order parameters of machine learning models that, while often not explicitly optimized during the model estimation process, can have important impacts on the outcome and predictive performance of a model.
Typically, hyperparameters are fixed before training a model.
However, because the output of a model can be sensitive to the specification of hyperparameters, it is often recommended to make an informed decision about which hyperparameter settings may yield better model performance.
In many cases, hyperparameter settings may be chosen _a priori_, but it can be advantageous to try different settings before fitting your model on the training data.
This process is often called 'tuning' your model.

Hyperparameter tuning is supported via the extension package [mlr3tuning](https://mlr3tuning.mlr-org.com).
Below you can find an illustration of the process:


\begin{center}\includegraphics{images/tuning_process} \end{center}

At the heart of [mlr3tuning](https://mlr3tuning.mlr-org.com) are the R6 classes:

* [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html), [`TuningInstanceMultiCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceMultiCrit.html): These two classes describe the tuning problem and store the results.
* [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html): This class is the base class for implementations of tuning algorithms.

### The `TuningInstance` Classes {#tuning-optimization}

The following sub-section examines the optimization of a simple classification tree on the [`Pima Indian Diabetes`](https://mlr3.mlr-org.com/reference/mlr_tasks_pima.html) data set.


```r
task = tsk("pima")
print(task)
```

```
## <TaskClassif:pima> (768 x 9)
## * Target: diabetes
## * Properties: twoclass
## * Features (8):
##   - dbl (8): age, glucose, insulin, mass, pedigree, pregnant, pressure,
##     triceps
```

We use the classification tree from [rpart](https://cran.r-project.org/package=rpart) and choose a subset of the hyperparameters we want to tune.
This is often referred to as the "tuning space".


```r
learner = lrn("classif.rpart")
learner$param_set
```

```
## <ParamSet>
##                 id    class lower upper      levels        default value
##  1:       minsplit ParamInt     1   Inf                         20      
##  2:      minbucket ParamInt     1   Inf             <NoDefault[3]>      
##  3:             cp ParamDbl     0     1                       0.01      
##  4:     maxcompete ParamInt     0   Inf                          4      
##  5:   maxsurrogate ParamInt     0   Inf                          5      
##  6:       maxdepth ParamInt     1    30                         30      
##  7:   usesurrogate ParamInt     0     2                          2      
##  8: surrogatestyle ParamInt     0     1                          0      
##  9:           xval ParamInt     0   Inf                         10     0
## 10:     keep_model ParamLgl    NA    NA  TRUE,FALSE          FALSE
```

Here, we opt to tune two parameters:

* The complexity `cp`
* The termination criterion `minsplit`

The tuning space has to be bound, therefore one has to set lower and upper bounds:


```r
library("paradox")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
tune_ps
```

```
## <ParamSet>
##          id    class lower upper levels        default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault[3]>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault[3]>
```

Next, we need to specify how to evaluate the performance.
For this, we need to choose a [`resampling strategy`](https://mlr3.mlr-org.com/reference/Resampling.html) and a [`performance measure`](https://mlr3.mlr-org.com/reference/Measure.html).


```r
hout = rsmp("holdout")
measure = msr("classif.ce")
```

Finally, one has to select the budget available, to solve this tuning instance.
This is done by selecting one of the available [`Terminators`](https://bbotk.mlr-org.com/reference/Terminator.html):

* Terminate after a given time ([`TerminatorClockTime`](https://bbotk.mlr-org.com/reference/mlr_terminators_clock_time.html))
* Terminate after a given amount of iterations ([`TerminatorEvals`](https://bbotk.mlr-org.com/reference/mlr_terminators_evals.html))
* Terminate after a specific performance is reached ([`TerminatorPerfReached`](https://bbotk.mlr-org.com/reference/mlr_terminators_perf_reached.html))
* Terminate when tuning does not improve ([`TerminatorStagnation`](https://bbotk.mlr-org.com/reference/mlr_terminators_stagnation.html))
* A combination of the above in an *ALL* or *ANY* fashion ([`TerminatorCombo`](https://bbotk.mlr-org.com/reference/mlr_terminators_combo.html))

For this short introduction, we specify a budget of 20 evaluations and then put everything together into a [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html):


```r
library("mlr3tuning")

evals20 = trm("evals", n_evals = 20)

instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  search_space = tune_ps,
  terminator = evals20
)
instance
```

```
## <TuningInstanceSingleCrit>
## * State:  Not optimized
## * Objective: <ObjectiveTuning:classif.rpart_on_pima>
## * Search Space:
## <ParamSet>
##          id    class lower upper levels        default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault[3]>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault[3]>      
## * Terminator: <TerminatorEvals>
## * Terminated: FALSE
## * Archive:
## <Archive>
## Null data.table (0 rows and 0 cols)
```

To start the tuning, we still need to select how the optimization should take place.
In other words, we need to choose the **optimization algorithm** via the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) class.

### The `Tuner` Class

The following algorithms are currently implemented in [mlr3tuning](https://mlr3tuning.mlr-org.com):

* Grid Search ([`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html))
* Random Search ([`TunerRandomSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_random_search.html)) [@bergstra2012]
* Generalized Simulated Annealing ([`TunerGenSA`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_gensa.html))
* Non-Linear Optimization ([`TunerNLoptr`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_nloptr.html))

In this example, we will use a simple grid search with a grid resolution of 5.


```r
tuner = tnr("grid_search", resolution = 5)
```

Since we have only numeric parameters, [`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html) will create an equidistant grid between the respective upper and lower bounds.
As we have two hyperparameters with a resolution of 5, the two-dimensional grid consists of $5^2 = 25$ configurations.
Each configuration serves as hyperparameter setting for the previously defined [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) and triggers a 3-fold cross validation on the task.
All configurations will be examined by the tuner (in a random order), until either all configurations are evaluated or the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) signals that the budget is exhausted.

### Triggering the Tuning {#tuning-triggering}

To start the tuning, we simply pass the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) to the `$optimize()` method of the initialized [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html).
The tuner proceeds as follows:

1. The [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) proposes at least one hyperparameter configuration (the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) may propose multiple points to improve parallelization, which can be controlled via the setting `batch_size`).
2. For each configuration, the given [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is fitted on the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) using the provided [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html).
   All evaluations are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html).
3. The [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) is queried if the budget is exhausted.
   If the budget is not exhausted, restart with 1) until it is.
4. Determine the configuration with the best observed performance.
5. Store the best configurations as result in the instance object.
   The best hyperparameter settings (`$result_learner_param_vals`) and the corresponding measured performance (`$result_y`) can be acessed from the instance.


```r
tuner$optimize(instance)
```

```
## INFO  [08:29:18.941] Starting to optimize 2 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [08:29:18.980] Evaluating 1 configuration(s) 
## INFO  [08:29:19.100] Result of batch 1: 
## INFO  [08:29:19.102]   cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.102]  0.1        5     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.103] Evaluating 1 configuration(s) 
## INFO  [08:29:19.181] Result of batch 2: 
## INFO  [08:29:19.183]       cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.183]  0.07525        1     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.185] Evaluating 1 configuration(s) 
## INFO  [08:29:19.230] Result of batch 3: 
## INFO  [08:29:19.231]       cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.231]  0.02575        5     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.233] Evaluating 1 configuration(s) 
## INFO  [08:29:19.277] Result of batch 4: 
## INFO  [08:29:19.278]      cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.278]  0.0505       10     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.280] Evaluating 1 configuration(s) 
## INFO  [08:29:19.324] Result of batch 5: 
## INFO  [08:29:19.326]   cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.326]  0.1        8     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.328] Evaluating 1 configuration(s) 
## INFO  [08:29:19.379] Result of batch 6: 
## INFO  [08:29:19.381]     cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.381]  0.001        1     0.3164 <ResampleResult[18]> 
## INFO  [08:29:19.382] Evaluating 1 configuration(s) 
## INFO  [08:29:19.427] Result of batch 7: 
## INFO  [08:29:19.429]      cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.429]  0.0505        5     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.431] Evaluating 1 configuration(s) 
## INFO  [08:29:19.476] Result of batch 8: 
## INFO  [08:29:19.478]      cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.478]  0.0505        8     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.480] Evaluating 1 configuration(s) 
## INFO  [08:29:19.530] Result of batch 9: 
## INFO  [08:29:19.531]      cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.531]  0.0505        1     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.533] Evaluating 1 configuration(s) 
## INFO  [08:29:19.578] Result of batch 10: 
## INFO  [08:29:19.580]     cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.580]  0.001        3     0.3086 <ResampleResult[18]> 
## INFO  [08:29:19.581] Evaluating 1 configuration(s) 
## INFO  [08:29:19.627] Result of batch 11: 
## INFO  [08:29:19.629]     cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.629]  0.001       10     0.2969 <ResampleResult[18]> 
## INFO  [08:29:19.631] Evaluating 1 configuration(s) 
## INFO  [08:29:19.677] Result of batch 12: 
## INFO  [08:29:19.678]       cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.678]  0.02575        3     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.680] Evaluating 1 configuration(s) 
## INFO  [08:29:19.729] Result of batch 13: 
## INFO  [08:29:19.731]   cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.731]  0.1        1     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.733] Evaluating 1 configuration(s) 
## INFO  [08:29:19.777] Result of batch 14: 
## INFO  [08:29:19.779]       cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.779]  0.07525       10     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.781] Evaluating 1 configuration(s) 
## INFO  [08:29:19.826] Result of batch 15: 
## INFO  [08:29:19.828]   cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.828]  0.1       10     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.830] Evaluating 1 configuration(s) 
## INFO  [08:29:19.880] Result of batch 16: 
## INFO  [08:29:19.882]      cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.882]  0.0505        3     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.884] Evaluating 1 configuration(s) 
## INFO  [08:29:19.929] Result of batch 17: 
## INFO  [08:29:19.931]     cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.931]  0.001        8     0.3008 <ResampleResult[18]> 
## INFO  [08:29:19.933] Evaluating 1 configuration(s) 
## INFO  [08:29:19.978] Result of batch 18: 
## INFO  [08:29:19.980]       cp minsplit classif.ce      resample_result 
## INFO  [08:29:19.980]  0.07525        3     0.2383 <ResampleResult[18]> 
## INFO  [08:29:19.981] Evaluating 1 configuration(s) 
## INFO  [08:29:20.027] Result of batch 19: 
## INFO  [08:29:20.029]       cp minsplit classif.ce      resample_result 
## INFO  [08:29:20.029]  0.07525        8     0.2383 <ResampleResult[18]> 
## INFO  [08:29:20.031] Evaluating 1 configuration(s) 
## INFO  [08:29:20.080] Result of batch 20: 
## INFO  [08:29:20.081]       cp minsplit classif.ce      resample_result 
## INFO  [08:29:20.081]  0.02575        8     0.2383 <ResampleResult[18]> 
## INFO  [08:29:20.087] Finished optimizing after 20 evaluation(s) 
## INFO  [08:29:20.088] Result: 
## INFO  [08:29:20.089]   cp minsplit learner_param_vals  x_domain classif.ce 
## INFO  [08:29:20.089]  0.1        5          <list[3]> <list[2]>     0.2383
```

```
##     cp minsplit learner_param_vals  x_domain classif.ce
## 1: 0.1        5          <list[3]> <list[2]>     0.2383
```

```r
instance$result_learner_param_vals
```

```
## $xval
## [1] 0
## 
## $cp
## [1] 0.1
## 
## $minsplit
## [1] 5
```

```r
instance$result_y
```

```
## classif.ce 
##     0.2383
```

One can investigate all resamplings which were undertaken, as they are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) and can be accessed through `$data()` method:


```r
instance$archive$data()
```

```
##          cp minsplit classif.ce      resample_result  x_domain
##  1: 0.10000        5     0.2383 <ResampleResult[18]> <list[2]>
##  2: 0.07525        1     0.2383 <ResampleResult[18]> <list[2]>
##  3: 0.02575        5     0.2383 <ResampleResult[18]> <list[2]>
##  4: 0.05050       10     0.2383 <ResampleResult[18]> <list[2]>
##  5: 0.10000        8     0.2383 <ResampleResult[18]> <list[2]>
##  6: 0.00100        1     0.3164 <ResampleResult[18]> <list[2]>
##  7: 0.05050        5     0.2383 <ResampleResult[18]> <list[2]>
##  8: 0.05050        8     0.2383 <ResampleResult[18]> <list[2]>
##  9: 0.05050        1     0.2383 <ResampleResult[18]> <list[2]>
## 10: 0.00100        3     0.3086 <ResampleResult[18]> <list[2]>
## 11: 0.00100       10     0.2969 <ResampleResult[18]> <list[2]>
## 12: 0.02575        3     0.2383 <ResampleResult[18]> <list[2]>
## 13: 0.10000        1     0.2383 <ResampleResult[18]> <list[2]>
## 14: 0.07525       10     0.2383 <ResampleResult[18]> <list[2]>
## 15: 0.10000       10     0.2383 <ResampleResult[18]> <list[2]>
## 16: 0.05050        3     0.2383 <ResampleResult[18]> <list[2]>
## 17: 0.00100        8     0.3008 <ResampleResult[18]> <list[2]>
## 18: 0.07525        3     0.2383 <ResampleResult[18]> <list[2]>
## 19: 0.07525        8     0.2383 <ResampleResult[18]> <list[2]>
## 20: 0.02575        8     0.2383 <ResampleResult[18]> <list[2]>
##               timestamp batch_nr
##  1: 2020-08-07 08:29:19        1
##  2: 2020-08-07 08:29:19        2
##  3: 2020-08-07 08:29:19        3
##  4: 2020-08-07 08:29:19        4
##  5: 2020-08-07 08:29:19        5
##  6: 2020-08-07 08:29:19        6
##  7: 2020-08-07 08:29:19        7
##  8: 2020-08-07 08:29:19        8
##  9: 2020-08-07 08:29:19        9
## 10: 2020-08-07 08:29:19       10
## 11: 2020-08-07 08:29:19       11
## 12: 2020-08-07 08:29:19       12
## 13: 2020-08-07 08:29:19       13
## 14: 2020-08-07 08:29:19       14
## 15: 2020-08-07 08:29:19       15
## 16: 2020-08-07 08:29:19       16
## 17: 2020-08-07 08:29:19       17
## 18: 2020-08-07 08:29:19       18
## 19: 2020-08-07 08:29:20       19
## 20: 2020-08-07 08:29:20       20
```

In sum, the grid search evaluated 20/25 different configurations of the grid in a random order before the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) stopped the tuning.

Now the optimized hyperparameters can take the previously created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html), set the returned hyperparameters and train it on the full dataset.


```r
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)
```

The trained model can now be used to make a prediction on external data.
Note that predicting on observations present in the `task`,  should be avoided.
The model has seen these observations already during tuning and therefore results would be statistically biased.
Hence, the resulting performance measure would be over-optimistic.
Instead, to get statistically unbiased performance estimates for the current task, [nested resampling](#nested-resamling) is required.

### Automating the Tuning {#autotuner}

The [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) wraps a learner and augments it with an automatic tuning for a given set of hyperparameters.
Because the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) itself inherits from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) base class, it can be used like any other learner.
Analogously to the previous subsection, a new classification tree learner is created.
This classification tree learner automatically tunes the parameters `cp` and `minsplit` using an inner resampling (holdout).
We create a terminator which allows 10 evaluations, and use a simple random search as tuning algorithm:


```r
library("paradox")
library("mlr3tuning")

learner = lrn("classif.rpart")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
terminator = trm("evals", n_evals = 10)
tuner = tnr("random_search")

at = AutoTuner$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = tune_ps,
  terminator = terminator,
  tuner = tuner
)
at
```

```
## <AutoTuner:classif.rpart.tuned>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

We can now use the learner like any other learner, calling the `$train()` and `$predict()` method.
This time however, we pass it to [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) to compare the tuner to a classification tree without tuning.
This way, the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) will do its resampling for tuning on the training set of the respective split of the outer resampling.
The learner then undertakes predictions using the test set of the outer resampling.
This yields unbiased performance measures, as the observations in the test set have not been used during tuning or fitting of the respective learner.
This is called [nested resampling](#nested-resampling).

To compare the tuned learner with the learner using its default, we can use [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html):


```r
grid = benchmark_grid(
  task = tsk("pima"),
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)

# avoid console output from mlr3tuning
logger = lgr::get_logger("bbotk")
logger$set_threshold("warn")

bmr = benchmark(grid)
bmr$aggregate(msrs(c("classif.ce", "time_train")))
```

```
##    nr      resample_result task_id          learner_id resampling_id iters
## 1:  1 <ResampleResult[18]>    pima classif.rpart.tuned            cv     3
## 2:  2 <ResampleResult[18]>    pima       classif.rpart            cv     3
##    classif.ce time_train
## 1:     0.2734     0.5587
## 2:     0.2630     0.0060
```

Note that we do not expect any differences compared to the non-tuned approach for multiple reasons:

* the task is too easy
* the task is rather small, and thus prone to overfitting
* the tuning budget (10 evaluations) is small
* [rpart](https://cran.r-project.org/package=rpart) does not benefit that much from tuning

