library(tidyverse)
library(ggplot2)
library(vroom)
library(tidymodels)
library(embed)
library(ranger)
library(discrim)
library(naivebayes)
library(kknn)
library(themis)


#Imputation
train_missing <- read.csv("C:/Users/brook/Downloads/STAT348/GhostsGhoulsandGoblins/trainWithMissingValues.csv")
train <- read.csv("C:/Users/brook/Downloads/STAT348/GhostsGhoulsandGoblins/train.csv")
test <- read.csv("C:/Users/brook/Downloads/STAT348/GhostsGhoulsandGoblins/test.csv")


my_recipe <- recipe( type ~., data=train_missing) %>%
  step_impute_bag(hair_length,impute_with= imp_vars(has_soul, color, type), trees=500) %>%
  step_impute_bag(rotting_flesh,impute_with= imp_vars(has_soul, color, type, hair_length), trees=500) %>%
  step_impute_bag(bone_length,impute_with= imp_vars(has_soul, color, type, hair_length, rotting_flesh), trees=500)

prep <- prep(my_recipe)
bake <- bake(prep, new_data=train_missing)

rmse_vec(as.numeric(train[is.na(train_missing)]), bake[is.na(train_missing)])



#SVM
my_recipe_svm <- recipe(type ~., data=train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_lencode_glm(color???, outcome=vars(type)) %>%
  step_normalize(all_numeric_predictors())


## SVM Radial

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

wf_svm <- workflow() %>%
  add_recipe(my_recipe_svm) %>%
  add_model(svmRadial)

## Fit or Tune Model 
tuning_grid_svm <- grid_regular(rbf_sigma(),
                                cost(),
                                levels = 5) ## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)

CV_results_svm <- wf_svm %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_svm,
            metrics=metric_set(accuracy))

## Find best tuning parameters
bestTune_svm <- CV_results_svm %>%
  select_best("accuracy")

final_wf_svm <- wf_svm %>%
  finalize_workflow(bestTune_svm) %>%
  fit(data=train)

## Predict
predictions_svm <- final_wf_svm %>%
  predict(test, type = "class")

predictions_svm <- predictions_svm %>%
  bind_cols(., test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x= predictions_svm, file="predictions_svm_radial.csv", delim=",")



#Neural Networks
nn_recipe <- recipe(type ~., data=train) %>%
update_role(id, new_role="id") %>%
step_mutate_at(color, fn = factor) %>%
step_dummy(color) %>%## Turn color to factor then dummy encode color
step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
set_engine("nnet") %>% #verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 50)),
                            levels=5)

folds <- vfold_cv(train, v = 5, repeats=1)

tuned_nn <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results

## Find best tuning parameters
bestTune_nn <- tuned_nn %>%
  select_best("accuracy")

final_wf_nn <- nn_wf %>%
  finalize_workflow(bestTune_nn) %>%
  fit(data=train)

## Predict
predictions_nn <- final_wf_nn %>%
  predict(test, type = "class")

predictions_nn <- predictions_nn %>%
  bind_cols(., test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x= predictions_nn, file="predictions_nn.csv", delim=",")



#Boosting
library(bonsai)
library(lightgbm)
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate9
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")


## CV tune, finalize and predict here and save results
wf_boost <- workflow() %>%
  add_recipe(my_recipe_svm) %>%
  add_model(boost_model)

## Fit or Tune Model 
tuning_grid_boost <- grid_regular(trees(),
                                  tree_depth(),
                                  learn_rate(),
                                levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train, v = 5, repeats=1)

CV_results_boost <- wf_boost %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_boost,
            metrics=metric_set(accuracy))

## Find best tuning parameters
bestTune_boost <- CV_results_boost %>%
  select_best("accuracy")

final_wf_boost <- wf_boost %>%
  finalize_workflow(bestTune_boost) %>%
  fit(data=train)

## Predict
predictions_boost <- final_wf_boost %>%
  predict(test, type = "class")

predictions_boost <- predictions_boost %>%
  bind_cols(., test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x= predictions_boost, file="predictions_boost.csv", delim=",")




#Naive Bayes Final Model 
nb_recipe <- recipe(type ~., data=train) %>%
  #update_role(id, new_role="id") %>%
  step_lencode_glm(all_nominal_predictors(), outcome=vars(type)) %>%
  step_interact(~ hair_length + bone_length) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1)  #scale to [0,1] 

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng6

nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
tuning_grid_nb <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5) ## L^2 total tuning possibilities


## Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results_nb <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_nb,
            metrics=metric_set(accuracy))

## Find best tuning parameters
bestTune_nb <- CV_results_nb %>%
  select_best("accuracy")

final_wf_nb <-
  nb_wf %>%
  finalize_workflow(bestTune_nb) %>%
  fit(data=train)

## Predict

predictions_nb <- final_wf_nb %>%
  predict(test, type = "class")

predictions_nb <- predictions_nb %>%
  bind_cols(., test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x= predictions_nb, file="predictions_nb_2.csv", delim=",")

