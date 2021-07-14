library(dplyr)
library(recipes)
library(caret)
library(magrittr)
library(skimr)
library(psych)
library(tictoc)

#### 0. Data Dictionary ####
  # PassengerId: int
  # Survived   : factor, Survival, 0=No, 1=Yes
  # Pclass     : factor, Ticket class, 1=1st(상), 2=2nd(중), 3=3rd(하)
  # Name       : w/ 891 levels "Abbing, Mr. Anthony",..
  # Sex        : w/ 2 levels "female","male"
  # Age        : Age in years
  # SibSp      : # of siblings / spouses aboard the Titanic(탑승 형제자매/배우자수)
  # Parch      : # of parents / children aboard the Titanic(탑승 부모/자녀수, 양녀/양자는 포함했으나 유모는 포함하지 않았음)
  # Ticket     : Ticket number
  # Fare       : Passenger fare(요금)
  # Cabin      : Cabin number(객실번호)
  # Embarked   : factor, Port of Embarkation(승선한 곳), C = Cherbourg, Q = Queenstown, S = Southampton

#### 1. 데이터 읽어서 확인 ####
  rm(list=ls())
  train_tbl <- read.csv("train.csv") %>% as_tibble()
  test_tbl <- read.csv("test.csv") %>% as_tibble()
  
  train_tbl  # 891 x 12
    # Survived, Pclass : int -> factor
    # 불필요 : Name, Fare
    # Ticket과 Cabin으로 groupby 해볼 필요가 있을까? 없다면 불필요할 듯한데
    # Embarked도 필요한 변수인가?
  test_tbl   # 418 x 11

  train_tbl %>% str()
  train_tbl %>% summary()
    # Age : NA 177
    # Embarked : NA 2, C 168, Q 77, S 644
    # Sex : female 314, male 577
    # Fare : Max 512? 이상치처리가 필요한가?
  test_tbl %>% summary()
    # Age : NA 86
    # Embarked : C 102, Q 46, S 270
    # Sex : female 152, male 266
    # Fare : Max 512? 이상치처리가 필요한가? 그리고 NA 1
  
  train_tbl %>% describe()
  train_tbl %>% skim()
  
#### 2. 전처리 ####
  # train, test 결합
  f_tbl <- bind_rows(trn=train_tbl, tst=test_tbl, .id="dataset")
  f_tbl$Survived %<>% as.factor()
  #f_tbl$Survived <- factor(f_tbl$Survived, levels=c(0,1), labels=c("No", "Yes"))
  f_tbl$Pclass %<>% as.factor()
  f_tbl %>% head()
  
  # recipe
  recp <- recipe(Survived~., data=f_tbl) %>% 
    step_impute_linear(Age, impute_with = imp_vars(Age)) %>% 
    step_impute_linear(Fare, impute_with = imp_vars(Fare)) %>% 
    step_center(all_numeric(), -PassengerId) %>% 
    step_scale(all_numeric(), -PassengerId) %>% 
    step_BoxCox(all_numeric(), -PassengerId) %>% 
    step_zv(all_numeric(), -PassengerId) %>%
    #step_other(all_nominal(), -Survived, -dataset, threshold=0.1) %>% 
    step_dummy(all_nominal(), -Survived, -dataset, -Name, -Ticket, -Cabin, one_hot=TRUE) %>%
    prep()
  
  recp
  f_tbl_r <- recp %>% juice()
  f_tbl_r
  f_tbl_r %>% summary()
  
  # train, test 분리
  aft_trn <- f_tbl_r %>% filter(dataset=="trn") %>% select(-dataset)
  aft_tst <- f_tbl_r %>% filter(dataset=="tst") %>% select(-dataset, Survived)
  aft_trn  # 891 x 18
  aft_tst  # 418 x 18

#### 3. 모델링 ####
  fit_ctl <- trainControl(
                            method="repeatedcv",
                            10, 10,
                            #sampling="up",
                            #classProbs = TRUE,
                            #summaryFunction=twoClassSummary
                          )

  # RandomForest
    tic()
    set.seed(2021)
    rf_fit <- train(
                      Survived~.,
                      #data=aft_trn %>% select(-PassengerId, -Name, -Ticket, -Cabin, -Age),
                      data=aft_trn %>% select(-PassengerId, -Name, -Ticket, -Cabin),
                      method="rf",
                      #preProc=c("center", "scale"),
                      #max_depth=2,
                      tuneLength=7,
                      trControl=fit_ctl,
                      #metric="ROC"
                    )
    toc() ## 68.54 sec elapsed(10,3) -> 224.09 sec elapsed(10,10)
    
    rf_fit
      # mtry  Accuracy   Kappa    
      # 5    0.8348180  0.6410668
  
  # rpart
    tic()
    set.seed(2021)
    rpart_fit <- train(
                      Survived~.,
                      #data=aft_trn %>% select(-PassengerId, -Name, -Ticket, -Cabin, -Age),
                      data=aft_trn %>% select(-PassengerId, -Name, -Ticket, -Cabin),
                      method="rpart",
                      #preProc=c("center", "scale"),
                      #max_depth=2,
                      tuneLength=7,
                      trControl=fit_ctl,
                      #metric="ROC"
                    )
    toc() ## 2.85 sec elapsed
    
    rpart_fit
      # cp           Accuracy   Kappa    
      # 0.006578947  0.8138262  0.5917288
    rpart_fit %>% plot()
  
  #변수중요도
    rf_fit %>% varImp(scale = FALSE) %>% plot()
    rf_fit %>% varImp()
      # Fare       100.000
      # Age         96.412
      # Sex_female  89.471
      # Sex_male    83.725
      # Pclass_X3   34.821
      # SibSp       23.194
      # Parch       15.983
      # Pclass_X1   15.344
      
    rpart_fit %>% varImp()  
      # Sex_male   100.000
      # Sex_female 100.000
      # Pclass_X3   67.293
      # Fare        65.649
      # Pclass_X1   46.759
      # Age         19.533
      # SibSp       19.475
    
#### 4. 예측 ####
  finl_fit <- rf_fit
  #finl_pred <- predict(finl_fit, aft_tst, type="prob")
  finl_pred <- predict(finl_fit, aft_tst %>% select(-Name, -Ticket, -Cabin))
  finl_pred %>% summary()
    # 0 276, 1 142
  finl_pred %>% head()

#### 5. 제출 ####
  aft_tst %>% 
    select(PassengerId) %>% 
    mutate(Survived=finl_pred) %>% 
    write.csv(., "gender_submission_gdata.csv", row.names = FALSE)

#### 6. 제출 결과 ####
  # rf_fit : 31788/50276, Score 0.77033
  
  