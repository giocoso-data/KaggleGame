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
    # 불필요할듯 : Name, Fare
    # Ticket과 Cabin으로 groupby 해볼 필요가 있을까? 없다면 불필요할 듯한데
    # Embarked도 필요한 변수인가? 승선을 어디서했는지에 따라서 생존결과가 달라질수 있나?
  
  # Embarked 연관성 확인
  chisq.test(train_tbl$Embarked, train_tbl$Survived)
    # p-value = 1.619e-06. 연관성 있음
    # 승선장소에 따라서 사고당시 있던 위치가 달라지나?? 객실배치가 달라지나?? 모르겠넹;
  
  test_tbl   # 418 x 11
  
  train_tbl %>% str()
    # Name, Ticket, Cabin : factor -> char
  train_tbl %>% summary()
    # Age : NA 177
    # Embarked : "" 2, C 168, Q 77, S 644, "" -> NA
    # Cabin : NA 687, 변수 삭제하자
    # Sex : female 314, male 577
    # Fare : Max 512? 이상치처리가 필요한가?
  test_tbl %>% summary()
    # Age : NA 86
    # Embarked : C 102, Q 46, S 270
    # Sex : female 152, male 266
    # Fare : Max 512? 이상치처리가 필요한가? 그리고 NA 1

#### 2. 전처리 ####
  # train, test 결합
  f_tbl <- bind_rows(trn=train_tbl, tst=test_tbl, .id="dataset") %>% 
    select(-Cabin, -Name)
  f_tbl
    # # A tibble: 1,309 x 11
    # dataset PassengerId Survived Pclass Sex      Age SibSp Parch Ticket            Fare Embarked
    # <chr>         <int>    <int>  <int> <fct>  <dbl> <int> <int> <fct>            <dbl> <fct>       
  
  # Survived, Pclass : int -> factor  
  f_tbl$Survived %<>% as.factor()
  #f_tbl$Survived <- factor(f_tbl$Survived, levels=c(0,1), labels=c("No", "Yes"))
  f_tbl$Pclass %<>% as.factor()
  
  # Name, Ticket, Cabin : factor -> char  
  # Name, Cabin은 이미 삭제함
  f_tbl$Ticket %<>% as.character()
  
  #f_tbl$Embarked <- ifelse(f_tbl$Embarked %>% is.na(), NA, f_tbl$Embarked)     # fct -> int로 바뀜;
  #f_tbl$Embarked %>% levels()
  
  f_tbl
  f_tbl %>% summary()
  #f_tbl %>% describe()
  f_tbl %>% skim()
  
  # recipe
  recp <- recipe(Survived~., data=f_tbl) %>% 
    step_impute_linear(Age, impute_with = imp_vars(Pclass, Sex, SibSp, Parch, Embarked)) %>% 
    step_impute_linear(Fare, impute_with = imp_vars(Pclass, Sex, SibSp, Parch, Embarked)) %>% 
    step_center(all_numeric(), -PassengerId) %>% 
    step_scale(all_numeric(), -PassengerId) %>% 
    step_BoxCox(all_numeric(), -PassengerId) %>% 
    step_zv(all_numeric(), -PassengerId) %>%
    #step_other(all_nominal(), -Survived, -dataset, threshold=0.1) %>% 
    step_dummy(all_nominal(), -Survived, -dataset, -Ticket, one_hot=TRUE) %>%
    prep()
  
  recp
  f_tbl_r <- recp %>% juice()
  f_tbl_r
  f_tbl_r %>% summary()
    # Ticket이 왜 factor로 바꼈지?
  f_tbl_r$Ticket %<>% as.character()
  
  # train, test 분리
  aft_trn <- f_tbl_r %>% filter(dataset=="trn") %>% select(-dataset)
  aft_tst <- f_tbl_r %>% filter(dataset=="tst") %>% select(-dataset, Survived)
  aft_trn  # 891 x 16
  aft_tst  # 418 x 16

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
                    data=aft_trn %>% select(-PassengerId, -Ticket),
                    method="rf",
                    #preProc=c("center", "scale"),
                    #max_depth=2,
                    tuneLength=7,
                    trControl=fit_ctl,
                    #metric="ROC"
                  )
  toc() ## 228.67 sec elapsed(10,10)
  
  rf_fit
  # mtry  Accuracy   Kappa    
  # 5    0.8334784  0.6386876
  
  # rpart
  tic()
  set.seed(2021)
  rpart_fit <- train(
                      Survived~.,
                      #data=aft_trn %>% select(-PassengerId, -Ticket, -Age),
                      data=aft_trn %>% select(-PassengerId, -Ticket),
                      method="rpart",
                      #preProc=c("center", "scale"),
                      #max_depth=2,
                      tuneLength=7,
                      trControl=fit_ctl,
                      #metric="ROC"
                    )
  toc() ## 2.63 sec elapsed(10,10)
  
  rpart_fit
  # cp           Accuracy   Kappa    
  # 0.007309942  0.8113631  0.5899446
  rpart_fit %>% plot()
  
  #변수중요도
  rf_fit %>% varImp(scale = FALSE) %>% plot()
  rf_fit %>% varImp()
    # Fare       100.000
    # Age         99.893
    # Sex_male    87.530
    # Sex_female  80.882
    # Pclass_X3   34.235
    # SibSp       22.815
    # Parch       15.313
    # Pclass_X1   13.252
  
  rpart_fit %>% varImp(scale = FALSE)
  rpart_fit %>% varImp()
    # Sex_female 100.0000
    # Sex_male   100.0000
    # Fare        68.5143
    # Pclass_X3   66.1096
    # Pclass_X1   50.5358
    # Age         44.7478
    # SibSp       17.5582
    # Parch       10.0720

#### 4. 예측 ####
  finl_fit <- rf_fit
  #finl_pred <- predict(finl_fit, aft_tst, type="prob")
  finl_pred <- predict(finl_fit, aft_tst %>% select(-PassengerId, -Ticket))
  finl_pred %>% summary()
  # 0 278, 1 140
  finl_pred %>% head()

#### 5. 제출 ####
  aft_tst %>% 
    select(PassengerId) %>% 
    mutate(Survived=finl_pred) %>% 
    write.csv(., "gender_submission_gdata.csv", row.names = FALSE)

#### 6. 제출 결과 ####
# Accuracy가 더 낮아서 제출안함

