library(dplyr)
library(recipes)
library(caret)
library(magrittr)
library(skimr)
library(readr)
library(stringr)
library(psych)
library(tictoc)
library(ggplot2)

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
  setwd("~/R/Kaggle/Titanic")
  #train_tbl <- read.csv("train.csv", stringAsFactors=F) %>% as_tibble()
  #test_tbl <- read.csv("test.csv", stringAsFactors=F) %>% as_tibble()
  train_tbl <- read_csv("train.csv")
  test_tbl <- read_csv("test.csv")
  colnames(train_tbl) <- train_tbl %>% colnames() %>% tolower()
  colnames(test_tbl) <- test_tbl %>% colnames() %>% tolower()
  
  # train, test 결합
  f_tbl <- bind_rows(trn=train_tbl, tst=test_tbl, .id="dataset")
  
  train_tbl  # 891 x 12
  test_tbl   # 418 x 11
  f_tbl      # 1309 x 13
    # survived, pclass : dbl -> factor
    # sex, embarked : chr -> factor
    # ticket과 cabin으로 groupby 해볼 필요가 있을까? 없다면 불필요할 듯한데
    # embarked도 필요한 변수인가? 승선을 어디서했는지에 따라서 생존결과가 달라질수 있나?
  
  # embarked 연관성 확인
  chisq.test(train_tbl$embarked, train_tbl$survived)
    # p-value = 1.619e-06. 연관성 있음
    # 승선장소에 따라서 사고당시 있던 위치가 달라지나?? 객실배치가 달라지나?? 모르겠넹;
  
  # survived, pclass : dbl -> factor
  f_tbl$survived %<>% as.factor()
  #f_tbl$survived <- factor(f_tbl$survived, levels=c(0,1), labels=c("No", "Yes"))
  f_tbl$pclass %<>% as.factor()  

  # sex, embarked : chr -> factor
  f_tbl$sex %<>% as.factor()
  f_tbl$embarked %<>% as.factor()
  
#### 2. 전처리 ####
  f_tbl
    # # A tibble: 1,309 x 13
    #    dataset passengerid survived pclass name                       sex      age sibsp parch ticket         fare cabin embarked
    #     <chr>         <dbl> <fct>    <fct>  <chr>                    <fct>  <dbl> <dbl> <dbl> <chr>            <dbl> <chr> <fct>   
    #   1 trn               1 0        3      Braund, Mr. Owen Harris   male      22     1     0 A/5 21171         7.25 NA    S       
  f_tbl %>% str()
  f_tbl %>% summary()
    # age : NA 263
    # embarked : "" 2, C 270, Q 123, S 914, "" -> NA
    # sex : female 466, male 843
    # fare : Min 0? Max 512? 이상치처리가 필요한가? NA 1(test에)
  
  # chr NA 확인 및 전체 unique 등 확인
  #f_tbl$cabin %>% is.na() %>% sum()   # 1309 중 1014가 NA, 객실번호라 불필요할듯도 하고. 삭제하자
  f_tbl %>% skim()
    # cabin              n_missing 1014
    # name, ticket unique가 1307, 929? ticket은 그렇다쳐도 이름이 같은게 있음
    
  
  #f_tbl$embarked <- ifelse(f_tbl$embarked %>% is.na(), NA, f_tbl$embarked)     # fct -> int로 바뀜; 왜????????????
  #f_tbl$embarked %>% levels()
  
  #f_tbl %>% describe()
  
  # add title
  str_replace_all(f_tbl$name, "(.*, )|(\\.\\s.*)", "")
  f_tbl %<>% mutate(title = str_replace_all(f_tbl$name, "(.*, )|(\\.\\s.*)", ""))
  f_tbl$title %>% table()
  f_tbl$title <- ifelse(f_tbl$title %in% c("Capt","Col","Dr","Jonkheer","Major","Rev","the Countess","Sir"), "etc", f_tbl$title)
  f_tbl$title <- ifelse(f_tbl$title %in% c("Don","Dona","Lady","Miss","Mlle","Mme","Ms"), "Ms", f_tbl$title)
  f_tbl$title %>% table()
  f_tbl$title %<>% as.factor()
  f_tbl %>% filter(dataset=="trn") %>% ggplot(aes(title, fill=survived)) +
                geom_bar(position='dodge')
  
  # family_cnt, family_name
  f_tbl %>% filter(title=="Master") %>% summary()
  f_tbl %>% skim()
  f_tbl %<>% mutate(family_cnt=sibsp+parch+1)
  f_tbl %<>% select(-c(sibsp, parch, cabin))
  
  # f_tbl2 <- f_tbl %>% 
  #             mutate(family_name=str_split_fixed(f_tbl$name, pattern=",", n=2) %>% as_tibble() %>% select(V1)) %>% 
  #             select(-name)
  x <- str_split_fixed(f_tbl$name, pattern=",", n=2) %>% as_tibble()
  f_tbl2 <- f_tbl %>% mutate(family_name=x$V1) %>% select(-name)
  f_tbl2$family_name %>% unique()
  f_tbl2$family_name <- ifelse(f_tbl2$family_cnt <3, "SMALL", f_tbl2$family_name)
  f_tbl2$family_name %>% as.factor()
  
  # plot
  f_tbl2 %>% filter(dataset=="trn") %>% ggplot(aes(pclass, fare)) +
    geom_boxplot()
  f_tbl2 %>% filter(fare==0 | fare>200) %>% print(n=55)
  
  f_tbl2 %>% filter(dataset=="trn") %>% ggplot(aes(age, fill=survived)) +
    geom_histogram(bins=30)
  f_tbl2 %>% filter(dataset=="trn") %>% ggplot(aes(sex, fill=survived)) +
    geom_bar(position='dodge')
  f_tbl2 %>% filter(dataset=="trn") %>% ggplot(aes(age, fill=survived)) +
    geom_histogram(bins=30) +
    facet_grid(.~sex)
  f_tbl2 %>% filter(dataset=="trn") %>% ggplot(aes(pclass, fill=survived)) +
    geom_bar(position='dodge') +
    facet_grid(.~sex)
  
  f_tbl2 %>% skim()
  
  # recipe
  recp <- recipe(survived~., data=f_tbl2) %>%
    step_impute_knn(embarked) %>% 
    step_impute_knn(age) %>% 
    step_impute_knn(fare) %>%
    step_center(all_numeric(), -passengerid) %>% 
    step_scale(all_numeric(), -passengerid) %>% 
    step_BoxCox(all_numeric(), -passengerid) %>% 
    step_zv(all_numeric(), -passengerid) %>%
    #step_other(all_nominal(), -survived, -dataset, threshold=0.1) %>% 
    step_dummy(all_nominal(), -survived, -dataset, -ticket, -family_name, one_hot=TRUE) %>%
    prep()
  
  recp
  f_tbl_r <- recp %>% juice()
  f_tbl_r
  f_tbl_r %>% skim()
    # Ticket이 왜 factor로 바꼈지?
  f_tbl_r$ticket %<>% as.character()
  
  # train, test 분리
  aft_trn <- f_tbl_r %>% filter(dataset=="trn") %>% select(-dataset)
  aft_tst <- f_tbl_r %>% filter(dataset=="tst") %>% select(-dataset, survived)
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
                    survived~.,
                    data=aft_trn %>% select(-passengerid, -ticket),
                    method="rf",
                    #preProc=c("center", "scale"),
                    #max_depth=2,
                    tuneLength=7,
                    trControl=fit_ctl,
                    #metric="ROC"
                  )
  toc() ## 231.42 sec elapsed(10,10)
  
  rf_fit
  # mtry  Accuracy   Kappa    
  # 5    0.8349354  0.6411614
  
  # rpart
  tic()
  set.seed(2021)
  rpart_fit <- train(
                      survived~.,
                      #data=aft_trn %>% select(-passengerid, -ticket, -age),
                      data=aft_trn %>% select(-passengerid),
                      method="rpart",
                      #preProc=c("center", "scale"),
                      #max_depth=2,
                      tuneLength=7,
                      trControl=fit_ctl,
                      #metric="ROC"
                    )
  toc() ## 35.33 sec elapsed(10,10)
  
  rpart_fit
  # cp           Accuracy   Kappa    
  # 0.008040936  0.8154092  0.5963309
  rpart_fit %>% plot()
  
  #변수중요도
  rf_fit %>% varImp(scale = FALSE) %>% plot()
  rf_fit %>% varImp()
    # Age        100.000
    # Fare        91.523
    # Sex_male    81.158
    # Sex_female  80.257
    # Pclass_X3   33.357
    # SibSp       22.049
    # Parch       14.234
    # Pclass_X1   13.410
  
  rpart_fit %>% varImp()  
    # Sex_female       100.000
    # Sex_male         100.000
    # Pclass_X3         67.704
    # Fare              58.996
    # Pclass_X1         46.759
    # SibSp             19.266
    # Age               17.904

#### 4. 예측 ####
  finl_fit <- rf_fit
  #finl_pred <- predict(finl_fit, aft_tst, type="prob")
  finl_pred <- predict(finl_fit, aft_tst %>% select(-ticket))
  finl_pred %>% summary()
  # 0 278, 1 140
  finl_pred %>% head()

#### 5. 제출 ####
  aft_tst %>% 
    select(passengerid) %>% 
    mutate(survived=finl_pred) %>% 
    write.csv(., "gender_submission_gdata.csv", row.names = FALSE)

#### 6. 제출 결과 ####
# rf_fit : 28101/50276, Score 0.77511

