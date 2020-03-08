#!/bin/sh

echo "Running for Question 1 - Text Classification"
sleep 1
cd ques1/
echo "\n Running for Raw-NB model \n"
python ques1_a_b_c.py
echo "\n Running for feature Engineering \n "
python ques1_d_e.py
echo "\n Running TFIDF \n"
python ques1-tfidf-MultiNomNB.py
python ques1-tfidf-GaussNB-final.py

sleep 2
cd ../ques2
echo "Running for Question 2 - SVM"
sleep 1
echo "\n Running SVM for Binary Classification \n"
python ques2_1abc.py
echo ("\n Running SVM for Multi-Class Classification \n")
python ques2-2abc.py
echo ("\n Running 5-fold cross validation \n")
python ques2-2d.py
echo ("\n Runnning SVM for OVR multi-class classification \n")
python ques2-3b.py
