#!/bin/sh

echo "Running for Question 1 - Text Classification"
sleep 1
python ques1/ques1_a_b_c.py
python ques1/ques1_d_e.py
python ques1/ques1-tfidf-MultiNomNB.py
python ques1/ques1-tfidf-GaussNB-final.py

sleep 2

echo "Running for Question 2 - SVM"
sleep 1
python ques2/ques2_1abc.py
python ques2/ques2-2abc.py
python ques2/ques2-2d.py
python ques2/ques2-3b.py
