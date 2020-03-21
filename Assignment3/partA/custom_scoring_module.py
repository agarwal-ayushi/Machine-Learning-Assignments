def custom_scoring_fuction(clf,X,y):
    clf.fit(X,y)
    return clf.oob_score_

def main():
    print("hello!")
    
if __name__== "__main__" :
    main()