import nltk
import string
import math
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import json
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox


def ListFetch(termm,Inv_index,filename):#this function will need terms and find you in which documents it is occurred
    listt1 = []
    if(termm in Inv_index):
        for m in filename:
            if(str(m) in Inv_index[termm]):
                listt1.append(m)
    return listt1


#fetching the stop words in Stopwords list 
f1 = open("Stopword-List.txt","r")
Stopword = f1.read()
# print(Stopword)
Stopwords = []
stop = ""
for k in range(len(Stopword)):
    if(Stopword[k]==" "):
        continue
    if(Stopword[k]=="\n"):
        Stopwords.append(stop)
        stop = ""
        continue
    stop += Stopword[k]

R_document1 = []
Inverted_Index = {
}
IDF = {
}

filenames = [1,2,3,7,8,9,11,12,13,14,15,16,17,18,21,22,23,24,25,26] #these all documents are saved in a list for easyness
N = len(filenames)


# Preprocessing of All documents 

# for k in filenames:
#     f = open("{}.txt".format(k), "r")  #as k will be increasing the all docunments will be read one by one
#     document1 = f.read()
#     strr = ""
#     curr = [k]
#     curr1 = k
#     count = 1
#     flag1 = False
#     flag2 = False
#     flag3 = False
#     flag4 = False
#     flag5 = False
#     flag6 = False

#     i = -1
#     while i < len(document1):#iterating through single character
#         i = i+1
#         if(i==len(document1)):# if counter is equal to len of douments then break and read next docunment
#             break
#         if(document1[i]=='('):# if ( will occure then flag3 will true and 
#             flag3 = True
#             continue
#         if(document1[i]==')' and flag4==True):#when closed then flag4 will be true and the word's will be appended into inverted index
#             if(document1[i+1]==" " or document1[i+1]=="." or document1[i+1]==","):
#                 i = i+1
#                 continue
#         if(document1[i]=='-' or (document1[i-1]=='-' and document1[i]=='\n')):# this condition will concanite the - words 
#             continue
#         if(document1[i]=='\n'): # when next line occur then append words by checking condition
#             flag1 = False
#             flag2 = False
#             flag3 = False
#             flag4 = False
#             flag5 = False
#             flag6 = False
#             if(strr.lower() in Stopwords):
#                 strr = ""
#                 count = count + 1
#                 continue
#             if(len(strr)<=1):
#                 str = ""
#                 count = count + 1
#                 continue
#             if(ps.stem(strr.lower()) in Inverted_Index):
#                 if(k not in Inverted_Index[ps.stem(strr.lower())]):
#                     Inverted_Index[ps.stem(strr.lower())][k] = 1
#                 else:
#                     Inverted_Index[ps.stem(strr.lower())][k] += 1
#                 df = []
#                 for m in filenames:
#                     if(m in Inverted_Index[ps.stem(strr.lower())]):
#                         df.append(m)
#                 IDF[ps.stem(strr.lower())] = math.log10(N/(len(df)))
#                 count = count + 1
#                 strr = ""
#                 continue
#             Inverted_Index[ps.stem(strr.lower())] = {k:1}
#             df = []
#             for m in filenames:
#                 if(m in Inverted_Index[ps.stem(strr.lower())]):
#                     df.append(m)
#             IDF[ps.stem(strr.lower())] = math.log10(N/(len(df)))
#             count = count + 1
#             strr = ""
#             continue
#         if(document1[i]==" " or document1[i]=="." or document1[i]==","):# when space occur then append words by checking condition
#             flag1 = False
#             flag2 = False
#             flag3 = False
#             flag4 = False
#             flag5 = False
#             flag6 = False
#             if(strr.lower() in Stopwords):
#                 strr = ""
#                 count = count + 1
#                 continue
#             if(len(strr)==1 or strr==""):
#                 strr = ""
#                 count = count + 1
#                 continue
#             if(ps.stem(strr.lower()) in Inverted_Index):
#                 if(k not in Inverted_Index[ps.stem(strr.lower())]):
#                     Inverted_Index[ps.stem(strr.lower())][k] = 1
#                 else:
#                     Inverted_Index[ps.stem(strr.lower())][k] += 1
#                 df = []
#                 for m in filenames:
#                     if(m in Inverted_Index[ps.stem(strr.lower())]):
#                         df.append(m)
#                 IDF[ps.stem(strr.lower())] = math.log10(N/(len(df)))
#                 count = count + 1
#                 strr = ""
#                 continue
#             Inverted_Index[ps.stem(strr.lower())] = {k:1}
#             df = []
#             for m in filenames:
#                 if(m in Inverted_Index[ps.stem(strr.lower())]):
#                     df.append(m)
#             IDF[ps.stem(strr.lower())] = math.log10(N/(len(df)))
#             count = count + 1
#             strr = ""
#             continue
#         if(not((document1[i]>="A" and document1[i]<="Z") or (document1[i]>="a" and document1[i]<="z"))): #if its an other character so skip
#             if(flag2==True):  # it will check now the other character are on the both sides of words then ignore it 
#                 while document1[i]!=" ":
#                     i = i+1
#                     if(i==len(document1)):
#                         break
#                 strr = ""
#                 flag1 = False
#                 flag2 = False
#                 # i = i+1
#                 continue
#             flag1 = True  # it will true when an other character will be occured
#             if(flag5==True):
#                 flag6 = True
#             continue
#         if(((document1[i]>="A" and document1[i]<="Z") or (document1[i]>="a" and document1[i]<="z")) and flag1==True):
#             flag2 = True  # it will true when and a-z character will be occured after any other character
#         if(((document1[i]>="A" and document1[i]<="Z") or (document1[i]>="a" and document1[i]<="z"))):
#             if(flag3==True):
#                 flag4 = True # for brackets
#             flag5 = True
#             if(flag6==True):  
#                 while document1[i]!=" ":
#                     i = i+1
#                     if(i==len(document1)):
#                         break
#                 strr = ""
#                 flag5 = False
#                 flag6 = False
#                 continue
#         strr+=document1[i]

# For Saving Whole Inverted index and IDF dictionary of VSM in a txt file

# with open("IvertedIndexofVSM.txt", "w") as file:
#     json.dump(Inverted_Index, file) #first I have converted the whole dictionary into JSON format then saving in txt file named "IvertedIndexofVSM"
# with open("IDF.txt", "w") as file:  
#     json.dump(IDF, file)    # Also Saved the IDF file for decreasing the time complexity of Code as I converted IDF dictionary into JSON format then wrote in txt file named "IDF.txt"

with open("IvertedIndexofVSM.txt", "r") as file:  #loading the Inverted index from txt file 
    Inverted_Index = json.load(file)
with open("IDF.txt", "r") as file1:  #loading the IDF dictionary from txt file
    IDF = json.load(file1)

# queryinput = input("Enter the Query: ")
# alpha = float(input("Enter the Value of Alpha: "))



def VSM(queryinput):   #Function For Whole logic for VSM
    DocumentsVectorSpace = [
        [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]    
    ]
    magnitude1 = 0
    Dmagnitude = 0
    TotalScoreDocument = 0
    Qitems = []
    Qitems2 = []
    Qitems1 = []
    Qitems1 = queryinput.split() #Spilting the Query terms according to space
    Qindex = {
    }
    Qindexlists = []
    QVector = []
    magnitude = 0
    Qmagnitude = 0

    filesScore = {
    }

    for q1 in Qitems1:
        if(q1 not in Qitems):
            Qitems.append(q1)
            Qitems2.append(ps.stem(q1.lower()))
        #Calculating the Term frequency of terms in Query 
        if(ps.stem(q1) in Qindex):
            Qindex[ps.stem(q1.lower())] += 1  
        else:
            Qindex[ps.stem(q1.lower())] = 1

    CopyInvertedIndex = Inverted_Index
    for q in Qitems:
        #I have maintained Qindexlists for getting the index of Query terms in Inverted Index 
        if(ps.stem(q.lower()) not in CopyInvertedIndex):
            Qindexlists.append(-1)
            # IF query term doesn't occur in inverted index then assining the index as -1
            continue
        Qindexlists.append(list(CopyInvertedIndex).index(ps.stem(q.lower()))) #saving the index number 



    counter1 = 0
    while counter1 < len(Inverted_Index):
        # Firstlt Assigning the 0 value for whole Query Vector
        QVector.append(0)
        counter1 += 1
    counter1 = 0
    while counter1 < len(Qitems): #traversing the all Query terms
        if(ps.stem(Qitems[counter1].lower()) not in Inverted_Index):
            counter1 += 1
            continue
        # Calculating the TF-IDF value for Query terms and assigning the value at inverted dictionary index number
        QVector[Qindexlists[counter1]] = Qindex[ps.stem(Qitems[counter1].lower())] * IDF[ps.stem(Qitems[counter1].lower())] 
        magnitude += (Qindex[ps.stem(Qitems[counter1].lower())] * IDF[ps.stem(Qitems[counter1].lower())])**2  #Also Calculating the euclidean length of Query Vector
        counter1 += 1
    Qmagnitude = math.sqrt(magnitude)





    counter = 0
    #Creating the Document Vector 
    for m in filenames: 
        for items in Inverted_Index: #travesing the Whole Inverted index for Document Vector
            list1 = ListFetch(items,Inverted_Index,filenames)  #Fetching the list of Documents in which the current term occur 
            if(m in list1):#checking whether the current document occur in list1 
                DocumentsVectorSpace[counter].append(IDF[items]*Inverted_Index[items][str(m)]) #calulating the tf-idf value and storing it into the vector
                # if(items in Qitems2):
                magnitude1 += (IDF[items]*Inverted_Index[items][str(m)]) ** 2  #Calculating the euclidean length 
                # print("dd")
            else:
                DocumentsVectorSpace[counter].append(0)  # append Zero if current document doesn't occur in list
                # print("ss")


        Dmagnitude = math.sqrt(magnitude1)
        if(Dmagnitude==0):  # is euclidean length of current Document vector is Zero then skiping this document 
            filesScore[m] = 0
            counter += 1
            continue
        TotalSimilarityNumerator = []
        Denometer = Qmagnitude*Dmagnitude
        magnitude1 = 0

        counter2 = 0
        #Now finally Calculating the total similarity between a document and query 
        while counter2 < len(Inverted_Index):#firstly Assigning the zero in the whole Total Similarity list
            TotalSimilarityNumerator.append(0)
            counter2 += 1
        counter2 = 0
        while counter2 < len(Qindexlists):# Now multplying the Both vector Where the Query term occur in Inverted index 
            if(Qindexlists[counter2] == -1):
                #if the Query term doesn't occur in inverted index then skip 
                counter2 += 1
                continue
            #otherwise Multiplying the both vector (Document Vector and Query Vector) and divinding it by their euclidean lengths  
            # COsine Similarity = (Document Vector * Query Vector) / (euclidean length of Document * euclidean length of Query)
            TotalSimilarityNumerator[Qindexlists[counter2]] = (QVector[Qindexlists[counter2]] * DocumentsVectorSpace[counter][Qindexlists[counter2]])/Denometer
            TotalScoreDocument += TotalSimilarityNumerator[Qindexlists[counter2]]  #finally multiplying and adding 
            counter2 += 1
        filesScore[m] = TotalScoreDocument  #appending the cosing similarity for every document and Query into a dictionary 
        TotalScoreDocument  = 0
        counter += 1

    FinalResult = []

    DesSort = sorted(filesScore.items(), key=lambda x:x[1],reverse=True)  #Sorting the final result Dictionary in decreasing order
    counter3 = 0
    while counter3 < len(DesSort):
        if(DesSort[counter3][1] >= 0.05):
            FinalResult.append(DesSort[counter3][0])  #Creating the final result for Display
        counter3 += 1
    return FinalResult



# from your_script_file_name import Simple_queryy, Proximity_Queryy
def perform_simple_query():
    query = VSM_query_entry.get()
    results = VSM(query)
    display_results(results)

def display_results(results):
    result_text.delete(1.0, tk.END)
    if results:
        for result in results:
            result_text.insert(tk.END, f"{result}\n")
    else:
        result_text.insert(tk.END, "No results found.")

# Create main window
root = tk.Tk()
root.title("Information Retrieval System")

# Create input fields and buttons
simple_query_label = tk.Label(root, text="VSM Query:")
simple_query_label.grid(row=0, column=0, padx=5, pady=5)

VSM_query_entry = tk.Entry(root, width=50)
VSM_query_entry.grid(row=0, column=1, padx=5, pady=5)

simple_query_button = tk.Button(root, text="Search", command=perform_simple_query)
simple_query_button.grid(row=0, column=2, padx=5, pady=5)

# Create text area to display results
result_text = scrolledtext.ScrolledText(root, width=80, height=20)
result_text.grid(row=2, columnspan=3, padx=5, pady=5)

# Run the main event loop
root.mainloop()
