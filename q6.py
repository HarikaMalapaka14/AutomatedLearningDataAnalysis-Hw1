import pandas as pd
import numpy as np
data=open('data_alda','r')
dataset=pd.read_csv(data,sep=" ")


# sol a)
yes=0
no=0
for i in dataset.values:
   sent =(i[0])
   words=sent.split("\t")
   diabeties=words[6]
   if(diabeties=='1'):
       yes=yes+1
   else:
       no=no+1


print(" The no of diabetic patients are : ", yes, " and no of non diabetic patients are : ", no)

# sol b)


glucose=0
bp=0
skinThick=0
bmi=0
diabPedFunc=0
age=0

for i in dataset.values:
   sent =(i[0])
   words=sent.split("\t")
   if(words[0]=='0'):
       glucose=glucose+1
   if(words[1]=='0'):
       bp=bp+1
   if(words[2]=='0'):
       skinThick=skinThick+1
   if(words[3]=='0'):
       bmi=bmi+1
   if(words[4]=='0'):
       diabPedFunc=diabPedFunc+1
   if(words[5]=='0'):
       age=age+1

print("The percentage of missing values in : \n")
print("Glucose (Attribute 1) : ", (glucose/768)*100)
print("Blood Preasure (Attribute 2) : ", (bp/768)*100)
print("Skin Thickness (Attribute 3) : ", (skinThick/768)*100)
print("BMI (Attribute 4) : ", (bmi/768)*100)
print("Diabetic Pedigree Function (Attribute 5) : ", (diabPedFunc/768)*100)
print("Age (Attribute 6) : ", (age/768)*100)

# part d)

for index, i in enumerate(dataset.values):
    sent=i[0]

    words=sent.split("\t")
    if(words[0]=='0'):
        dataset.drop(index, inplace=True)
        continue

    if(words[1]=='0'):
        dataset.drop(index,inplace=True)
        continue

    if(words[2]=='0'):
        dataset.drop(index,inplace=True)
        continue

    if(words[3]=='0'):
        dataset.drop(index,inplace=True)
        continue

    if(words[4]=='0'):
        dataset.drop(index,inplace=True)
        continue

    if(words[5]=='0'):
        dataset.drop(index,inplace=True)



print("The data after removing missing values : \n")

yes_new=0
no_new=0

for i in dataset.values:

   sent =(i[0])

   words=sent.split("\t")
   diabeties=words[6]
   if(diabeties=='1'):
       yes_new=yes_new+1
   else:
       no_new=no_new+1


print(" The no of diabetic patients are : ", yes_new, " and no of non diabetic patients are : ", no_new)


# part e)

glu=[]
bp2=[]
ski=[]
bmi2=[]
dpf=[]
age2=[]
for i in dataset.values:
   sent =(i[0])
   words=sent.split("\t")
   att1=float(words[0])
   glu.append(att1)
   att2=float(words[1])
   bp2.append(att2)
   att3=float(words[2])
   ski.append(att3)
   att4=float(words[3])
   bmi2.append(att4)
   att5=float(words[4])
   dpf.append(att5)
   att6=float(words[5])
   age2.append(att6)



import numpy as np
print(" \n \n ")


print(" For attribute Glucose, the mean, median, standard deviation, range, 25th percentile, 50th percentile and 75th percentile are respectively :")
print(np.mean(glu), np.median(glu), np.std(glu),np.max(glu)-np.min(glu), np.percentile(glu,25), np.percentile(glu,50), np.percentile(glu, 75))
print("\n")

print(" For attribute Blood Preasure, the mean, median, standard deviation,range, 25th percentile, 50th percentile and 75th percentile are respectively :")
print(np.mean(bp2), np.median(bp2), np.std(bp2),np.max(bp2)-np.min(bp2), np.percentile(bp2, 25), np.percentile(bp2,50), np.percentile(bp2,75))
print("\n")
print(" For attribute SkinThickness, the mean, median, standard deviation,range, 25th percentile, 50th percentile and 75th percentile are respectively :")
print(np.mean(ski), np.median(ski), np.std(ski),np.max(ski)-np.min(ski), np.percentile(ski,25), np.percentile(ski,50), np.percentile(ski,75))
print("\n")
print(" For attribute BMI, the mean, median, standard deviation,range, 25th percentile, 50th percentile and 75th percentile are respectively :")
print(np.mean(bmi2),np.median(bmi2),np.std(bmi2), np.max(bmi2)-np.min(bmi2), np.percentile(bmi2, 25), np.percentile(bmi2, 50), np.percentile(bmi2, 75))
print("\n")
print(" For attribute DiabetesPedigreeFunction, the mean, median, standard deviation,range, 25th percentile, 50th percentile and 75th percentile are respectively :")
print(np.mean(dpf), np.median(dpf), np.std(dpf),np.max(dpf)-np.min(dpf), np.percentile(dpf, 25), np.percentile(dpf, 50), np.percentile(dpf, 75))
print("\n")
print(" For attribute age, the mean, median, standard deviation,range, 25th percentile, 50th percentile and 75th percentile are respectively :")
print(np.mean(age2), np.median(age2), np.std(age2),np.max(age2)-np.min(age2), np.percentile(age2,25), np.percentile(age2, 50), np.percentile(age2, 75))



#part f)
# for Blood Pressure
import matplotlib.pyplot as plt
plt.hist(bp2, bins=10)
plt.xlabel("Blood Pressure (mm Hg)")
plt.ylabel("Frequency")
plt.title("Histogram for feature Blood Pressure")
plt.show()

# for diabetesPedigreeFunction
import matplotlib.pyplot as plt
c=['yellow','blue','green','yellow','blue','green','yellow','blue','green','yellow']
plt.hist(dpf, bins=10)
plt.xlabel("Diabetes Pedigree Function")
plt.ylabel("Frequency")
plt.title("Histogram for feature- Diabetes Pedigree FUnction")
plt.show()

#part g)
# Blood pressure QQ plot
from scipy import stats
stats.probplot(bp2, plot=plt)
plt.title("Quantile Quantile for Blood Pressure")
plt.show()

# DiabetesPedigreeFunction QQ Plot
stats.probplot(dpf, plot=plt)
plt.title("DiabeterPedigreeFunction")
plt.show()









