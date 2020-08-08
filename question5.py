import pandas as pd
import numpy as np

data = open('data_5.txt', 'r')
haha = pd.read_csv(data, sep="\t")

area_A = []
kernal_width = []

for m in haha.values:
    a = m[0]
    b = m[4]
    area_A.append(a)
    kernal_width.append(b)

# part a)
raw_dataset = pd.DataFrame(list(zip(area_A, kernal_width)))
from sklearn import preprocessing

normalized = (raw_dataset - raw_dataset.min()) / (raw_dataset.max() - raw_dataset.min())

## Standardizing

standardized = preprocessing.StandardScaler()
scaled = standardized.fit_transform(raw_dataset)
scaled = pd.DataFrame(scaled)

raw_dataset.columns = ['area_A', 'kernal_width']
scaled.columns = ['area_A', 'kernal_width']
normalized.columns = ['area_A', 'kernal_width']

print("\n Range for raw_dataset")
print(raw_dataset.max() - raw_dataset.min())

print(" \n Range for Standardized dataset : ")
print(scaled.max() - scaled.min())

print("\n Range for Normalized dataset : ")
print(normalized.max() - normalized.min())

## part b)

# i)

import matplotlib.pyplot as plt

plt.scatter(raw_dataset['area_A'], raw_dataset['kernal_width'])
plt.xlabel("area_A")
plt.ylabel("kernal_width")
plt.title("Raw dataset")
plt.show()
plt.close()

import matplotlib.pyplot as plt

plt.scatter(normalized['area_A'], normalized['kernal_width'])
plt.xlabel("area_A")
plt.ylabel("kernal_width")
plt.title("Normalized dataset")
plt.show()
plt.close()

import matplotlib.pyplot as plt

plt.scatter(scaled['area_A'], scaled['kernal_width'])
plt.xlabel("area_A")
plt.ylabel("kernal_width")
plt.title("Standardized dataset")
plt.show()
plt.close()

# part ii)

# mean

P_raw = [raw_dataset['area_A'].mean(), raw_dataset['kernal_width'].mean()]
P_normalized = [normalized['area_A'].mean(), normalized['kernal_width'].mean()]
P_standardized = [scaled['area_A'].mean(), scaled['kernal_width'].mean()]

print(" P_raw : ", P_raw, " P_normlaized : ", P_normalized, "P_standardized : ", P_standardized)

# part iii)

from scipy.spatial import distance

dist_raw = pd.DataFrame(
    columns=['point_x', 'point_y', 'euc', 'maha', 'cityblock', 'minskow', 'cheb', 'cosine', 'canberra'])
dist_std = pd.DataFrame(
    columns=['point_x', 'point_y', 'euc', 'maha', 'cityblock', 'minskow', 'cheb', 'cosine', 'canberra'])
dist_norm = pd.DataFrame(
    columns=['point_x', 'point_y', 'euc', 'maha', 'cityblock', 'minskow', 'cheb', 'cosine', 'canberra'])

for index, i in enumerate(raw_dataset.values):
    dist_raw.at[index, 'point_x'] = i[0]
    dist_raw.at[index, 'point_y'] = i[1]
    dist_raw.at[index, 'euc'] = distance.euclidean(P_raw, i)
    dist_raw.at[index, 'maha'] = distance.mahalanobis(P_raw, i, np.cov(P_raw, i))
    dist_raw.at[index, 'cityblock'] = distance.cityblock(P_raw, i)
    dist_raw.at[index, 'minskow'] = distance.minkowski(P_raw, i, 3)
    dist_raw.at[index, 'cheb'] = distance.chebyshev(P_raw, i)
    dist_raw.at[index, 'cosine'] = distance.cosine(P_raw, i)
    dist_raw.at[index, 'canberra'] = distance.canberra(P_raw, i)

for index, i in enumerate(scaled.values):
    dist_std.at[index, 'point_x'] = i[0]
    dist_std.at[index, 'point_y'] = i[1]
    dist_std.at[index, 'maha'] = distance.mahalanobis(P_standardized, i, np.cov(P_standardized, i))
    dist_std.at[index, 'euc'] = round(distance.euclidean(P_standardized, i), 3)
    dist_std.at[index, 'cityblock'] = round(distance.cityblock(P_standardized, i), 3)
    dist_std.at[index, 'minskow'] = round(distance.minkowski(P_standardized, i, 3), 3)
    dist_std.at[index, 'cheb'] = round(distance.chebyshev(P_standardized, i), 3)
    dist_std.at[index, 'cosine'] = round(distance.cosine(P_standardized, i), 3)
    dist_std.at[index, 'canberra'] = distance.canberra(P_standardized, i)

for index, i in enumerate(normalized.values):
    dist_norm.at[index, 'point_x'] = i[0]
    dist_norm.at[index, 'point_y'] = i[1]
    dist_norm.at[index, 'maha'] = distance.mahalanobis(P_normalized, i, np.cov(P_normalized, i))
    dist_norm.at[index, 'euc'] = distance.euclidean(P_normalized, i)
    dist_norm.at[index, 'cityblock'] = distance.cityblock(P_normalized, i)
    dist_norm.at[index, 'minskow'] = distance.minkowski(P_normalized, i, 3)
    dist_norm.at[index, 'cheb'] = distance.chebyshev(P_normalized, i)
    dist_norm.at[index, 'cosine'] = distance.cosine(P_normalized, i)
    dist_norm.at[index, 'canberra'] = distance.canberra(P_normalized, i)

### To print the anser for part iii), please uncomment the following

"""
for i in raw_dataset.values:
    print(i)

for j in dist_norm.values:
    print(j)

for k in dist_std.values:
    print(k)

"""
######################################################################
# part iv)

# sorting rawdataset
print("\n Raw dataset sorting")
# euclidean distance
print("\n Euclidean distance")
df_raw_euc = dist_raw.sort_values(by=['euc'])
df_raw_euc = df_raw_euc[['point_x', 'point_y', 'euc']]
one = df_raw_euc.head(10)
print(df_raw_euc.head(10))

# mahanabolis distance

print("\n Mahanabolis distance : ")
df_raw_maha = dist_raw.sort_values(by=['maha'])
df_raw_maha = df_raw_maha[['point_x', 'point_y', 'maha']]
two = df_raw_maha.head(10)
print(df_raw_maha.head(10))

# cityblock distance
print("\n City block distance : ")
df_raw_citb = dist_raw.sort_values(by=['cityblock'])
df_raw_citb = df_raw_citb[['point_x', 'point_y', 'cityblock']]
three = df_raw_citb.head(10)
print(df_raw_citb.head(10))

# Minskowki distance
print("\n Minskowki distance: ")
df_raw_mins = dist_raw.sort_values(by=['minskow'])
df_raw_mins = df_raw_mins[['point_x', 'point_y', 'minskow']]
four = df_raw_mins.head(10)
print(df_raw_mins.head(10))

# Chebishev distance

print("\n Chebishev distance: ")
df_raw_cheb = dist_raw.sort_values(by=['cheb'])
df_raw_cheb = df_raw_cheb[['point_x', 'point_y', 'cheb']]
five = df_raw_cheb.head(10)
print(df_raw_cheb.head(10))

# Cosine distance

print("\n Cosine distance: ")
df_raw_cos = dist_raw.sort_values(by=['cosine'])
df_raw_cos = df_raw_cos[['point_x', 'point_y', 'cosine']]
six = df_raw_cos.head(10)
print(df_raw_cos.head(10))

# Canberra distance
print("\n Canberra distance: ")
df_raw_can = dist_raw.sort_values(by=['canberra'])
df_raw_can = df_raw_can[['point_x', 'point_y', 'canberra']]
seven = df_raw_can.head(10)
print(df_raw_can.head(10))

################################

# Normalized dataset sorting
print("\n Normalized dataset sorting")
# euclidean distance
print("\n Euclidean distance")
df_norm_euc = dist_norm.sort_values(by=['euc'])
df_norm_euc = df_norm_euc[['point_x', 'point_y', 'euc']]
eight = df_norm_euc.head(10)
print(df_norm_euc.head(10))

# mahanabolis distance
print("\n Mahanabolis distance : ")
df_norm_maha = dist_norm.sort_values(by=['maha'])
df_norm_maha = df_norm_maha[['point_x', 'point_y', 'maha']]
nine = df_norm_maha.head(10)
print(df_norm_maha.head(10))

# cityblock distance
print("\n City block distance : ")
df_norm_citb = dist_norm.sort_values(by=['cityblock'])
df_norm_citb = df_norm_citb[['point_x', 'point_y', 'cityblock']]
ten = df_norm_citb.head(10)
print(df_norm_citb.head(10))

# Minskowki distance
print("\n Minskowki distance: ")
df_norm_mins = dist_norm.sort_values(by=['minskow'])
df_norm_mins = df_norm_mins[['point_x', 'point_y', 'minskow']]
eleven = df_norm_mins.head(10)
print(df_norm_mins.head(10))

# Chebishev distance

print("\n Chebishev distance: ")
df_norm_cheb = dist_norm.sort_values(by=['cheb'])
df_norm_cheb = df_norm_cheb[['point_x', 'point_y', 'cheb']]
twelve = df_norm_cheb.head(10)
print(df_norm_cheb.head(10))

# Cosine distance

print("\n Cosine distance: ")
df_norm_cos = dist_norm.sort_values(by=['cosine'])
df_norm_cos = df_norm_cos[['point_x', 'point_y', 'cosine']]
thirteen = df_norm_cos.head(10)
print(df_norm_cos.head(10))

# Canberra distance
print("\n Canberra distance: ")
df_norm_can = dist_norm.sort_values(by=['canberra'])
df_norm_can = df_norm_can[['point_x', 'point_y', 'canberra']]
fourteen = df_norm_can.head(10)
print(df_norm_can.head(10))

##############################################
## STandardized dataset sorting

print(" \n Standardized Dataset sorting")
# euclidean distance
print("\n Euclidean distance")
df_std_euc = dist_std.sort_values(by=['euc'])
df_std_euc = df_std_euc[['point_x', 'point_y', 'euc']]
fifteen = df_std_euc.head(10)
print(df_std_euc.head(10))

# mahanabolis distance

print("\n Mahanabolis distance : ")
df_std_maha = dist_std.sort_values(by=['maha'])
df_std_maha = df_std_maha[['point_x', 'point_y', 'maha']]
sixteen = df_std_maha.head(10)
print(df_std_maha.head(10))

# cityblock distance
print("\n City block distance : ")
df_std_citb = dist_std.sort_values(by=['cityblock'])
df_std_citb = df_std_citb[['point_x', 'point_y', 'cityblock']]
seventeen = df_std_citb.head(10)
print(df_std_citb.head(10))

# Minskowki distance
print("\n Minskowki distance: ")
df_std_mins = dist_std.sort_values(by=['minskow'])
df_std_mins = df_std_mins[['point_x', 'point_y', 'minskow']]
eighteen = df_std_mins.head(10)
print(df_std_mins.head(10))

# Chebishev distance

print("\n Chebishev distance: ")
df_std_cheb = dist_std.sort_values(by=['cheb'])
df_std_cheb = df_std_cheb[['point_x', 'point_y', 'cheb']]
nineteen = df_std_cheb.head(10)
print(df_std_cheb.head(10))

# Cosine distance

print("\n Cosine distance: ")
df_std_cos = dist_std.sort_values(by=['cosine'])
df_std_cos = df_std_cos[['point_x', 'point_y', 'cosine']]
twenty = df_std_cos.head(10)
print(df_std_cos.head(10))

# Canberra distance

print("\n Canberra distance: ")
df_std_can = dist_std.sort_values(by=['canberra'])
df_std_can = df_std_can[['point_x', 'point_y', 'canberra']]
twentyone = df_std_can.head(10)
print(df_std_can.head(10))



## 21 plots


## 1 to 7 graphs
import matplotlib.pyplot as plt

plt.scatter(df_raw_euc['point_x'], df_raw_euc['point_y'], color='purple')
plt.scatter(one['point_x'], one['point_y'], color='yellow')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='X')
for i in one.values:
    plt.plot([P_raw[0],i[0]],[P_raw[1],i[1]],'k-')
plt.title("Euclidean distance for Raw dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/one.png')
plt.close()

# -- Maha


plt.scatter(df_raw_maha['point_x'], df_raw_maha['point_y'], color='purple')
plt.scatter(two['point_x'], two['point_y'], color='yellow', label='True Position')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='X')
for i in two.values:
    plt.plot([P_raw[0],i[0]],[P_raw[1],i[1]],'k-')
plt.title("Mahanabolis distance for Raw dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/two.png')
plt.close()

plt.scatter(df_raw_citb['point_x'], df_raw_citb['point_y'], color='purple')
plt.scatter(three['point_x'], three['point_y'], color='yellow', label='True Position')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='X')
for i in three.values:
    plt.plot([P_raw[0],i[0]],[P_raw[1],i[1]],'k-')
plt.title("City Block distance for Raw dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/three.png')
plt.close()

plt.scatter(df_raw_mins['point_x'], df_raw_mins['point_y'], color='purple')
plt.scatter(four['point_x'], four['point_y'], color='yellow', label='True Position')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='X')
for i in four.values:
    plt.plot([P_raw[0],i[0]],[P_raw[1],i[1]],'k-')
plt.title("Minkowski distance for Raw dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/four.png')
plt.close()

plt.scatter(df_raw_cheb['point_x'], df_raw_cheb['point_y'], color='purple')
plt.scatter(five['point_x'], five['point_y'], color='yellow', label='True Position')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='X')
for i in five.values:
    plt.plot([P_raw[0],i[0]],[P_raw[1],i[1]],'k-')
plt.title("Chebyshev distance for Raw dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/five.png')
plt.close()

plt.scatter(df_raw_cos['point_x'], df_raw_cos['point_y'], color='purple')
plt.scatter(six['point_x'], six['point_y'], color='yellow', label='True Position')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='X')
for i in six.values:
    plt.plot([P_raw[0],i[0]],[P_raw[1],i[1]],'k-')
plt.title("Cosine distance for Raw dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/six.png')
plt.close()

plt.scatter(df_raw_can['point_x'], df_raw_can['point_y'], color='purple')
plt.scatter(seven['point_x'], seven['point_y'], color='yellow', label='True Position')
plt.scatter(P_raw[0], P_raw[1], color='red', marker='X')
for i in seven.values:
    plt.plot([P_raw[0],i[0]],[P_raw[1],i[1]],'k-')
plt.title("Canberra distance for Raw dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/seven.png')
plt.close()

###
## 8 to 14 graphs
## for normalized data

plt.scatter(df_norm_euc['point_x'], df_norm_euc['point_y'], color='purple')
plt.scatter(eight['point_x'], eight['point_y'], color='yellow')
plt.scatter(P_normalized[0], P_normalized[1], color='red', marker='X')
for i in eight.values:
    plt.plot([P_normalized[0],i[0]],[P_normalized[1],i[1]],'k-')
plt.title("Euclidean distance for Normalized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/eight.png')
plt.close()

plt.scatter(df_norm_maha['point_x'], df_norm_maha['point_y'], color='purple')
plt.scatter(nine['point_x'], nine['point_y'], color='yellow')
plt.scatter(P_normalized[0], P_normalized[1], color='red', marker='X')
for i in nine.values:
    plt.plot([P_normalized[0],i[0]],[P_normalized[1],i[1]],'k-')
plt.title("Mahanabolis distance for Normalized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/nine.png')
plt.close()

plt.scatter(df_norm_citb['point_x'], df_norm_citb['point_y'], color='purple')
plt.scatter(ten['point_x'], ten['point_y'], color='yellow')
plt.scatter(P_normalized[0], P_normalized[1], color='red', marker='X')
for i in ten.values:
    plt.plot([P_normalized[0],i[0]],[P_normalized[1],i[1]],'k-')
plt.title("City Block distance for Normalized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/ten.png')
plt.close()

plt.scatter(df_norm_mins['point_x'], df_norm_mins['point_y'], color='purple')
plt.scatter(eleven['point_x'], eleven['point_y'], color='yellow')
plt.scatter(P_normalized[0], P_normalized[1], color='red', marker='X')
for i in eleven.values:
    plt.plot([P_normalized[0],i[0]],[P_normalized[1],i[1]],'k-')
plt.title("Minkowski distance for Normalized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/eleven.png')
plt.close()

plt.scatter(df_norm_cheb['point_x'], df_norm_cheb['point_y'], color='purple')
plt.scatter(twelve['point_x'], twelve['point_y'], color='yellow')
plt.scatter(P_normalized[0], P_normalized[1], color='red', marker='X')
for i in twelve.values:
    plt.plot([P_normalized[0],i[0]],[P_normalized[1],i[1]],'k-')
plt.title("Chebyshev distance for Normalized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/twelve.png')
plt.close()

plt.scatter(df_norm_cos['point_x'], df_norm_cos['point_y'], color='purple')
plt.scatter(thirteen['point_x'], thirteen['point_y'], color='yellow')
plt.scatter(P_normalized[0], P_normalized[1], color='red', marker='X')
for i in thirteen.values:
    plt.plot([P_normalized[0],i[0]],[P_normalized[1],i[1]],'k-')
plt.title("Cosine distance for Normalized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/thirteen.png')
plt.close()

plt.scatter(df_norm_can['point_x'], df_norm_can['point_y'], color='purple')
plt.scatter(fourteen['point_x'], fourteen['point_y'], color='yellow')
plt.scatter(P_normalized[0], P_normalized[1], color='red', marker='X')
for i in fourteen.values:
    plt.plot([P_normalized[0],i[0]],[P_normalized[1],i[1]],'k-')
plt.title("Canberra distance for Normalized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/fourteen.png')
plt.close()

##
# standardized graphs
# 15 to 21 plots


plt.scatter(df_std_euc['point_x'], df_std_euc['point_y'], color='purple')
plt.scatter(fifteen['point_x'], fifteen['point_y'], color='yellow')
plt.scatter(P_standardized[0], P_standardized[1], color='red', marker='X')
for i in fifteen.values:
    plt.plot([P_standardized[0],i[0]],[P_standardized[1],i[1]],'k-')
plt.title("Euclidean distance for Standardized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/fifteen.png')
plt.close()

plt.scatter(df_std_maha['point_x'], df_std_maha['point_y'], color='purple')
plt.scatter(sixteen['point_x'], sixteen['point_y'], color='yellow')
plt.scatter(P_standardized[0], P_standardized[1], color='red', marker='X')
for i in sixteen.values:
    plt.plot([P_standardized[0],i[0]],[P_standardized[1],i[1]],'k-')
plt.title("Mahanabolis distance for Standardized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/sixteen.png')
plt.close()

plt.scatter(df_std_citb['point_x'], df_std_citb['point_y'], color='purple')
plt.scatter(seventeen['point_x'], seventeen['point_y'], color='yellow')
plt.scatter(P_standardized[0], P_standardized[1], color='red', marker='X')
for i in seventeen.values:
    plt.plot([P_standardized[0],i[0]],[P_standardized[1],i[1]],'k-')
plt.title("CityBloack distance for Standardized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/seventeen.png')
plt.close()

plt.scatter(df_std_mins['point_x'], df_std_mins['point_y'], color='purple')
plt.scatter(eighteen['point_x'], eighteen['point_y'], color='yellow')
plt.scatter(P_standardized[0], P_standardized[1], color='red', marker='X')
for i in eighteen.values:
    plt.plot([P_standardized[0],i[0]],[P_standardized[1],i[1]],'k-')
plt.title("Minkowski distance for Standardized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/eighteen.png')
plt.close()

plt.scatter(df_std_cheb['point_x'], df_std_cheb['point_y'], color='purple')
plt.scatter(nineteen['point_x'], nineteen['point_y'], color='yellow')
plt.scatter(P_standardized[0], P_standardized[1], color='red', marker='X')
for i in nineteen.values:
    plt.plot([P_standardized[0],i[0]],[P_standardized[1],i[1]],'k-')
plt.title("Chebyshev distance for Standardized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/nineteen.png')
plt.close()

plt.scatter(df_std_cos['point_x'], df_std_cos['point_y'], color='purple')
plt.scatter(twenty['point_x'], twenty['point_y'], color='yellow')
plt.scatter(P_standardized[0], P_standardized[1], color='red', marker='X')
for i in twenty.values:
    plt.plot([P_standardized[0],i[0]],[P_standardized[1],i[1]],'k-')
plt.title("Cosine distance for Standardized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/twenty.png')
plt.close()


plt.scatter(df_std_can['point_x'], df_std_can['point_y'], color='purple')
plt.scatter(twentyone['point_x'], twentyone['point_y'], color='yellow')
plt.scatter(P_standardized[0], P_standardized[1], color='red', marker='X')

for i in twentyone.values:
    plt.plot([P_standardized[0],i[0]],[P_standardized[1],i[1]],'k-')

plt.title("Canberra distance for Standardized dataset")
plt.ylabel("kernal_width")
plt.xlabel("area_A")
plt.savefig('C:/Users/harik/PycharmProjects/randomForests/21graphs/twentyone.png')

