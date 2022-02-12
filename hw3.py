from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.manifold import MDS
from matplotlib.pyplot import boxplot, scatter, show, figure
import pandas as pd



'''a = load_breast_cancer()
prof = MDS(n_components=2).fit_transform(a.data)
scatter(prof[a.target == 0,0], prof[a.target == 0,1], c = 'red')
scatter(prof[a.target == 1,0], prof[a.target == 1,1], c = 'blue')
show()'''

data = pd.read_csv('covid-variants.csv')
US = data.loc[data['location'] == "United States"]
num_sequences = US["num_sequences"]
perc_sequences = US["perc_sequences"]
num_sequences_total = US["num_sequences_total"]

boxplot([num_sequences,perc_sequences,num_sequences_total])
show()

'''data1 = load_breast_cancer()

projection = MDS(n_components = 2)
plot2D = projection.fit_transform(data1.data)

scatter(plot2D[:,0],plot2D[:,1], c = data1.target)


projection3d = MDS(n_components = 3)
plot3D = projection3d.fit_transform(data1.data)

fig = figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(plot3D[:,0], plot3D[:,1], plot3D[:,2], c = data1.target)
'''
'''data2 = load_digits()
projectionDigits = MDS(n_components = 2)
plot2DDigits = projectionDigits.fit_transform(data2.data)

scatter(plot2DDigits[:,0],plot2DDigits[:,1], c = data2.target)

projection3dDigits = MDS(n_components = 3)
plot3DDigits = projection3dDigits.fit_transform(data2.data)

figDigits = figure()
axDigits = figDigits.add_subplot(projection = '3d')
axDigits.scatter(plot3DDigits[:,0], plot3DDigits[:,1], plot3DDigits[:,2], c = data2.target)
show()'''
