#Coder: Vicky

#Imports
import matplotlib.pyplot as plt

#Data
file = 'path'
data = pd.read_csv(file) #or read_excel 'For more see pandas documentation'
data.iloc[rows, columns] #positional indices returns a dataframe 'using .values will return a numpy array' "columns = -1 == last column"


#Plotting graphs with matplotlib
'''
ploting a graph with multiple y-axes and different ramge
'''
def invisble_spines(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

fig, host = plt.subplots()
fig.subplots_adjust(right=0.7)

#displaying multiple y-axes
var1 = host.twix()
var2 = host.twix()
var'n' = host.twix()

#positioning multiple y-axes
var2.spines['right'].set_position(('axes', 1.10))
var'n'.spines['left'].set_position(('axes', 1.10))

#make patch spines invisible
var2.spines['right'].set_visible(True)
var'n'.spines['left'].set_visible(True)

v1, = host.plot(x, y, 'color', linewidth='int', label='string')
v2, = var1.plot(x, y1, 'color', linewidth='int', label='string')
v'n', = var2.plot(x, y2, 'color', linewidth='int', label='string')

#different y-axes limits
host.set_limit(int, int)
var1.set_limt(int, int)
var'n'.set_limit(int, int)

#labels
host.set_xlabel('string')
host.set_ylabel('string')
var1.set_ylabel('string')
var'n'.set_ylabel('string')

#axes labels color
host.yaxis.label.set_color(v1.get_color())
var1.yaxis.label.set_color(v1.get_color())
var'n'.yaxis.label.set_color(v1.get_color())

#getting same colored axes values or tick parameters
tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=v1.get_color, **tkw)
var1.tick_params(axis='y', colors=v1.get_color, **tkw)
var'n'.tick_params(axis='y', colors=v1.get_color, **tkw)
host.tick_params(axis='x', **tkw)

#Title and plotting
plt.title('string', color='color', fontweight='medium') #see documentation for more fontweights
lines = [v1, v2, v'n']
host.legends(lines, [l.get_label() for l in lines], loc=2)

plt.show()

#removing legend and also x and y values
ax = df.plot()
plt.xlabel('Time')
plt.ylabel('Target variables')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.get_legend().remove()