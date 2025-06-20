import numpy as np  
import pandas as pd  
from scipy.stats import norm  
import matplotlib.pyplot as plt  
import tensorflow as tf  
from tensorflow import keras  
import keras.backend as k  
from tensorflow.keras import layers 
from tensorflow.keras.layers.experimental import preprocessing 
#from tensorflow.keras.layers.experimental.preprocessing import preprocessing  
from sklearn import preprocessing as sk_prep  
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm, use  
from matplotlib.ticker import LinearLocator, FormatStrFormatter  
'''  
Read csv file with ASTR data  
    csv file consists of stock data of 'ASTR'  
'''  
astr = pd.read_csv('data/ASTR.csv')  
'''  
Determining sigma  
    daily_sd is standart deviation of percent changing  
    adjustment close value  
    sigma is parametr of hv calculated as daily deviation  
    multiple by sqrt of trading days in one year.  
'''  
daily_sd = astr['Close'].pct_change().std()  
sigma = daily_sd * 255 ** 0.5  
'''    


Creating dataset  
    obs is number of observations in dataset  
    data is DataFrame which stores all parameters  
    'K' is strike price  
    'S0' is stock price  
    'Sigma' is coefficient of volatility  
    't' is time values  
    'r' is free risk rate  
'''  
obs = 20000  
data = pd.DataFrame( columns = ["S0", "K", "t", "r", "Sigma"])  
data['K'] = np.full(obs, astr['Close'].mean())  
data["S0"] = np.multiply(np.random.uniform(low=0.8, high=1.2, size=obs), data["K"])  
data["Sigma"] = np.full(obs, sigma)  
data["t"] = (np.around(np.random.uniform(low=0.01, high=1.0, size=obs), decimals = 2))  
data["r"] = np.full(obs, 0.20)  
'''  
Black-Scholes formula for determining  
    analytical option price  
'''  
data["d1"] = (np.log(data["S0"]/data["K"]) + (data['r']+0.5*data["Sigma"]**2)*data["t"]) /  (data["Sigma"]*data["t"]**0.5)  
data["d2"]  = data["d1"] - data["Sigma"]*data["t"]**0.5  
data["c"] = data["S0"]*norm.cdf(data["d1"]) - data["K"]*np.exp(-data["r"]*data["t"])*norm.cdf(data["d2"])  
data.drop(["d1", "d2"],axis=1, inplace=True)  
'''  
Split data on train and test datasets  
    with 0.8 ration  
'''  
train_dataset = data.sample(frac=0.8, random_state=0)  
test_dataset = data.drop(train_dataset.index)  
'''  
Separation datasets on features and labels  
    as preparation to fit model  
'''   



train_features = train_dataset.copy()  
test_features = test_dataset.copy()  

train_labels = train_features.pop('c')  
test_labels = test_features.pop('c')  
'''  
Creating normalization layer by using  
    preprocessing module. This layer receives  
    parameters and normalize it before fed into model  
'''  
normalizer = preprocessing.Normalization()  
normalizer.adapt(np.array(train_features))  
'''  
Define Neural Netowrk model:  
    Model consists of 4 hiden layer with  
    64 neurons each, normalizer on input layer,  
    and output layer with 1 neuron.  
      
    Loss function is MSE  
    Optimizer is Adam with learning rate = 0.001  
'''  
DNN_model = keras.Sequential([  
    normalizer,  
    layers.Dense(64, activation=layers.LeakyReLU(alpha=0.1)),      layers.Dense(64, activation='elu'),  
    layers.Dense(64, activation='relu'),  
    layers.Dense(64, activation='elu'),  
    layers.Dense(1)  
])  

DNN_model.compile(loss='mean_squared_error',  
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))  
DNN_model.build(train_features.shape)  


'''  
Train model with validation split is 0.2  
    and number of epoch is 100  
'''  
history = DNN_model.fit(  
    train_features, train_labels,  
    validation_split=0.2,  
    verbose=1, epochs=100)  
'''  
Loss values history plot  
'''  
def plot_loss(history):  
  plt.plot(history.history['loss'], label='loss value', color='blue')    
  plt.xlabel('Epoch')  
  plt.ylabel('Mean Squared Error [c]')  
  plt.legend()  
  plt.title("MSE loss through the epochs", fontsize=16)  
  plt.savefig('MSE loss history.pdf', dpi=1200)  
  plt.grid(True)  

'''  
Check data:  
    test_features includes 'S0', 'K', 't', 'r', 'Sigma'  
    test_labels includes 'c' true values  
    test_prediction is the output of the model  
      
Then to get delta for true c values I used N(d1) formula for call options.  
'''  
test_predictions = DNN_model.predict(test_features).flatten()  
check = test_features.copy()  
check['c'] = test_labels  
check['c_hat'] = test_predictions  
check['delta'] = norm.cdf((np.log(check["S0"]/check["K"])   
                           + (check['r']+0.5*check["Sigma"]**2)*check["t"])   
                           / (check["Sigma"]*check["t"]**0.5))  
check = check.sort_values(by='S0')  
'''  
Loc the data into 4 time slots for visualization  
'''  
t1 = check.loc[check['t'] == 0.02]  
t2 = check.loc[check['t'] == 0.33]  
t3 = check.loc[check['t'] == 0.67]  
t4 = check.loc[check['t'] == 1.0]  
#Plot loss  
plt.figure(figsize = [10,5])  
plot_loss(history)  

'''  
Prediction Error histogram  
'''  
plt.figure(figsize = [10,5])  
plt.title('Distribution of prediction error', fontsize=16)  
error = test_predictions - test_labels  
plt.hist(error, bins=250, color = 'blue')  
plt.xlabel('Prediction Error')  
plt.savefig('Error distribution.pdf', dpi=1200)  
plt.ylabel('Count');  

'''  
n_plot - number of points in one dimension  
'''  
n_plot = 51  
#%% Heatmap - Neural network model error  
# S_plot - vector of S values for plots  
# t_plot - vector of t values for plots  
S_plot = np.linspace(check['S0'].min(), check['S0'].max(), n_plot)  
t_plot = np.linspace(check['t'].min(), check['t'].max(), n_plot)  
# compute option price for each (t,S) pair  


'''  
Black-Scholes func formula  
'''  
def bs(S, K, t, r, sigma):  
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*t) / (sigma*t**0.5)  
    d2  = d1 - sigma*t**0.5  
    c = S*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)  
    return c  

'''   


Evaluate model quality with MAE, MSE, R2  
'''  
option_value_mesh = np.zeros([n_plot, n_plot])  
for i in range(n_plot):  
    for j in range(n_plot):      
        option_value_mesh[j,i] = bs(S_plot[j], 300.621356, t_plot[i], 0.20, 0.576758)  # compute predicted option price for each (t,S) pair  
fitted_option_value_mesh = np.zeros([n_plot, n_plot])  


for i in range(n_plot):  
    for j in range(n_plot):      
        fitted_option_value_mesh[j,i] = DNN_model.predict([S_plot[j], 300.621356, t_plot[i], 0.20,  
0.576758])  
t_mesh, S_mesh = np.meshgrid(t_plot, S_plot)  
MAE = np.sum(np.abs(test_predictions - test_labels))/n_plot**2  
MSE = np.sum((test_predictions - test_labels)**2)/n_plot**2  
R2 = 1 - np.sum((test_predictions - test_labels)**2)/n_plot**2/np.var(option_value_mesh)  
MSE,MAE,R2  

'''  
Black-Scholes vs Neural network option pricing  
'''  
fig = plt.figure(figsize = [17,17], dpi=400)  

fig.suptitle('Black-Scholes model and Neural network model comparison', fontsize=20, y=0.94)  
plt.subplot(221)  
plt.grid()  
plt.title('t = 0.01', fontsize=16)  
plt.plot(t1['S0'], t1['c'], label='BS solution', color='red', linewidth = 5)  
plt.plot(t1['S0'], t1['c_hat'] ,':', label='Neural Network estimate', c='blue', linewidth = 5)  
plt.xlabel('Stock price (S)')  
plt.ylabel('Option price (c)')  
plt.legend()  
plt.subplot(222)  
plt.grid()  
plt.title('t = 0.33', fontsize=16)  
plt.plot(t2['S0'], t2['c'], label='BS solution', color='red', linewidth = 5)  
plt.plot(t2['S0'], t2['c_hat'],':', label='Neural Network estimate', c='blue', linewidth = 5)  
plt.xlabel('Stock price (S)')  
plt.ylabel('Option price (c)')  
plt.legend()  
plt.subplot(223)  
plt.grid()  
plt.title('t = 0.67', fontsize=16)  
plt.plot(t3['S0'], t3['c'], label='BS solution', color='red', linewidth = 5)  
plt.plot(t3['S0'], t3['c_hat'],':', label='Neural Network estimate', c='blue', linewidth = 5)  
plt.xlabel('Stock price (S)')  
plt.ylabel('Option price (c)')  
plt.legend()  
plt.subplot(224)  
plt.grid()  
plt.title('t = 1.0', fontsize=16)  
plt.plot(t4['S0'], t4['c'], label='BS solution', color='red', linewidth = 5)  
plt.plot(t4['S0'], t4['c_hat'],':', label='Neural Network estimate', c='blue', linewidth = 5)  
plt.xlabel('Stock price (S)')  
plt.ylabel('Option price (c)')  
plt.legend()  
plt.savefig('BSM and DNN comparison.pdf', dpi=1200)  


plt.show()  



'''  
PLOT ABSOLUTE ERROR  
'''   

plt.figure()  
plt.figure(figsize = (8,6))  

plt.pcolormesh(t_mesh, S_mesh, np.abs(option_value_mesh - fitted_option_value_mesh),                  shading='auto', cmap = "rainbow")  


plt.colorbar()  
plt.title("Absolute Error of Neural Network Estimate", fontsize=18)  
plt.ylabel(r"Price $S$", fontsize=15, labelpad=10)  
plt.xlabel(r"Time $t$", fontsize=15, labelpad=20)  
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  
plt.savefig('DNN_abs_err.pdf', dpi=1200)  

# PLOT RELATIVE ERROR  plt.figure()  
plt.figure(figsize = (8,6))  

plt.pcolormesh(t_mesh, S_mesh, np.minimum(np.where(option_value_mesh==0.,1., np.abs(1 - np.divide(fitted_option_value_mesh, option_value_mesh))),1.),   
               shading='auto', cmap = "rainbow")  


plt.colorbar()  
plt.title("Relative Error of Neural Network Estimate", fontsize=20) 
plt.ylabel("Price", fontsize=15, labelpad=10)  
plt.xlabel("Time", fontsize=15, labelpad=20)  
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  
plt.savefig('DNN_rel_err.pdf', dpi=1200)  
plt.show();  
'''  
Analytical option price surface plot  
'''  
fig = plt.figure(figsize=(10, 8))   


ax = fig.add_subplot(111, projection='3d')  


# Plot the surface.  
surf = ax.plot_surface(t_mesh, S_mesh, option_value_mesh, rstride=1, cstride=1, linewidth=0.05, cmap=cm.jet, antialiased=False, edgecolors='gray') # rainbow winter_r coolwarm cool   color='c',  

ax.set_xlabel(r'Time $t$', fontsize=14) # , fontsize=18  ax.set_ylabel(r'Price $S$', fontsize=14)  
ax.set_zlabel(r'Analytical Option Price', fontsize=14)   
ax.set_title(r'Analytical Option Price Surface', fontsize=16)  


# Customize the z axis.  
ax.set_zlim(0., option_value_mesh.max())  
ax.zaxis.set_major_locator(LinearLocator(11))  
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  


ax.view_init(elev = 20, azim = -130)  

# Add a color bar which maps values to colors.  fig.colorbar(surf, shrink=0.5, aspect=10)  surf.set_clim(0., option_value_mesh.max())  fig.tight_layout()  

#fig.set_size_inches(5.5,3.5)  plt.savefig('Analytical_option_price.pdf', dpi=1200)  plt.draw()  
plt.show()  
plt.pause(0.5)  
'''  
DNN option price surface plot  
'''  
fig = plt.figure(figsize=(10, 8))  
ax = fig.add_subplot(111, projection='3d')   



# Plot the surface.  
surf = ax.plot_surface(t_mesh, S_mesh, fitted_option_value_mesh, rstride=1, cstride=1, linewidth=0.05, cmap=cm.jet, antialiased=False, edgecolors='gray') # rainbow winter_r coolwarm cool   color='c',  

ax.set_xlabel(r'Time $t$', fontsize=14) # , fontsize=18  ax.set_ylabel(r'Price $S$', fontsize=14)  
ax.set_zlabel(r'DGM Option Price', fontsize=14)   
ax.set_title(r'Neural Network Option Price Surface', fontsize=16)  


# Customize the z axis.  
ax.set_zlim(fitted_option_value_mesh.min(), fitted_option_value_mesh.max())  
ax.zaxis.set_major_locator(LinearLocator(11))  
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  


ax.view_init(elev = 20, azim = -60)  


# Add a color bar which maps values to colors.  
fig.colorbar(surf, shrink=0.5, aspect=10)  
surf.set_clim(fitted_option_value_mesh.min(), fitted_option_value_mesh.max())  
fig.tight_layout()  

#fig.set_size_inches(5.5,3.5)  plt.savefig('DNN_option_price.pdf', dpi=1200)  plt.draw()  
plt.show()  
plt.pause(0.5)  
'''  
DNN abs error surface plot  
'''  
fig = plt.figure(figsize=(10, 8))  
ax = fig.add_subplot(111, projection='3d')  


# Plot the surface.  
surf = ax.plot_surface(t_mesh, S_mesh, np.abs(option_value_mesh - fitted_option_value_mesh), rstride=1, cstride=1, linewidth=0.05, cmap=cm.jet, antialiased=False,  edgecolors='grey') # rainbow winter_r coolwarm cool   color='c',  


ax.set_xlabel(r'Time $t$', fontsize=14) # , fontsize=18  
ax.set_ylabel(r'Price $S$', fontsize=14)  
ax.set_zlabel(r'Absolute Option Price Error', fontsize=14)   
ax.set_title(r'Absolute Error Surface of Neural Network Option Price Estimate', fontsize=16)  


# Customize the z axis.  
ax.set_zlim(0., (np.abs(option_value_mesh - fitted_option_value_mesh)).max())  
ax.zaxis.set_major_locator(LinearLocator(11))  
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  


ax.view_init(elev = 20, azim = -120)  


# Add a color bar which maps values to colors.  
fig.colorbar(surf, shrink=0.5, aspect=10)  
surf.set_clim(0., (np.abs(option_value_mesh - fitted_option_value_mesh)).max())  
fig.tight_layout()  

#fig.set_size_inches(5.5,3.5)  
plt.savefig('DNN_abs_err_3d.pdf', dpi=1200)  
plt.draw()  
plt.show()  
plt.pause(1.)  
#   
#   
# GRADIENT STARTING  >>>>>>>>>>>>>  
#   
#   
delta = norm.cdf((np.log(test_features["S0"]/test_features["K"])   
                           + (test_features['r']+0.5*test_features["Sigma"]**2)*test_features["t"])    

                           / (test_features["Sigma"]*test_features["t"]**0.5))  
with tf.GradientTape() as tape:  
    s0 = tf.Variable(test_features.S0)  
    K = tf.Variable(test_features.K)  
    t = tf.Variable(test_features.t)  
    r = tf.Variable(test_features.r)  
    sigma = tf.Variable(test_features.Sigma)  
    tape.watch(s0)  
    c_pred = DNN_model(tf.stack([s0,K,t,r,sigma], axis=1))  
    gradients = tape.gradient(c_pred, s0)  
    gradients = np.array(gradients)  


# DELTA - ANALYTICAL SOLUTION  
# GRADIENTS - PARTIAL DERIVATIVE  
delta, gradients  
'''  
Write predicted values in test predict  
and visualize Predictions/True values  
'''  
plt.figure(figsize = [17,7])  
plt.axes(aspect='equal')  
plt.title("Analytical vs Neural model delta coefficient")  
plt.scatter(delta, gradients, color='blue', alpha=0.6, s = 3)  
plt.xlabel('Analytical coefficient')  
plt.ylabel('Model predicted coefficient')  
plt.savefig('DNN_delta.pdf', dpi=1200)  
plt.show()  
'''  
Evaluate delta coeficient quality  
'''  
MAE = np.sum(np.abs(delta - gradients))/n_plot**2  
MSE = np.sum((delta - gradients)**2)/n_plot**2  
R2 = 1 - np.sum((delta - gradients)**2)/n_plot**2/np.var(option_value_mesh)  
MSE,MAE,R2  