from numpy import gradient
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from data_process import writeData



class Reinforcement_GradientTape:

    def __init__(self,plot_parameters,initial_points,iterations,learning_rate):
        self.xmin,self.xmax,self.ymin,self.ymax,self.N_points = plot_parameters
        self.initial_pointx,self.initial_pointy = initial_points
        self.iterations = iterations
        self.learning_rate = learning_rate
        

    
    def F(self,x,y):
        #Due to the structure of the grid
        P = np.zeros((len(y),len(x)))
        for i in range(len(y)):
            for j in range(len(x)):
                P[i][j] =1- ( 4*tf.math.exp(-( (x[j]-0.5)**2 + (y[i]-0.7)**2 ) ) 
                              + 6*tf.math.exp(-( (x[j]-2.4)**2 + (y[i]-4)**2 ) )
                            )
        return P


    def Plot_Results(self,x_plot,y_plot,x_progress,y_progress):
        P, P_progress = self.F(x_plot,y_plot),self.F(x_progress,y_progress)
        #3D plot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(x_plot, y_plot, P,cmap='viridis_r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        #2D heatmap
        #extent = [left,rigth,bottom,top]
        plt.imshow(P,origin = 'upper',extent=[self.xmin,self.xmax,self.ymax,self.ymin])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.scatter(x_progress,y_progress,c='k')
        plt.show()

    def ExecuteTwoVariables(self):
        x_plot,y_plot = np.linspace(self.xmin,self.xmax,self.N_points),np.linspace(self.ymin,self.ymax,self.N_points)
        
        x,y = tf.Variable(self.initial_pointx,trainable=True),tf.Variable(self.initial_pointy,trainable=True)
        x_progress,y_progress,loss = [],[],[]
        counter = 0
        for k in range(self.iterations):
            print('\rIteration {} out of {}'.format(k+1,self.iterations),end='')
            with tf.GradientTape() as tape:
                #tape.watch([x,y])
                P =1- ( 4*tf.math.exp(-( (x-0.5)**2 + (y-0.7)**2 ) ) 
                    + 6*tf.math.exp(-( (x-2.4)**2 + (y-4)**2 ) )
                    )
                
                Gradient = tape.gradient(P, [x,y])
                if k%100 == 0:
                    x_progress.append(x.numpy())
                    y_progress.append(y.numpy())
            x.assign(x-self.learning_rate*Gradient[0].numpy())
            y.assign(y-self.learning_rate*Gradient[1].numpy())
            

        self.Plot_Results(x_plot,y_plot,x_progress,y_progress)
        print('\nx:{}, y:{}'.format(x_progress[len(x_progress)-1],y_progress[len(y_progress)-1]))

trial1 = Reinforcement_GradientTape(plot_parameters=[-2,5,-2,5,500],initial_points=[1.5,1.5],iterations=20000,learning_rate=0.5)

trial1.ExecuteTwoVariables()





