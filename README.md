# Kernel-Support-Vector-Machine
In this project I tried various Kernels and used the most suited one of Support Vector Machine , both in Python and R

# We use Kernel SVM when the data is not linearly separable 
### Basically we project our problem in higher dimension and then we use our normal SVM classification and then to project the data in 2D , we re map into 2D .

We should remember that it is very computation intensive to project something into higher dimension 

### So we use the [Gaussian RBF Kernel](http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex8/ex8.html)

![down](http://image.ibb.co/iXy3Ww/RBF_KERNEL.png)

The projection in higher dimension looks something like this :

![down](http://www.cs.toronto.edu/~duvenaud/cookbook/multidimensional_kernels/Sqaured-exp%20kernel%20in%202d.png)

# We can add up two kernel functions :

K(x,l1) + K(x,l2)


### The types of Kernel we can use are (sorry , for my bad handwritting ) :

![down](https://scontent.fccu3-1.fna.fbcdn.net/v/t1.0-9/26992744_185113902235823_8631901595683109275_n.jpg?oh=92858bf69c230770b8f900264686ccec&oe=5AE255E9)

This is one of the most useful method of Classification
