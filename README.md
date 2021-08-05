# Dachshund Detector

> You can try out the final model at: [https://doxie-detector.herokuapp.com](https://doxie-detector.herokuapp.com)

Our task is to build a binary classifier to decide from user-submitted dog photos
whether the breed is dachshund or not. We will also deploy the final model to
Heroku with the help of a Flask server. As our dataset we use roughly
1000 pictures of dachshunds and another 1000 of dogs of various other breeds.
The images are scraped from a variety of
sources such as Reddit and DogAPI on [dog.ceo](http://dog.ceo).

It is recommended to view the HTML versions of the notebooks.

## Linear Models

> See [linear_models.ipynb](linear_models.ipynb) for details.
> ([HTML](https://nlaaksonen.github.io/doxie-detector/linear_models.html))

In order to measure our success, we set up a baseline with some simple linear
models (without any hyperparameter tuning). To speed up learning we first
collapse the RGB pictures to greyscale and then apply PCA.
This ends up reducing the dimension of the input vector from
150 528 to 50. We achieve about 60% and 61% classification accuracies on
the validation set with logistic regression and SVMs, respectively.

## Convolutional Neural Nets and Transfer Learning in TensorFlow

> See [cnn_models.ipynb](cnn_models.ipynb) for details. ([HTML](https://nlaaksonen.github.io/doxie-detector/cnn_models.html))

Our main focus are various CNN models. In this notebook we discuss the limitations
of the dataset that we've collected. We then create a classical sequential CNN
and train it from scratch, which gives us an accuracy of ~70%. To improve on this
we do transfer learning by using ResNet and EfficientNet models. Our finished model
obtains roughly a 94% accuracy after fine-tuning. We investigate its behaviour by
computing *class saliency maps* and *class model visualisations* as introduced
by [Simonyan et al. 2013](https://arxiv.org/abs/1312.6034). Class saliency maps
help us visualise which parts of the input contribute the most to the output by
computing the derivative of the output w.r.t. the input.
Here is one example of such a map from our model:

![Saliency map](https://raw.githubusercontent.com/nlaaksonen/doxie-detector/main/img/saliency.png)

Moreover, we show how class saliency maps can be used to construct a very simple
object localisation algorithm:

![Object localisation](https://raw.githubusercontent.com/nlaaksonen/doxie-detector/main/img/localisation.png)

While saliency maps are *input dependent*, class
model visualisations are *input independent*. That is we let the network create
(through backpropagation) a picture from scratch that maximises the class score.
Here are examples of such a visualisation from our model for the dachshund
class:

![Dachund class model visualisation](https://raw.githubusercontent.com/nlaaksonen/doxie-detector/main/img/class_model_dachshund.png)

As you can see, it might be possible to make out some shapes that one might
typically expect from dachshund, but our results are less prominent than in the
original paper. We discuss possible reasons as to why this might happen in
section 5.3 and 5.4 of the notebook. Finally, we'll also briefly show how to
use SHAP and Lime methods to obtain explanations for the predictions of a
TensorFlow model.

